"""
FULL CLOSED-LOOP (GA -> bed_wall.dep -> run XBeach -> compute ΔV -> append Trial -> repeat)

THIS VERSION INCLUDES EVERYTHING WE DISCUSSED:
1) THREE windows computed each run (FULL alongshore):
   - MAIN  : 30 m seaward + includes the 5 m wall thickness
   - TOE   : 0–10 m seaward of wall face
   - OUTER : 10–60 m seaward of wall face

2) "0 erosion" auto-fix for sign convention:
   - If BOTH TOE and OUTER erosion come out 0 AND dz never goes negative,
     the code also tries flipping posdwn just for ΔV calculation and uses the better one.
   - This catches cases where params posdwn doesn't match how bed/zb are actually signed.

3) GA objective update (solution for persistent 0 erosion):
   - By default, GA minimizes erosion in BOTH windows equally:
       objective = 0.5*Ero_toe + 0.5*Ero_outer
     (This is a better "sand regression" target than |NetΔV|.)

4) Trial numbering bug fix:
   - next_trial_number() scans the file with a robust encoding reader (UTF-8/UTF-16),
     so it won't restart at Trial 1 if the text file encoding changes.

5) Lock file:
   - Prevents running two loops at once in the same folder.

------------------------------------------------------------
EDIT ONLY THESE TOP SETTINGS:
------------------------------------------------------------
"""

import os
import re
import struct
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import numpy as np

# ============================================================
# USER EDITS
# ============================================================
RUN_DIR = Path(r"C:\Users\brend\Desktop\sf_1yr_case")

# XBeach now under Documents (search recursively for xbeach.exe)
XBEACH_BASE = Path(os.path.join(os.path.expanduser("~"), "Documents"))

TARGET_SUCCESSFUL_TRIALS = 380
MAX_CRASH_RETRIES_PER_TRIAL = 5

# GA objective mode:
#   "erosion"  -> minimize erosion volumes (recommended for sand regression)
#   "absnet"   -> minimize 0.5*|Net_toe| + 0.5*|Net_outer|
GA_OBJECTIVE_MODE = "erosion"

# ============================================================
# CONSTANTS / FILE NAMES
# ============================================================
PARAMS_NAME = "params.txt"
TRIALS_TXT_NAME = "deltaV_trials.txt"
CSV_NAME = "wall_control_points.csv"
FOUNDATION_CSV_NAME = "wall_foundation_points.csv"
BED_WALL_NAME = "bed_wall.dep"

BASE_BED_CANDIDATES = ["bed_fliplr.dep", "bed.dep"]  # no-wall baseline bed files

# Wall / windows
WALL_THICKNESS_M = 5.0

WINDOW_MAIN_SEAWARD_M = 30.0  # MAIN window: 30 m seaward + include wall thickness

TOE_WIN_A = 0.0
TOE_WIN_B = 10.0

OUTER_WIN_A = 10.0
OUTER_WIN_B = 60.0

# Wall control points
NPTS = 10

# Hard constraints
TOE_ABOVE_SAND_M = 0.5
FOUNDATION_DEPTH_BELOW_SAND_M = 3.0
CREST_MIN = 2.5
CREST_MAX = 3.5

# GA settings
P_TWO_SEGMENT = 0.35
ELITE_K = 8
MUTATION_SIGMA_Z = 0.25
MUTATION_PROB_Z = 0.45

# Soft realism thresholds + weights (penalties only)
SLOPE_SOFT_SMOOTH = 4.0
CURV_SOFT_SMOOTH = 8.0
SLOPE_SOFT_2SEG = 12.0

PEN_W_CREST = 5_000.0
PEN_W_SLOPE = 1_500.0
PEN_W_CURV = 900.0
PEN_W_WAVY = 700.0
PEN_W_KINK = 600.0

# Crash detection
CRASH_PHRASES = [
    "halt_program",
    "time constrain criterium exceeded",
    "computational time implodes/explodes",
    "quit xbeach since computational time implodes/explodes",
    "traceback:",
]

# ============================================================
# SMALL HELPERS
# ============================================================
def die(msg: str):
    raise SystemExit(msg)

def snap_to_grid(x_m: float, dx: float) -> float:
    return float(round(x_m / dx) * dx)

def _strip_comments(s: str) -> str:
    for c in ("#", "!"):
        if c in s:
            s = s.split(c, 1)[0]
    return s.strip()

def read_text_auto(path: Path) -> str:
    """Robust: reads UTF-8 or UTF-16 (Notepad sometimes saves UTF-16)."""
    b = path.read_bytes()
    if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
        return b.decode("utf-16", errors="ignore")
    if len(b) > 0 and (b.count(b"\x00") > 0.2 * len(b)):
        return b.decode("utf-16-le", errors="ignore")
    return b.decode("utf-8", errors="ignore")

def read_params(params_path: Path) -> dict:
    params = {}
    with open(params_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = _strip_comments(line)
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = _strip_comments(v).strip()
            if not v:
                continue
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            try:
                if v.isdigit() or (v.startswith(("+", "-")) and v[1:].isdigit()):
                    params[k] = int(v)
                else:
                    params[k] = float(v)
            except Exception:
                params[k] = v
    return params

def ensure_depfile(params_path: Path, depfile_name: str):
    text = params_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out = []
    found = False
    for line in text:
        if re.match(r"^\s*depfile\s*=", line, flags=re.IGNORECASE):
            out.append(f"depfile      = {depfile_name}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"depfile      = {depfile_name}")
    params_path.write_text("\n".join(out) + "\n", encoding="utf-8")

def read_bed_nodes(dep_path: Path, ny: int, nx: int) -> np.ndarray:
    nrows, ncols = ny + 1, nx + 1
    data = np.loadtxt(dep_path, dtype=float)
    if data.ndim == 2 and data.shape == (nrows, ncols):
        return data
    flat = np.array(data, dtype=float).ravel()
    if flat.size == nrows * ncols:
        return flat.reshape((nrows, ncols), order="C")
    raise ValueError(f"{dep_path} shape {getattr(data,'shape',None)}; expected {(nrows, ncols)}")

def write_bed_nodes(dep_path: Path, arr: np.ndarray):
    np.savetxt(dep_path, arr, fmt="%.6f")

def to_pos_up(arr: np.ndarray, posdwn: float) -> np.ndarray:
    return -arr if posdwn >= 0 else arr

def from_pos_up(arr_up: np.ndarray, posdwn: float) -> np.ndarray:
    return -arr_up if posdwn >= 0 else arr_up

def find_xbeach_exe(base: Path) -> Path:
    if base.is_file() and base.name.lower() == "xbeach.exe":
        return base
    candidates = list(base.rglob("xbeach.exe"))
    if not candidates:
        die(f"Could not find xbeach.exe under: {base}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

def newest_file_matching(root: Path, pattern: str, min_mtime: Optional[float] = None) -> Optional[Path]:
    files = list(root.rglob(pattern))
    if min_mtime is not None:
        files = [p for p in files if p.exists() and p.stat().st_mtime >= min_mtime]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

# ============================================================
# TRIALS PARSE / APPEND
# ============================================================
TRIAL_RE = re.compile(r"\bTrial\s+(\d+)\b", re.IGNORECASE)

def parse_bracket_list(s: str) -> np.ndarray:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected bracket list like [..], got: {s[:80]}")
    inner = s[1:-1].strip()
    if not inner:
        return np.array([], dtype=float)
    return np.array([float(p.strip()) for p in inner.split(",")], dtype=float)

def fmt_points(arr: np.ndarray, ndp: int = 3) -> str:
    return "[" + ",".join(f"{v:.{ndp}f}" for v in arr) + "]"

def next_trial_number(trials_path: Path) -> int:
    if not trials_path.exists():
        return 1
    txt = read_text_auto(trials_path)
    nums = [int(n) for n in re.findall(r"\bTrial\s+(\d+)\b", txt, flags=re.IGNORECASE)]
    return 1 if not nums else (max(nums) + 1)

def load_trials(trials_path: Path) -> List[dict]:
    """
    This script writes the NEW format with toe+outer windows.
    It can still read old lines but may not extract toe/out values from them.
    """
    if not trials_path.exists():
        return []
    text = read_text_auto(trials_path)
    lines = text.splitlines()

    trials: List[dict] = []
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("trial |"):
            continue
        m = TRIAL_RE.search(line)
        if not m:
            continue

        parts = [p.strip() for p in line.split("|")]
        # Expect at least up to z_points
        if len(parts) < 13:
            continue

        try:
            trial_n = int(m.group(1))
            # NEW format positions:
            # ... | Net_main | Dep_main | Ero_main | x_pts | z_pts | [toe_bounds] | Net_toe | Dep_toe | Ero_toe | [out_bounds] | Net_out | Dep_out | Ero_out
            net_main = float(parts[8])
            dep_main = float(parts[9])
            ero_main = float(parts[10])
            x_pts = parse_bracket_list(parts[11])
            z_pts = parse_bracket_list(parts[12])
        except Exception:
            continue

        if x_pts.size != NPTS or z_pts.size != NPTS:
            continue

        rec = {
            "trial": trial_n,
            "x_pts": x_pts,
            "z_pts": z_pts,
            "net_toe": None, "dep_toe": None, "ero_toe": None,
            "net_out": None, "dep_out": None, "ero_out": None,
            "net_main": net_main, "dep_main": dep_main, "ero_main": ero_main,
        }

        # Try to parse toe/out if present
        if len(parts) >= 22:
            try:
                rec["net_toe"] = float(parts[14])
                rec["dep_toe"] = float(parts[15])
                rec["ero_toe"] = float(parts[16])
                rec["net_out"] = float(parts[18])
                rec["dep_out"] = float(parts[19])
                rec["ero_out"] = float(parts[20])
            except Exception:
                pass

        # If missing, fall back to main (so GA can still work)
        if rec["ero_toe"] is None: rec["ero_toe"] = ero_main
        if rec["ero_out"] is None: rec["ero_out"] = ero_main
        if rec["net_toe"] is None: rec["net_toe"] = net_main
        if rec["net_out"] is None: rec["net_out"] = net_main

        trials.append(rec)

    return trials

def append_trial(
    trials_path: Path,
    run_dir: Path,
    nx: int, ny: int, dx: float, dy: float, posdwn_used_for_calc: float,
    x_wall_face: float,
    x_main_min: float, x_main_max: float,
    net_main: float, dep_main: float, ero_main: float,
    x_pts: np.ndarray, z_pts: np.ndarray,
    x_toe_min: float, x_toe_max: float, net_toe: float, dep_toe: float, ero_toe: float,
    x_out_min: float, x_out_max: float, net_out: float, dep_out: float, ero_out: float,
) -> int:
    trial_n = next_trial_number(trials_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = (
        "Trial | timestamp | run_dir | nx ny dx dy posdwn_used_for_calc | wall_face_x_m | x_window_main_m | "
        "Net_main_m3 | Dep_main_m3 | Ero_main_m3 | x_points_m | z_points_m | "
        "x_window_toe_m | Net_toe_m3 | Dep_toe_m3 | Ero_toe_m3 | "
        "x_window_outer_m | Net_outer_m3 | Dep_outer_m3 | Ero_outer_m3\n"
    )

    line = (
        f"Trial {trial_n} | {timestamp} | {run_dir} | "
        f"{nx} {ny} {dx:.6g} {dy:.6g} {posdwn_used_for_calc:.6g} | "
        f"{x_wall_face:.3f} | [{x_main_min:.3f}, {x_main_max:.3f}] | "
        f"{net_main:.6e} | {dep_main:.6e} | {ero_main:.6e} | "
        f"{fmt_points(x_pts)} | {fmt_points(z_pts)} | "
        f"[{x_toe_min:.3f}, {x_toe_max:.3f}] | {net_toe:.6e} | {dep_toe:.6e} | {ero_toe:.6e} | "
        f"[{x_out_min:.3f}, {x_out_max:.3f}] | {net_out:.6e} | {dep_out:.6e} | {ero_out:.6e}\n"
    )

    needs_header = (not trials_path.exists()) or (trials_path.stat().st_size == 0)
    with open(trials_path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write(header)
        f.write(line)
    return trial_n

# ============================================================
# CONTROL POINTS CSV READ/WRITE
# ============================================================
def read_control_points_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    order = np.argsort(data["id"])
    x = np.array(data["x_from_left_m"], dtype=float)[order]
    z = np.array(data["z_m"], dtype=float)[order]
    return x, z

def write_control_points_csv(csv_path: Path, x_pts: np.ndarray, z_pts: np.ndarray):
    lines = ["id,z_m,x_from_left_m"]
    for i, (x, z) in enumerate(zip(x_pts, z_pts)):
        lines.append(f"{i},{z:.6f},{x:.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_foundation_points_csv(csv_path: Path, x_pts: np.ndarray, z_base: np.ndarray):
    lines = ["id,z_base_m,x_from_left_m"]
    for i, (x, zb) in enumerate(zip(x_pts, z_base)):
        lines.append(f"{i},{zb:.6f},{x:.6f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ============================================================
# WALL LOCATION + OFFSHORE SIDE
# ============================================================
def infer_offshore_side_from_bed(bed_up_nodes: np.ndarray) -> str:
    left = float(np.nanmean(bed_up_nodes[:, 0]))
    right = float(np.nanmean(bed_up_nodes[:, -1]))
    return "left" if left < right else "right"

def make_x_pts_from_face(x_wall_face: float, offshore_side: str) -> np.ndarray:
    landward_sign = +1.0 if offshore_side == "left" else -1.0
    x_landward = x_wall_face + landward_sign * WALL_THICKNESS_M
    return np.linspace(x_wall_face, x_landward, NPTS)

# ============================================================
# GA: penalties + objective
# ============================================================
def crest_penalty(z_crest: float) -> float:
    below = max(0.0, CREST_MIN - z_crest)
    above = max(0.0, z_crest - CREST_MAX)
    return PEN_W_CREST * (below*below + above*above)

def slope_penalty(x: np.ndarray, z: np.ndarray, slope_soft: float) -> float:
    s = np.diff(z) / np.maximum(np.diff(x), 1e-9)
    excess = np.maximum(0.0, np.abs(s) - slope_soft)
    return PEN_W_SLOPE * float(np.sum(excess**2))

def curvature_penalty(x: np.ndarray, z: np.ndarray) -> float:
    dxm = float(np.mean(np.diff(x)))
    if dxm <= 1e-9:
        return 1e9
    d2 = (z[2:] - 2*z[1:-1] + z[:-2]) / (dxm*dxm)
    excess = np.maximum(0.0, np.abs(d2) - CURV_SOFT_SMOOTH)
    return PEN_W_CURV * float(np.sum(excess**2))

def waviness_penalty(x: np.ndarray, z: np.ndarray) -> float:
    s = np.diff(z) / np.maximum(np.diff(x), 1e-9)
    signs = np.sign(s)
    changes = np.sum(signs[1:] * signs[:-1] < 0)
    return PEN_W_WAVY * float(max(0, changes - 1)**2)

def kink_penalty(z: np.ndarray) -> float:
    if z.size < 3:
        return 0.0
    mid = z[1:-1]
    span = float(mid.max() - mid.min())
    return PEN_W_KINK * float(max(0.0, span - 2.5)**2)

def penalty_smooth(x: np.ndarray, z: np.ndarray) -> float:
    return (
        crest_penalty(float(z[-1])) +
        slope_penalty(x, z, SLOPE_SOFT_SMOOTH) +
        curvature_penalty(x, z) +
        waviness_penalty(x, z)
    )

def penalty_two_segment(x: np.ndarray, z: np.ndarray) -> float:
    return crest_penalty(float(z[-1])) + slope_penalty(x, z, SLOPE_SOFT_2SEG) + kink_penalty(z)

def fitness_from_trial(trial: dict) -> float:
    """
    GA objective (solution for persistent "0 erosion"):
      - default: minimize erosion volumes in toe + outer equally
      - optional: minimize abs(net) in toe + outer equally
    """
    if GA_OBJECTIVE_MODE.lower() == "absnet":
        base = 0.5 * abs(float(trial["net_toe"])) + 0.5 * abs(float(trial["net_out"]))
    else:
        base = 0.5 * float(trial["ero_toe"]) + 0.5 * float(trial["ero_out"])
    return base + penalty_smooth(trial["x_pts"], trial["z_pts"])

def smooth_filter(z: np.ndarray, strength: float = 0.22) -> np.ndarray:
    z2 = z.copy()
    for i in range(1, len(z)-1):
        z2[i] = (1-strength)*z[i] + strength*0.5*(z[i-1] + z[i+1])
    return z2

def mutate_z(z: np.ndarray) -> np.ndarray:
    z = z.copy()
    for i in range(1, len(z)-1):
        if np.random.rand() < MUTATION_PROB_Z:
            z[i] += np.random.normal(0.0, MUTATION_SIGMA_Z)
    if np.random.rand() < 0.7:
        z[-1] += np.random.normal(0.0, MUTATION_SIGMA_Z * 0.8)
    return z

def crossover_z(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    child = z1.copy()
    for i in range(1, len(z1)-1):
        child[i] = z2[i] if (np.random.rand() < 0.5) else (0.7*z1[i] + 0.3*z2[i])
    child[-1] = 0.5*z1[-1] + 0.5*z2[-1]
    return child

def generate_two_segment(toe_z: float) -> np.ndarray:
    crest = float(np.random.uniform(CREST_MIN, CREST_MAX))
    f = float(np.random.uniform(0.15, 0.85))
    zb = toe_z + (crest - toe_z) * float(np.random.uniform(0.15, 0.95))
    t = np.linspace(0.0, 1.0, NPTS)
    z = np.empty_like(t)
    for i, ti in enumerate(t):
        if ti <= f:
            z[i] = toe_z + (zb - toe_z) * (ti / max(f, 1e-9))
        else:
            z[i] = zb + (crest - zb) * ((ti - f) / max(1.0 - f, 1e-9))
    z[0] = toe_z
    z[-1] = crest
    return z

def enforce_hard_constraints(z: np.ndarray, toe_z: float) -> np.ndarray:
    z = z.copy()
    z[0] = toe_z
    z[-1] = float(np.clip(z[-1], CREST_MIN, CREST_MAX))
    return z

def propose_next_wall_z(trials: List[dict], current_z: np.ndarray, toe_z: float, x_pts_fixed: np.ndarray) -> Tuple[np.ndarray, str, float]:
    if not trials:
        z = smooth_filter(mutate_z(current_z))
        z = enforce_hard_constraints(z, toe_z)
        return z, "bootstrap-mutate", penalty_smooth(x_pts_fixed, z)

    for t in trials:
        t["fit"] = fitness_from_trial(t)

    trials_sorted = sorted(trials, key=lambda tr: tr["fit"])
    elites = trials_sorted[:max(2, min(ELITE_K, len(trials_sorted)))]

    weights = np.linspace(1.0, 0.4, len(elites))
    weights = weights / weights.sum()
    a, b = np.random.choice(len(elites), size=2, replace=False, p=weights)
    p1, p2 = elites[a], elites[b]

    two_seg = (np.random.rand() < P_TWO_SEGMENT)
    if two_seg:
        z = generate_two_segment(toe_z)
        z = enforce_hard_constraints(z, toe_z)
        return z, "two-segment", penalty_two_segment(x_pts_fixed, z)
    else:
        z = crossover_z(p1["z_pts"], p2["z_pts"])
        z = smooth_filter(mutate_z(z))
        z = enforce_hard_constraints(z, toe_z)
        return z, "smooth", penalty_smooth(x_pts_fixed, z)

# ============================================================
# BUILD bed_wall.dep FROM BASE BED + WALL TOP SURFACE
# ============================================================
def build_bed_wall_from_base(
    base_bed_path: Path,
    out_bed_wall_path: Path,
    ny: int, nx: int, dx: float, posdwn: float,
    x_pts: np.ndarray, z_top: np.ndarray
):
    bed_raw = read_bed_nodes(base_bed_path, ny, nx)
    bed_up = to_pos_up(bed_raw, posdwn)

    x_nodes = np.arange(nx + 1) * dx

    if x_pts[0] <= x_pts[-1]:
        z_wall_nodes = np.interp(x_nodes, x_pts, z_top, left=z_top[0], right=z_top[-1])
        mask = (x_nodes >= x_pts[0] - 1e-9) & (x_nodes <= x_pts[-1] + 1e-9)
    else:
        xr = x_pts[::-1]
        zr = z_top[::-1]
        z_wall_nodes = np.interp(x_nodes, xr, zr, left=zr[0], right=zr[-1])
        mask = (x_nodes >= xr[0] - 1e-9) & (x_nodes <= xr[-1] + 1e-9)

    cols = np.where(mask)[0]
    if cols.size == 0:
        raise ValueError("Wall footprint selected zero x-columns (check x_pts and dx).")

    bed_up_new = bed_up.copy()
    for ix in cols:
        bed_up_new[:, ix] = np.maximum(bed_up_new[:, ix], z_wall_nodes[ix])

    bed_out = from_pos_up(bed_up_new, posdwn)
    write_bed_nodes(out_bed_wall_path, bed_out)

# ============================================================
# RUN XBEACH
# ============================================================
def run_xbeach(xbeach_exe: Path, run_dir: Path) -> Tuple[bool, Path]:
    log_path = run_dir / "XBlog_auto.txt"
    start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== XBeach run started: {start} =====\n")
        log.write(f"EXE: {xbeach_exe}\nCWD: {run_dir}\n")

    proc = subprocess.Popen(
        [str(xbeach_exe)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    crashed = False
    with open(log_path, "a", encoding="utf-8", errors="ignore") as log:
        for line in proc.stdout:
            log.write(line)
            print(line.rstrip())
            low = line.lower()
            if any(p in low for p in CRASH_PHRASES):
                crashed = True

    rc = proc.wait()
    if rc != 0:
        crashed = True

    end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"===== XBeach run ended: {end}  returncode={rc}  crashed={crashed} =====\n")

    return (not crashed), log_path

# ============================================================
# READ zb.dat (RAW float64 fields preferred, Fortran sequential fallback)
# ============================================================
class FortranRecordError(Exception):
    pass

def iter_fortran_records_safe(f, marker_size: int, endian: str, max_reclen: int):
    fmt = endian + ("i" if marker_size == 4 else "q")
    mlen = marker_size
    while True:
        head = f.read(mlen)
        if not head:
            return
        if len(head) != mlen:
            raise FortranRecordError("truncated record header")
        (reclen,) = struct.unpack(fmt, head)
        if reclen < 0 or reclen > max_reclen:
            raise FortranRecordError("bad record length")
        payload = f.read(reclen)
        if len(payload) != reclen:
            raise FortranRecordError("truncated record data")
        tail = f.read(mlen)
        if len(tail) != mlen:
            raise FortranRecordError("truncated record trailer")
        (reclen2,) = struct.unpack(fmt, tail)
        if reclen2 != reclen:
            raise FortranRecordError("record length mismatch")
        yield payload

def read_last_zb_nodes(zb_path: Path, ny: int, nx: int, baseline_nodes: np.ndarray) -> Tuple[np.ndarray, str]:
    nrows, ncols = ny + 1, nx + 1
    nvals = nrows * ncols
    bytes_per_field = nvals * 8

    size = zb_path.stat().st_size

    # 1) RAW float64 fields
    if size % bytes_per_field == 0:
        data = np.fromfile(zb_path, dtype=np.float64)
        if data.size % nvals == 0:
            buf = data[-nvals:]
            candidates = [
                ("RAW f64 order=C", buf.reshape((nrows, ncols), order="C")),
                ("RAW f64 order=F", buf.reshape((nrows, ncols), order="F")),
            ]
            best = None
            mask0 = np.isfinite(baseline_nodes)
            for name, arr in candidates:
                mask = mask0 & np.isfinite(arr)
                if mask.mean() < 0.2:
                    continue
                a = baseline_nodes[mask].ravel()
                b = arr[mask].ravel()
                if a.size < 500:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                score = abs(corr)
                if best is None or score > best[0]:
                    best = (score, name, arr)
            if best is not None:
                return best[2], best[1]

    # 2) Fortran sequential fallback
    max_reclen = max(4_194_304, 64 * (nvals + 16) * 8)
    combos = [(4, "<"), (4, ">"), (8, "<"), (8, ">")]
    floats = ["f4", "f8"]
    for marker_size, endian in combos:
        for fcode in floats:
            dtype = np.dtype(endian + fcode)
            last_field = None
            try:
                with open(zb_path, "rb") as f:
                    for payload in iter_fortran_records_safe(f, marker_size, endian, max_reclen):
                        if len(payload) % dtype.itemsize != 0:
                            continue
                        nf = len(payload) // dtype.itemsize
                        if nf < nvals:
                            continue
                        arr = np.frombuffer(payload, dtype=dtype)
                        if nf == nvals:
                            field = arr.reshape((nrows, ncols), order="F")
                        elif nf == nvals + 1:
                            field = arr[1:].reshape((nrows, ncols), order="F")
                        elif (nf - nvals) <= 16:
                            field = arr[-nvals:].reshape((nrows, ncols), order="F")
                        else:
                            continue
                        last_field = field
                if last_field is not None:
                    return last_field.astype(float), f"Fortran seq {marker_size}-byte {'LE' if endian=='<' else 'BE'} {fcode}"
            except Exception:
                continue

    raise ValueError("Could not parse zb.dat as RAW float64 fields or Fortran sequential records.")

# ============================================================
# ΔV COMPUTE (with sign auto-fix)
# ============================================================
def dz_cell_from_nodes(bed0_up: np.ndarray, zbf_up: np.ndarray) -> np.ndarray:
    dz_nodes = zbf_up - bed0_up
    return 0.25 * (dz_nodes[:-1, :-1] + dz_nodes[1:, :-1] + dz_nodes[:-1, 1:] + dz_nodes[1:, 1:])

def window_volumes(cell_vol: np.ndarray, xcell: np.ndarray, x_min: float, x_max: float) -> Tuple[float, float, float]:
    mx = (xcell >= x_min) & (xcell < x_max)
    sub = cell_vol[:, mx]
    net = float(np.nansum(sub))
    dep = float(np.nansum(np.clip(sub, 0.0, np.inf)))
    ero = float(np.nansum(np.clip(-sub, 0.0, np.inf)))
    return net, dep, ero

def seaward_window_bounds(x_wall_face: float, offshore_side: str, nx: int, dx: float, a_m: float, b_m: float) -> Tuple[float, float]:
    landward_sign = +1.0 if offshore_side == "left" else -1.0
    x1 = x_wall_face - landward_sign * b_m
    x2 = x_wall_face - landward_sign * a_m
    Lx = nx * dx
    x_min = max(0.0, min(x1, x2))
    x_max = min(Lx, max(x1, x2))
    return snap_to_grid(x_min, dx), snap_to_grid(x_max, dx)

def compute_windows_with_posdwn(
    bed0_raw: np.ndarray,
    zbf_raw: np.ndarray,
    nx: int, ny: int, dx: float, dy: float,
    posdwn_try: float,
    x_wall_face: float,
    offshore_side: str,
) -> Dict[str, object]:
    bed0_up = to_pos_up(bed0_raw, posdwn_try)
    zbf_up  = to_pos_up(zbf_raw,  posdwn_try)

    dz_up_cell = dz_cell_from_nodes(bed0_up, zbf_up)
    cell_vol = dz_up_cell * (dx * dy)
    xcell = (np.arange(nx) + 0.5) * dx

    # toe & outer bounds
    x_toe_min, x_toe_max = seaward_window_bounds(x_wall_face, offshore_side, nx, dx, TOE_WIN_A, TOE_WIN_B)
    x_out_min, x_out_max = seaward_window_bounds(x_wall_face, offshore_side, nx, dx, OUTER_WIN_A, OUTER_WIN_B)

    net_toe, dep_toe, ero_toe = window_volumes(cell_vol, xcell, x_toe_min, x_toe_max)
    net_out, dep_out, ero_out = window_volumes(cell_vol, xcell, x_out_min, x_out_max)

    # check whether dz ever goes negative in either window
    mx_toe = (xcell >= x_toe_min) & (xcell < x_toe_max)
    mx_out = (xcell >= x_out_min) & (xcell < x_out_max)
    dz_toe = dz_up_cell[:, mx_toe]
    dz_out = dz_up_cell[:, mx_out]
    dzmin = float(np.nanmin([np.nanmin(dz_toe), np.nanmin(dz_out)]))
    dzmax = float(np.nanmax([np.nanmax(dz_toe), np.nanmax(dz_out)]))

    return {
        "posdwn_used": posdwn_try,
        "x_toe_min": x_toe_min, "x_toe_max": x_toe_max,
        "x_out_min": x_out_min, "x_out_max": x_out_max,
        "net_toe": net_toe, "dep_toe": dep_toe, "ero_toe": ero_toe,
        "net_out": net_out, "dep_out": dep_out, "ero_out": ero_out,
        "dzmin": dzmin, "dzmax": dzmax,
    }

def compute_and_log_deltaV(
    run_dir: Path,
    nx: int, ny: int, dx: float, dy: float, posdwn_param: float,
    x_pts: np.ndarray, z_pts: np.ndarray,
    zb_path: Path,
    trials_path: Path,
) -> int:
    bed_wall_path = run_dir / BED_WALL_NAME
    bed0_raw = read_bed_nodes(bed_wall_path, ny, nx)

    zbf_raw, fmt = read_last_zb_nodes(zb_path, ny, nx, bed0_raw)
    print(f"Parsed zb using: {fmt}")

    # offshore side from posdwn_param bed interpretation (good enough to set left/right)
    bed0_up_for_side = to_pos_up(bed0_raw, posdwn_param)
    offshore_side = infer_offshore_side_from_bed(bed0_up_for_side)

    x_wall_face = float(x_pts[0])

    # MAIN window bounds (depends on seaward direction)
    landward_sign = +1.0 if offshore_side == "left" else -1.0
    x_seaward_start = x_wall_face - landward_sign * WINDOW_MAIN_SEAWARD_M
    x_landward_end  = x_wall_face + landward_sign * WALL_THICKNESS_M
    x_main_min = snap_to_grid(max(0.0, min(x_seaward_start, x_landward_end)), dx)
    x_main_max = snap_to_grid(min(nx * dx, max(x_seaward_start, x_landward_end)), dx)

    # --- Sign-auto-fix: try params posdwn first ---
    r1 = compute_windows_with_posdwn(bed0_raw, zbf_raw, nx, ny, dx, dy, posdwn_param, x_wall_face, offshore_side)

    # If erosion is 0 in BOTH windows AND dz never goes negative, try flipping
    use_flip = (r1["ero_toe"] == 0.0 and r1["ero_out"] == 0.0 and r1["dzmin"] >= -1e-9)
    posdwn_used = posdwn_param
    r_used = r1

    if use_flip:
        r2 = compute_windows_with_posdwn(bed0_raw, zbf_raw, nx, ny, dx, dy, -posdwn_param, x_wall_face, offshore_side)
        # If flipped produces any negative dz or any erosion, use it
        if (r2["dzmin"] < -1e-9) or (r2["ero_toe"] > 0.0) or (r2["ero_out"] > 0.0):
            print("WARNING: Erosion was 0 with posdwn from params; flipping sign convention for ΔV calculation.")
            posdwn_used = -posdwn_param
            r_used = r2

    # Now compute MAIN window volumes under the chosen posdwn_used
    bed0_up = to_pos_up(bed0_raw, posdwn_used)
    zbf_up  = to_pos_up(zbf_raw,  posdwn_used)
    dz_up_cell = dz_cell_from_nodes(bed0_up, zbf_up)
    cell_vol = dz_up_cell * (dx * dy)
    xcell = (np.arange(nx) + 0.5) * dx
    net_main, dep_main, ero_main = window_volumes(cell_vol, xcell, x_main_min, x_main_max)

    # Toe / Outer results from r_used
    x_toe_min = float(r_used["x_toe_min"]); x_toe_max = float(r_used["x_toe_max"])
    x_out_min = float(r_used["x_out_min"]); x_out_max = float(r_used["x_out_max"])
    net_toe = float(r_used["net_toe"]); dep_toe = float(r_used["dep_toe"]); ero_toe = float(r_used["ero_toe"])
    net_out = float(r_used["net_out"]); dep_out = float(r_used["dep_out"]); ero_out = float(r_used["ero_out"])

    print("\nΔV results (m³), positive-up convention (baseline=bed_wall.dep):")
    print(f"  MAIN  [{x_main_min:.1f},{x_main_max:.1f}] m: Net={net_main: .6e} Dep={dep_main: .6e} Ero={ero_main: .6e}")
    print(f"  TOE   [{x_toe_min:.1f},{x_toe_max:.1f}] m: Net={net_toe: .6e} Dep={dep_toe: .6e} Ero={ero_toe: .6e}")
    print(f"  OUTER [{x_out_min:.1f},{x_out_max:.1f}] m: Net={net_out: .6e} Dep={dep_out: .6e} Ero={ero_out: .6e}")
    print(f"  DEBUG dzmin={float(r_used['dzmin']): .6e} m, dzmax={float(r_used['dzmax']): .6e} m  (posdwn_used_for_calc={posdwn_used})")

    trial_n = append_trial(
        trials_path, run_dir,
        nx, ny, dx, dy, posdwn_used,
        x_wall_face,
        x_main_min, x_main_max,
        net_main, dep_main, ero_main,
        x_pts, z_pts,
        x_toe_min, x_toe_max, net_toe, dep_toe, ero_toe,
        x_out_min, x_out_max, net_out, dep_out, ero_out,
    )
    print(f"\nAppended Trial {trial_n} to: {trials_path}")
    return trial_n

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    if not RUN_DIR.exists():
        die(f"RUN_DIR not found: {RUN_DIR}")

    lock_path = RUN_DIR / ".ga_loop.lock"
    if lock_path.exists():
        die(f"Lock file exists: {lock_path}\nClose other loops or delete this lock if you're sure nothing is running.")

    lock_path.write_text(f"started {datetime.now()}\n", encoding="utf-8")

    try:
        params_path = RUN_DIR / PARAMS_NAME
        if not params_path.exists():
            die(f"params.txt not found in run dir: {params_path}")

        xbeach_exe = find_xbeach_exe(XBEACH_BASE)
        print(f"Using XBeach exe: {xbeach_exe}")
        print(f"Run dir: {RUN_DIR}")

        params = read_params(params_path)
        nx = int(params.get("nx"))
        ny = int(params.get("ny"))
        dx = float(params.get("dx"))
        dy = float(params.get("dy"))
        posdwn_param = float(params.get("posdwn", 1.0))

        base_bed_path = None
        for nm in BASE_BED_CANDIDATES:
            p = RUN_DIR / nm
            if p.exists():
                base_bed_path = p
                break
        if base_bed_path is None:
            die(f"Could not find a NO-WALL base bed in {RUN_DIR}. Expected one of: {BASE_BED_CANDIDATES}")

        # Ensure model uses bed_wall.dep
        ensure_depfile(params_path, BED_WALL_NAME)

        trials_path = RUN_DIR / TRIALS_TXT_NAME
        csv_path = RUN_DIR / CSV_NAME
        foundation_csv = RUN_DIR / FOUNDATION_CSV_NAME
        bed_wall_path = RUN_DIR / BED_WALL_NAME

        if not csv_path.exists():
            die(f"Missing {CSV_NAME} in run folder. Create it first (10 points).")

        x_pts_current, z_pts_current = read_control_points_csv(csv_path)

        # base bed mean sand profile (pos-up using params posdwn for wall generation)
        base_bed_raw = read_bed_nodes(base_bed_path, ny, nx)
        base_bed_up = to_pos_up(base_bed_raw, posdwn_param)
        sand_mean_x = np.nanmean(base_bed_up, axis=0)  # (nx+1,)
        x_nodes = np.arange(nx + 1) * dx

        offshore_side = infer_offshore_side_from_bed(base_bed_up)

        x_wall_face = snap_to_grid(float(x_pts_current[0]), dx)

        ix_face = int(round(x_wall_face / dx))
        ix_face = max(0, min(nx, ix_face))
        sand_at_face = float(sand_mean_x[ix_face])
        toe_z = sand_at_face + TOE_ABOVE_SAND_M

        x_pts_fixed = make_x_pts_from_face(x_wall_face, offshore_side)

        print("\nFixed wall placement:")
        print(f"  offshore_side = {offshore_side}")
        print(f"  wall face x   = {x_wall_face:.3f} m from LEFT")
        print(f"  x_pts_fixed   = {x_pts_fixed}")

        print("\nHard constraints:")
        print(f"  toe_z = sand@face({sand_at_face:.3f}) + 0.5 = {toe_z:.3f} m (pos-up)")
        print(f"  foundation base = sand(x) - 3.0 m (saved to {FOUNDATION_CSV_NAME})")
        print(f"  crest clipped into [{CREST_MIN},{CREST_MAX}] m")

        print("\nGA objective mode:", GA_OBJECTIVE_MODE)
        if GA_OBJECTIVE_MODE.lower() == "absnet":
            print("  minimize 0.5*|NetΔV_toe| + 0.5*|NetΔV_outer| + soft penalties")
        else:
            print("  minimize 0.5*Erosion_toe + 0.5*Erosion_outer + soft penalties")

        successful = 0
        attempt_total = 0

        while successful < TARGET_SUCCESSFUL_TRIALS:
            attempt_total += 1
            print("\n" + "=" * 110)
            print(f"TARGET trial {successful+1}/{TARGET_SUCCESSFUL_TRIALS}  (overall attempts so far: {attempt_total})")

            trials = load_trials(trials_path)

            crash_retries = 0
            while crash_retries <= MAX_CRASH_RETRIES_PER_TRIAL:
                z_new, mode, pen_est = propose_next_wall_z(trials, z_pts_current, toe_z, x_pts_fixed)
                z_new = enforce_hard_constraints(z_new, toe_z)

                sand_at_xpts = np.interp(x_pts_fixed, x_nodes, sand_mean_x)
                z_base = sand_at_xpts - FOUNDATION_DEPTH_BELOW_SAND_M

                print(f"\nGA proposal: mode={mode}  soft_penalty_est≈{pen_est:.1f}")
                print(f"  crest={z_new[-1]:.3f} m, toe={z_new[0]:.3f} m")

                # Backups (optional)
                if csv_path.exists():
                    shutil.copy2(csv_path, RUN_DIR / "wall_control_points_prev.csv")
                if bed_wall_path.exists():
                    shutil.copy2(bed_wall_path, RUN_DIR / "bed_wall_prev.dep")

                # Write updated control points + foundation points
                write_control_points_csv(csv_path, x_pts_fixed, z_new)
                write_foundation_points_csv(foundation_csv, x_pts_fixed, z_base)

                # Rebuild bed_wall.dep from NO-WALL bed + new wall surface (uses params posdwn)
                build_bed_wall_from_base(base_bed_path, bed_wall_path, ny, nx, dx, posdwn_param, x_pts_fixed, z_new)

                # Run XBeach
                print("\n--- Running XBeach ---")
                t0 = datetime.now().timestamp()
                ok, log_path = run_xbeach(xbeach_exe, RUN_DIR)
                print(f"XBeach finished. success={ok}. log={log_path}")

                if not ok:
                    crash_retries += 1
                    print(f"\nRUN CRASHED -> does NOT count. Retrying ({crash_retries}/{MAX_CRASH_RETRIES_PER_TRIAL})")
                    if crash_retries > MAX_CRASH_RETRIES_PER_TRIAL:
                        die("Too many crashes in a row. Fix run stability (CFL/dtset/bathy) then rerun.")
                    continue

                zb_path = newest_file_matching(RUN_DIR, "zb*.dat", min_mtime=t0) or newest_file_matching(RUN_DIR, "zb*.dat", None)
                if zb_path is None:
                    die("XBeach succeeded but no zb*.dat found. Ensure you output zb and ran in the right folder.")

                print(f"\nUsing zb output: {zb_path}")

                # Compute ΔV and append Trial (uses sign auto-fix if erosion comes out 0)
                print("\n--- Computing ΔV (MAIN + TOE + OUTER) and appending Trial ---")
                trial_n = compute_and_log_deltaV(
                    RUN_DIR, nx, ny, dx, dy, posdwn_param,
                    x_pts_fixed, z_new,
                    zb_path, trials_path
                )

                z_pts_current = z_new.copy()
                successful += 1
                print(f"\n✅ Completed successful Trial {trial_n}. Progress: {successful}/{TARGET_SUCCESSFUL_TRIALS}")
                break

        print("\nALL DONE.")
        print(f"Completed {successful} successful trials.")
        print(f"Trials log: {trials_path}")

    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
