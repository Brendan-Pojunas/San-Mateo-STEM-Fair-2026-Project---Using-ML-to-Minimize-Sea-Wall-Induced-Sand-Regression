import os
import re
import numpy as np

# =========================
# SETTINGS (edit if needed)
# =========================
RUN_DIR = r"C:\Users\brend\Desktop\sf_1yr_case"

# landward face of wall (meters from left edge)
WALL_LANDWARD_X_M = 465.0

# qx output filename (Fortran/raw stream)
QX_FILE = "qx.dat"

# choose bed baseline file if present
BED_PREFER = ["bed_fliplr.dep", "bed.dep"]


# =========================
# HELPERS
# =========================
def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def parse_params(params_path):
    txt = read_text(params_path)
    d = {}
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("#"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.split("%", 1)[0].split("#", 1)[0].strip()
            d[k] = v

    def to_int(x): return int(float(x))
    def to_float(x): return float(x)

    nx = to_int(d.get("nx", "0"))
    ny = to_int(d.get("ny", "0"))
    dx = to_float(d.get("dx", "0"))
    dy = to_float(d.get("dy", "0"))
    tstart = to_float(d.get("tstart", "0"))
    tstop  = to_float(d.get("tstop", "0"))

    return nx, ny, dx, dy, tstart, tstop

def load_bed_nodes(run_dir, ny1, nx1):
    for fn in BED_PREFER:
        p = os.path.join(run_dir, fn)
        if os.path.isfile(p):
            bed = np.loadtxt(p)
            if bed.ndim == 1:
                bed = bed.reshape((ny1, nx1))
            if bed.shape != (ny1, nx1):
                raise ValueError(f"{fn} has shape {bed.shape}, expected {(ny1, nx1)}")
            return bed, p
    raise FileNotFoundError(f"Could not find any of {BED_PREFER} in {run_dir}")

def onshore_sign_from_bed(bed_nodes):
    # onshore side tends to have higher elevation
    left_mean = float(np.mean(bed_nodes[:, 0]))
    right_mean = float(np.mean(bed_nodes[:, -1]))
    # if right is higher => onshore direction is +x
    return +1 if right_mean > left_mean else -1

def read_qx_raw_timeseries(path, ny1, nx1):
    arr = np.fromfile(path, dtype=np.float64)
    n_per = ny1 * nx1
    if arr.size % n_per != 0:
        raise ValueError(
            f"qx.dat size not divisible by (ny+1)*(nx+1). "
            f"Values={arr.size}, expected multiple of {n_per}."
        )
    nt = arr.size // n_per
    return arr.reshape((nt, ny1, nx1))

def estimate_dt(tstart, tstop, nt):
    if nt <= 1:
        return None
    return (tstop - tstart) / (nt - 1)

def integrate_overtopping(qx_ts, dy, x_index, onshore_sign, dt):
    """
    qx: m^2/s. Integrate positive onshore flux across y and time.
    Returns total volume [m^3].
    """
    if dt is None:
        return 0.0, None

    # make onshore positive
    q = qx_ts[:, :, x_index] * onshore_sign  # (nt, ny+1)

    # nodal->cell alongshore (trapezoid)
    q_cell = 0.5 * (q[:, :-1] + q[:, 1:])    # (nt, ny)

    # only flow going landward (positive)
    q_pos = np.maximum(q_cell, 0.0)

    # integrate over y -> m^3/s
    Q_total = np.sum(q_pos, axis=1) * dy     # (nt,)

    # integrate over time -> m^3
    V = float(np.sum(Q_total * dt))

    peak_Q = float(np.max(Q_total)) if Q_total.size else 0.0
    return V, peak_Q


# =========================
# MAIN
# =========================
def main():
    params_path = os.path.join(RUN_DIR, "params.txt")
    if not os.path.isfile(params_path):
        raise FileNotFoundError(f"params.txt not found: {params_path}")

    nx, ny, dx, dy, tstart, tstop = parse_params(params_path)
    nx1, ny1 = nx + 1, ny + 1

    # load bed to determine onshore direction
    bed, bed_path = load_bed_nodes(RUN_DIR, ny1, nx1)
    onshore_sign = onshore_sign_from_bed(bed)

    # choose transect x-index
    i_land = int(round(WALL_LANDWARD_X_M / dx))
    i_land = max(0, min(nx1 - 1, i_land))
    x_used = i_land * dx

    qx_path = os.path.join(RUN_DIR, QX_FILE)
    if not os.path.isfile(qx_path):
        raise FileNotFoundError(
            f"{QX_FILE} not found in run folder: {qx_path}\n"
            f"Make sure nglobalvar included qx and XBeach produced outputs."
        )

    qx_ts = read_qx_raw_timeseries(qx_path, ny1, nx1)
    nt = qx_ts.shape[0]
    dt = estimate_dt(tstart, tstop, nt)

    V_over, peak_Q = integrate_overtopping(qx_ts, dy, i_land, onshore_sign, dt)

    print("==============================================")
    print(f"Run dir: {RUN_DIR}")
    print(f"Grid: nx={nx}, ny={ny}, dx={dx}, dy={dy}")
    print(f"Bed used: {bed_path}")
    print(f"Onshore direction sign: {onshore_sign}  (+1 means onshore to the RIGHT)")
    print(f"Transect at x={x_used:.3f} m (index {i_land})")
    print(f"qx timesteps (nt): {nt}")
    if dt is None:
        print("WARNING: nt<=1 so dt can't be estimated; overtopping integral will be 0.")
    else:
        print(f"Estimated dt between qx snapshots: {dt:.3f} s")
    print("----------------------------------------------")
    print(f"OVERTOPPING VOLUME onto land: {V_over:.6e} m^3")
    print(f"Peak discharge across transect: {peak_Q:.6e} m^3/s")
    print("==============================================")

if __name__ == "__main__":
    main()
