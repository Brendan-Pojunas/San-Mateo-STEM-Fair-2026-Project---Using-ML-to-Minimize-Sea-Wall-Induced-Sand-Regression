import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# SETTINGS
# ===============================
run_dir  = r"C:\Users\brend\Desktop\sf_1yr_case"
bed_path = os.path.join(run_dir, "bed.dep")      # BEFORE
zb_path  = os.path.join(run_dir, "zb.dat")       # AFTER
wall_csv = os.path.join(run_dir, "vertical_wall_points.csv")  # wall x-range (min/max x)

nx, ny = 120, 40
dx, dy = 5.0, 5.0

# Use the posdwn that was used in THIS run
posdwn = 1  # you said vertical-wall trial used posdwn=1

# ΔV window (meters). Set both None for full domain.
xwin_min = None
xwin_max = None

mid_y = ny // 2
out_png = os.path.join(run_dir, "FixedSign_VerticalWall_BeforeAfter_BedOnly.png")

# ===============================
# HELPERS
# ===============================
def read_wall_x_range(csv_path: str):
    xs = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["x_from_left_m"]))
    if not xs:
        raise ValueError("Wall CSV had no x points.")
    return float(min(xs)), float(max(xs))

def bridge_over_interval(x, z, x0, x1):
    """Replace z inside [x0,x1] with a straight line between points just outside."""
    z = z.astype(float).copy()
    inside = (x >= x0) & (x <= x1)
    if not np.any(inside):
        return z
    Lc = np.where(x < x0)[0]
    Rc = np.where(x > x1)[0]
    if len(Lc) == 0 or len(Rc) == 0:
        return z
    L = Lc[-1]
    R = Rc[0]
    z[inside] = np.interp(x[inside], [x[L], x[R]], [z[L], z[R]])
    return z

def node_to_cell_avg(z_nodes):
    return 0.25 * (z_nodes[:-1, :-1] + z_nodes[1:, :-1] + z_nodes[:-1, 1:] + z_nodes[1:, 1:])

def read_zb_raw_last_field(zb_path, nx, ny):
    raw = np.fromfile(zb_path, dtype=np.float64)
    nodes = (ny + 1) * (nx + 1)
    if raw.size < nodes:
        raise RuntimeError(f"zb.dat too small: {raw.size} float64 values; need {nodes}")
    nfields = raw.size // nodes
    last = raw[(nfields - 1) * nodes : nfields * nodes]
    return last.reshape((ny + 1, nx + 1)), nfields

def to_positive_up(z_nodes, posdwn):
    """Convert a file's z to positive-up convention."""
    return -z_nodes if posdwn == 1 else z_nodes

def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    den = (np.nanstd(a) * np.nanstd(b))
    return float(np.nanmean(a * b) / den) if den > 0 else 0.0

# ===============================
# LOAD DATA
# ===============================
bed0 = np.loadtxt(bed_path)
expected = (ny + 1, nx + 1)
if bed0.shape != expected:
    raise ValueError(f"bed.dep shape {bed0.shape} != expected {expected}")

zb_final, nfields = read_zb_raw_last_field(zb_path, nx, ny)
print(f"zb.dat float64 fields detected: {nfields} (using last field)")

# Convert BOTH to positive-up (first pass, using posdwn)
bed_up = to_positive_up(bed0, posdwn)
zb_up  = to_positive_up(zb_final, posdwn)

# ===============================
# AUTO-FIX THE PROBLEM YOU SEE:
# if after is basically the negative of before, corr will be ~ -1
# ===============================
x_nodes = np.arange(nx + 1, dtype=float) * dx
t_before = bed_up[mid_y, :]
t_after  = zb_up[mid_y, :]

c0 = corr(t_before, t_after)
print(f"Initial transect correlation (before vs after) = {c0:+.3f}")

if c0 < -0.2:
    # Fix: flip AFTER sign (most common for this exact plot)
    zb_up = -zb_up
    c1 = corr(bed_up[mid_y, :], zb_up[mid_y, :])
    print(f"Detected sign mismatch. Flipped AFTER (zb). New corr = {c1:+.3f}")

# ===============================
# ΔV USING CORRECTED ARRAYS (positive-up)
# dz_up = zb_up - bed_up   (because both already pos-up)
# ===============================
dz_up_nodes = (zb_up - bed_up)
dz_up_cell  = node_to_cell_avg(dz_up_nodes)  # (ny, nx)
cell_area = dx * dy

x_cell = (np.arange(nx) + 0.5) * dx
if xwin_min is not None and xwin_max is not None:
    mask = (x_cell >= xwin_min) & (x_cell <= xwin_max)
    dz_use = dz_up_cell[:, mask]
else:
    dz_use = dz_up_cell

dv = dz_use * cell_area
net = float(np.sum(dv))
dep = float(np.sum(dv[dv > 0]))
ero = float(-np.sum(dv[dv < 0]))

print("\nΔV results (m³), positive-up convention (after auto-fix):")
print(f"  Net ΔV          : {net:.6e}")
print(f"  Deposition (Σ+) : {dep:.6e}")
print(f"  Erosion    (Σ-) : {ero:.6e}")

# ===============================
# PLOT (wall removed from BOTH)
# ===============================
wall_x0, wall_x1 = read_wall_x_range(wall_csv)
print(f"\nWall x-range from CSV: [{wall_x0:.3f}, {wall_x1:.3f}] m")

z_before = bed_up[mid_y, :].copy()
z_after  = zb_up[mid_y, :].copy()

z_before_nw = bridge_over_interval(x_nodes, z_before, wall_x0, wall_x1)
z_after_nw  = bridge_over_interval(x_nodes, z_after,  wall_x0, wall_x1)

plt.figure(figsize=(11, 6))
plt.plot(x_nodes, z_before_nw, label="Before storm (bed only)", linewidth=2)
plt.plot(x_nodes, z_after_nw,  label="After storm (bed only)", linewidth=2)

plt.axhline(0.0, linestyle="--", linewidth=1)
plt.axvspan(wall_x0, wall_x1, alpha=0.15, label="Wall region (removed from lines)")

plt.xlabel("Cross-shore distance x (m from LEFT edge)")
plt.ylabel("Elevation z (m, positive up)")
plt.title("Vertical Wall: Cross-section Before vs After (Wall Removed) — FIXED")
plt.grid(True)
plt.legend()

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()
print(f"\nSaved plot: {out_png}")
