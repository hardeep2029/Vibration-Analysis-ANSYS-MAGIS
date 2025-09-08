from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tf_tools import (
    read_ansys_tabular,                  # for velocity (amplitude-only)
    read_ansys_tabular_complex,          # for Z-displacements (amp + phase)
    normalize_to_per_accel,
    angle_tf_from_pair_complex
)

# -------- USER CONFIG --------
DATA_DIR = Path("../data")
OUT_DIR  = Path("../tfs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Baselines (meters) — measure in Mechanical (centre ↔ probe)
BASELINE_Y_M = 0.0200   # <-- set to your true |ΔY(CENTRE_PT, YMIN_PT)|
BASELINE_X_M = 0.0200   # <-- set to your true |ΔX(CENTRE_PT, XPLUS_PT)|

# Your ANSYS run used base ACCELERATION = 1 mm/s^2
EXCITATION_MODE = "ACCEL"
BASE_VALUE      = 1e-3  # m/s^2

# Model units: if the Mechanical model uses mm, convert to meters here
LENGTH_TO_M = 1e-3      # 1.0 if model is already in meters

# ANSYS tabular exports (as .txt or .csv)
DEF_CENTRE_Z = DATA_DIR / "Def_CENTRE_Z.txt"
DEF_YMIN_Z   = DATA_DIR / "Def_YMIN_Z.txt"
DEF_XPLUS_Z  = DATA_DIR / "Def_XPLUS_Z.txt"

VEL_CENTRE_X = DATA_DIR / "Vel_CENTRE_X.txt"
VEL_CENTRE_Y = DATA_DIR / "Vel_CENTRE_Y.txt"
VEL_CENTRE_Z = DATA_DIR / "Vel_CENTRE_Z.txt"

# -------- helpers --------
def write_two_col(path: Path, f: np.ndarray, y: np.ndarray):
    pd.DataFrame({"f": f.astype(float),
                  "resp_per_accel": np.asarray(y).astype(float)}
                 ).to_csv(path, index=False, header=False)

def check_freq_alignment(tag_a: str, f_a: np.ndarray,
                         tag_b: str, f_b: np.ndarray, tol=1e-9):
    import numpy as np
    if len(f_a) != len(f_b) or np.max(np.abs(f_a - f_b)) > tol:
        print(f"[WARN] frequency grids differ: {tag_a} vs {tag_b} "
              f"(len {len(f_a)} vs {len(f_b)}, max |Δf|={np.max(np.abs(f_a - f_b)):.3g})")
    else:
        print(f"[OK] frequency grids match: {tag_a} & {tag_b} (N={len(f_a)})")

# -------- load & normalize (per m/s^2); apply unit scaling --------
def load_norm_complex(path: Path):
    f, y = read_ansys_tabular_complex(str(path))
    f, ypa = normalize_to_per_accel(f, y, excitation_mode=EXCITATION_MODE, base_value=BASE_VALUE)
    return f, ypa

def load_norm_real(path: Path):
    f, y = read_ansys_tabular(str(path))
    f, ypa = normalize_to_per_accel(f, y, excitation_mode=EXCITATION_MODE, base_value=BASE_VALUE)
    return f, ypa

# Z-displacement per accel at three points (complex), then mm→m
f_c, wz_c = load_norm_complex(DEF_CENTRE_Z);  wz_c *= LENGTH_TO_M
f_y, wz_y = load_norm_complex(DEF_YMIN_Z);    wz_y *= LENGTH_TO_M
f_x, wz_x = load_norm_complex(DEF_XPLUS_Z);   wz_x *= LENGTH_TO_M

# Frequency alignment sanity check
check_freq_alignment("CENTRE_Z", f_c, "YMIN_Z", f_y)
check_freq_alignment("CENTRE_Z", f_c, "XPLUS_Z", f_x)

# Velocity per accel at centre (amplitude-only), then mm→m
f_vx, vx_per = load_norm_real(VEL_CENTRE_X);  vx_per *= LENGTH_TO_M
f_vy, vy_per = load_norm_real(VEL_CENTRE_Y);  vy_per *= LENGTH_TO_M
f_vz, vz_per = load_norm_real(VEL_CENTRE_Z);  vz_per *= LENGTH_TO_M

# -------- angle TFs from COMPLEX subtraction --------
# θx ≈ (wZ(Centre) - wZ(Y-)) / L_Y
f_thx, thx_c = angle_tf_from_pair_complex(f_c, wz_c, f_y, wz_y, baseline_m=BASELINE_Y_M)
# θy ≈ (wZ(Centre) - wZ(X+)) / L_X
f_thy, thy_c = angle_tf_from_pair_complex(f_c, wz_c, f_x, wz_x, baseline_m=BASELINE_X_M)

# Save MAGNITUDES for downstream usage (plots multiply by |TF|)
OUT = OUT_DIR
write_two_col(OUT / "theta_x_tf.csv", f_thx, np.abs(thx_c))  # |rad/(m/s^2)|
write_two_col(OUT / "theta_y_tf.csv", f_thy, np.abs(thy_c))
write_two_col(OUT / "vel_x_tf.csv",   f_vx,  vx_per)         # s
write_two_col(OUT / "vel_y_tf.csv",   f_vy,  vy_per)
write_two_col(OUT / "vel_z_tf.csv",   f_vz,  vz_per)

def _summ(name, f, y, unit):
    print(f"[TF] {name}: f=[{f.min():.3g}, {f.max():.3g}] Hz, "
          f"|y| min/max/median = {np.nanmin(np.abs(y)):.3e} / "
          f"{np.nanmax(np.abs(y)):.3e} / {np.nanmedian(np.abs(y)):.3e} [{unit}]")

_summ("theta_x", f_thx, thx_c, "rad/(m/s^2)")
_summ("theta_y", f_thy, thy_c, "rad/(m/s^2)")
_summ("vel_x",   f_vx,  vx_per, "s")
_summ("vel_y",   f_vy,  vy_per, "s")
_summ("vel_z",   f_vz,  vz_per, "s")

print("Wrote TFs to:", OUT.resolve())
