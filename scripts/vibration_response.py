from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tf_tools import (
    read_single_column_csv, read_two_column_csv,
    splice_spectra, asd_to_rms
)

# -------------------- User options --------------------
DATA_DIR = Path("../data")
TFS_DIR  = Path("../tfs")

DO_X = False
DO_Y = True
DO_Z = False

F_CUT = 100.0

OUT_DIR = Path("../plots")
SAVE_PLOTS = True
SHOW_PLOTS = False
FIG_FMT = "png"
DPI = 300

MAKE_SANITY_PATCH_PLOT = True

# -------------------- Plot helpers --------------------
def style_loglog(ax, title, ylab):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Frequency (Hz)", fontsize=13)
    ax.set_ylabel(ylab, fontsize=13)
    ax.grid(True, which="both", ls="--", alpha=0.2)

def plot_series(f, curves, title, ylab, loc="upper left", savepath: Path | None = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, y in curves.items():
        ax.plot(f, y, label=label, alpha=0.9)
    style_loglog(ax, title, ylab)
    ax.legend(loc=loc, fontsize=12)
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
        print(f"Saved plot to {savepath}")
    if SHOW_PLOTS: plt.show()
    else: plt.close(fig)

def plot_patched_seismic_drive(f_up, a_up, f_side, a_side, savepath=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f_side, a_side, label="Side-to-side", alpha=0.9)
    ax.plot(f_up,   a_up,   label="Up-down",    alpha=0.9)
    style_loglog(ax, "Patched Seismic Drive Data", r"$(m/s^2)/\sqrt{Hz}$")
    ax.set_xlabel("Hz"); ax.legend(loc="best", fontsize=12)
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=DPI, bbox_inches="tight")
        print(f"Saved plot to {savepath}")
    if SHOW_PLOTS: plt.show()
    else: plt.close(fig)

def plot_tf_quicklook():
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(f_thx, np.abs(thx), label=r"$\Theta_x$ TF", alpha=0.9)
    ax.plot(f_thy, np.abs(thy), label=r"$\Theta_y$ TF", alpha=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Angle TF quick-look (|rad/(m/s²)|)")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
    ax.grid(True, which="both", ls="--", alpha=0.2); ax.legend()
    if SAVE_PLOTS:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / f"angle_tf_quicklook.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS: plt.show()
    else: plt.close(fig)

# -------------------- Load & splice detector ASDs --------------------
f_ref = read_single_column_csv(DATA_DIR / "f_151.csv")
asd_ud = read_single_column_csv(DATA_DIR / "s151_accel_ch1.csv")
asd_ss = read_single_column_csv(DATA_DIR / "s151_accel_ch2.csv")

f_pcb = read_single_column_csv(DATA_DIR / "f_pcb.csv")
asd_pcb = read_single_column_csv(DATA_DIR / "PCB_accel_ch1.csv")

f_updown,   a_updown  = splice_spectra(f_ref, asd_ud, f_pcb, asd_pcb, F_CUT)
f_side2side, a_side2  = splice_spectra(f_ref, asd_ss, f_pcb, asd_pcb, F_CUT)

# -------------------- Load transfer functions --------------------
f_thx, thx = read_two_column_csv(TFS_DIR / "theta_x_tf.csv")  # |rad/(m/s^2)|
f_thy, thy = read_two_column_csv(TFS_DIR / "theta_y_tf.csv")
f_vx,  tvx = read_two_column_csv(TFS_DIR / "vel_x_tf.csv")    # s
f_vy,  tvy = read_two_column_csv(TFS_DIR / "vel_y_tf.csv")
f_vz,  tvz = read_two_column_csv(TFS_DIR / "vel_z_tf.csv")

I_thx = interp1d(f_thx, thx, bounds_error=False, fill_value="extrapolate")
I_thy = interp1d(f_thy, thy, bounds_error=False, fill_value="extrapolate")
I_vx  = interp1d(f_vx,  tvx, bounds_error=False, fill_value="extrapolate")
I_vy  = interp1d(f_vy,  tvy, bounds_error=False, fill_value="extrapolate")
I_vz  = interp1d(f_vz,  tvz, bounds_error=False, fill_value="extrapolate")

# -------------------- Per-axis compute --------------------
def compute_for_axis(axis: str, f_base: np.ndarray, a_base: np.ndarray):
    # ANGLES: overlap of theta_x / theta_y only
    fmin_ang = max(np.min(f_thx), np.min(f_thy))
    fmax_ang = min(np.max(f_thx), np.max(f_thy))
    m_ang = (f_base >= fmin_ang) & (f_base <= fmax_ang)
    f_ang, a_ang = f_base[m_ang], a_base[m_ang]
    ang_x = a_ang * I_thx(f_ang)
    ang_y = a_ang * I_thy(f_ang)
    plot_series(f_ang, {r"$\theta_x$": ang_x, r"$\theta_y$": ang_y},
                f"Angular Response ASD (Drive {axis})",
                r"Angle ASD (rad / $\sqrt{\mathrm{Hz}}$)",
                savepath=(OUT_DIR / f"{axis}_angular_response.{FIG_FMT}") if SAVE_PLOTS else None)

    # VELOCITIES: overlap of vx / vy / vz
    fmin_vel = max(np.min(f_vx), np.min(f_vy), np.min(f_vz))
    fmax_vel = min(np.max(f_vx), np.max(f_vy), np.max(f_vz))
    m_vel = (f_base >= fmin_vel) & (f_base <= fmax_vel)
    f_vel, a_vel = f_base[m_vel], a_base[m_vel]
    vx_asd = a_vel * I_vx(f_vel)
    vy_asd = a_vel * I_vy(f_vel)
    vz_asd = a_vel * I_vz(f_vel)
    plot_series(f_vel, {r"$v_x$": vx_asd, r"$v_y$": vy_asd, r"$v_z$": vz_asd},
                f"Velocity Response ASD (Drive {axis})",
                r"Velocity ASD (m/s / $\sqrt{\mathrm{Hz}}$)",
                savepath=(OUT_DIR / f"{axis}_velocity_response.{FIG_FMT}") if SAVE_PLOTS else None)

    # RMS over velocity band
    print(f"{axis}-axis drive RMS (full band):")
    print(f"  v_x_rms = {asd_to_rms(vx_asd, f_vel):.6g} m/s")
    print(f"  v_y_rms = {asd_to_rms(vy_asd, f_vel):.6g} m/s")
    print(f"  v_z_rms = {asd_to_rms(vz_asd, f_vel):.6g} m/s")

# -------------------- Run --------------------
if __name__ == "__main__":
    if MAKE_SANITY_PATCH_PLOT:
        plot_patched_seismic_drive(
            f_updown, a_updown, f_side2side, a_side2,
            savepath=(OUT_DIR / f"patched_seismic_drive.{FIG_FMT}") if SAVE_PLOTS else None
        )
    plot_tf_quicklook()
    if DO_X:
        compute_for_axis("X", f_side2side, a_side2)
    if DO_Y:
        compute_for_axis("Z", f_updown, a_updown)   # vertical drive
    if DO_Z:
        compute_for_axis("Y", f_side2side, a_side2)
