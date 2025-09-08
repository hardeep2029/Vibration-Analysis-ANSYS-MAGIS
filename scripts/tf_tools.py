from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.interpolate import interp1d
from scipy.integrate import simpson

# ---------- robust readers ----------

def read_single_column_csv(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f.read().splitlines():
            s = raw.replace("\ufeff","").strip()
            if not s:
                continue
            # accept comma or tab
            if "," in s and "\t" not in s:
                v = s.split(",")[0]
            elif "\t" in s:
                v = s.split("\t")[0]
            else:
                v = s
            vals.append(float(v))
    return np.asarray(vals, float)

def read_two_column_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f.read().splitlines():
            s = raw.replace("\ufeff","").strip()
            if not s:
                continue
            if "," in s and "\t" not in s:
                parts = s.split(",")
            elif "\t" in s and "," not in s:
                parts = s.split("\t")
            else:
                parts = s.replace("\t", ",").split(",")
            if len(parts) >= 2:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
    return np.asarray(xs, float), np.asarray(ys, float)

def read_ansys_tabular(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ANSYS 'Tabular Data' export (.txt or .csv), comma or tab delimited.
    Returns (f_Hz, amplitude only).
    """
    df = pd.read_csv(path, sep=None, engine="python",
                     encoding="utf-8-sig", comment="#")
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    fcol = next((c for c in cols if "freq" in c), None)
    cand = ("amplitude", "magnitude", "amp", "abs", "value")
    acol = next((c for c in cols if any(k in c for k in cand)), None)
    if fcol is None or acol is None:
        num = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
        if len(num) < 2:
            raise ValueError(f"Could not find frequency/amplitude in {path}")
        fcol, acol = num[0], num[1]
    out = df[[fcol, acol]].dropna().sort_values(fcol).to_numpy(float)
    f, a = out[:, 0], out[:, 1]
    return f, a

def read_ansys_tabular_complex(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ANSYS tabular data with amplitude and (optional) phase columns.
    Returns (f_Hz, complex_response).
    Phase column is assumed in degrees if present.
    """
    df = pd.read_csv(path, sep=None, engine="python",
                     encoding="utf-8-sig", comment="#")
    cols = {c.strip().lower(): c for c in df.columns}
    # frequency
    fcol = next((cols[c] for c in cols if "freq" in c), None)
    if fcol is None:
        num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        if not num:
            raise ValueError(f"No numeric frequency in {path}")
        fcol = num[0]
    # amplitude
    acol = next((cols[c] for c in cols
                 if any(k in c for k in ("amplitude", "magnitude", "amp", "abs", "value"))), None)
    if acol is None:
        raise ValueError(f"No amplitude column in {path}")
    # optional phase (deg)
    pcol = next((cols[c] for c in cols if "phase" in c), None)

    f = df[fcol].to_numpy(float)
    A = df[acol].to_numpy(float)

    if pcol is not None and np.isfinite(df[pcol]).any():
        phi_deg = df[pcol].to_numpy(float)
        y = A * np.exp(1j * np.deg2rad(phi_deg))
    else:
        y = A.astype(float)

    # clean/sort
    m = np.isfinite(f) & np.isfinite(A) & (f > 0)
    f, y = f[m], y[m]
    order = np.argsort(f)
    return f[order], y[order]

# ---------- normalization & conversions ----------

def normalize_to_per_accel(
    f: np.ndarray,
    y: np.ndarray,
    excitation_mode: str = "ACCEL",  # "ACCEL" or "DISP"
    base_value: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize response to per-(m/s^2).
    If excitation was base displacement, also divide by (2πf)^2.
    """
    f = np.asarray(f, float)
    y = np.asarray(y)
    mask = f > 0.0
    f, y = f[mask], y[mask]
    if base_value == 0:
        raise ValueError("base_value must be nonzero")
    if excitation_mode.upper() == "ACCEL":
        return f, y / float(base_value)
    elif excitation_mode.upper() == "DISP":
        return f, (y / float(base_value)) / ((2 * np.pi * f) ** 2)
    else:
        raise ValueError("excitation_mode must be 'ACCEL' or 'DISP'")

def resample_to(f_target: np.ndarray, f_src: np.ndarray, y_src: np.ndarray):
    itp = interp1d(f_src, y_src, bounds_error=False,
                   fill_value="extrapolate", assume_sorted=False)
    return itp(f_target)

# ---------- angle TFs ----------

def angle_tf_from_pair(
    f_top: np.ndarray,
    w_top: np.ndarray,
    f_bot: np.ndarray,
    w_bot: np.ndarray,
    baseline_m: float,
    signed: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy real-only version: θ ≈ (w_top - w_bot)/L on the top grid."""
    if baseline_m <= 0:
        raise ValueError("baseline_m must be > 0")
    w_bot_on_top = resample_to(f_top, f_bot, w_bot)
    theta = (w_top - w_bot_on_top) / float(baseline_m)
    return (f_top, theta if signed else np.abs(theta))

def angle_tf_from_pair_complex(
    f_top: np.ndarray,
    w_top_c: np.ndarray,
    f_bot: np.ndarray,
    w_bot_c: np.ndarray,
    baseline_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complex version: use amplitude + phase from ANSYS.
    Returns complex θ TF on the top grid.
    """
    if baseline_m <= 0:
        raise ValueError("baseline_m must be > 0")
    w_bot_on_top = resample_to(f_top, f_bot, w_bot_c)
    theta = (w_top_c - w_bot_on_top) / float(baseline_m)
    return f_top, theta

# ---------- room ASD splice & RMS ----------

def splice_spectra(
    f_left: np.ndarray, asd_left: np.ndarray,
    f_right: np.ndarray, asd_right: np.ndarray,
    f_cut: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Take left < f_cut, right ≥ f_cut."""
    Ls = np.argsort(f_left)
    Rs = np.argsort(f_right)
    fL, aL = f_left[Ls], asd_left[Ls]
    fR, aR = f_right[Rs], asd_right[Rs]
    iL = np.searchsorted(fL, f_cut, side="right")
    iR = np.searchsorted(fR, f_cut, side="left")
    f = np.concatenate([fL[:iL], fR[iR:]])
    a = np.concatenate([aL[:iL], aR[iR:]])
    return f.astype(float), a.astype(float)

def asd_to_rms(asd: np.ndarray, f: np.ndarray,
               fmin: Optional[float] = None,
               fmax: Optional[float] = None) -> float:
    f = np.asarray(f, float); asd = np.asarray(asd, float)
    if fmin is not None or fmax is not None:
        m = np.ones_like(f, bool)
        if fmin is not None:
            m &= f >= fmin
        if fmax is not None:
            m &= f <= fmax
        f, asd = f[m], asd[m]
    var = simpson(asd * asd, f)
    return float(np.sqrt(max(var, 0.0)))
