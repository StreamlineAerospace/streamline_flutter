from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from pyNastran.f06.parse_flutter import make_flutter_response
from scipy.optimize import linear_sum_assignment
from flutter_nastran import stdatm1976

#set plotting style
plt.style.use("bmh")
try:
    plt.rcParams['font.family'] = 'Saira'
except: pass 

# Copyright (C) <2026>, Streamline Aerospace. All rights reserved.

def extract_vf_vg_data(f06_path: str, altitude_ft: float) -> dict:
    """Extracts raw flutter data from a NASTRAN SOL 145 .f06 file using pyNastran.
    Concatenates multi-subcase results to maintain a flat dictionary structure."""
    
    flutters = make_flutter_response(f06_path)

    # Handle standard parsing (single object) vs multi-subcase parsing (dict)
    if not isinstance(flutters, dict):
        flutters = {1: flutters}

    # true and equivalent airspeeds constants 
    rho_slug_ft3 = stdatm1976.air_density(h_ft=altitude_ft)
    rho0_slug_ft3 = stdatm1976.air_density(h_ft=0) 
    a_ft_s = stdatm1976.sonic_speed(h_ft=altitude_ft)
    a_kts = 0.592484 * a_ft_s

    res_dict = {}

    # Loop through all subcases (Mach numbers) to stitch arrays together
    for subcase_id, flutter in flutters.items():
        
        i_dmp = flutter.idamping  # Index 5
        i_frq = flutter.ifreq     # Index 6
        i_mach = flutter.imach

        results = flutter.results
        modes = flutter.modes

        for mode in modes:
            mode_key = f"Mode {mode}"
            
            # Initialize empty lists for the mode if it doesn't exist yet
            if mode_key not in res_dict:
                res_dict[mode_key] = {
                    "mach": [],
                    "damping": [],
                    "frequency": [],
                    "keas": [],
                    "ktas": []
                }
            
            # Extract data for this specific subcase/mode
            mode_data = results[mode - 1]
            mach_arr = mode_data[:, i_mach].copy()
            dmp_arr = mode_data[:, i_dmp].copy()
            frq_arr = mode_data[:, i_frq].copy()
            
            # Extend the global mode lists with this subcase's data
            res_dict[mode_key]["mach"].extend(mach_arr)
            res_dict[mode_key]["damping"].extend(dmp_arr)
            res_dict[mode_key]["frequency"].extend(frq_arr)
            res_dict[mode_key]["keas"].extend(mach_arr * a_kts * np.sqrt(rho_slug_ft3/rho0_slug_ft3))
            res_dict[mode_key]["ktas"].extend(mach_arr * a_kts)

    # Convert all lists back to numpy arrays to maintain original datatype formatting
    for mode_key in res_dict:
        for data_key in res_dict[mode_key]:
            res_dict[mode_key][data_key] = np.array(res_dict[mode_key][data_key])

    return res_dict


def repair_vf_vg_curves(vf_vg_data: dict) -> dict:
    """Repairs mode switching and discontinuity issues in V-f and V-g flutter curves.

    Tracks eigenvalues across velocity steps using forward state extrapolation
    and optimal assignment (Hungarian algorithm).
    """
    mode_keys = list(vf_vg_data.keys())
    num_modes = len(mode_keys)
    if num_modes <= 1:
        return vf_vg_data

    num_steps = len(vf_vg_data[mode_keys[0]]["mach"])

    freq_raw = np.array([vf_vg_data[k]["frequency"] for k in mode_keys])
    damp_raw = np.array([vf_vg_data[k]["damping"] for k in mode_keys])
    mach_raw = np.array([vf_vg_data[k]["mach"] for k in mode_keys])
    keas_raw = np.array([vf_vg_data[k]["keas"] for k in mode_keys])
    ktas_raw = np.array([vf_vg_data[k]["ktas"] for k in mode_keys])

    repaired_freq = np.zeros_like(freq_raw)
    repaired_damp = np.zeros_like(damp_raw)
    repaired_mach = np.zeros_like(mach_raw)
    repaired_keas = np.zeros_like(keas_raw)
    repaired_ktas = np.zeros_like(ktas_raw)
    
    repaired_freq[:, 0] = freq_raw[:, 0]
    repaired_damp[:, 0] = damp_raw[:, 0]
    repaired_mach[:, 0] = mach_raw[:, 0]
    repaired_keas[:, 0] = keas_raw[:, 0]
    repaired_ktas[:, 0] = ktas_raw[:, 0]

    f_scale = (
        np.std(freq_raw[freq_raw > 0])
        if np.any(freq_raw > 0)
        else np.mean(freq_raw) + 1.0
    )
    if f_scale == 0 or np.isnan(f_scale):
        f_scale = 1.0
    d_scale = np.std(damp_raw) if np.std(damp_raw) > 0 else 1.0

    for k in range(1, num_steps):
        cand_f = freq_raw[:, k]
        cand_d = damp_raw[:, k]
        cand_v = mach_raw[:, k]

        if k == 1:
            pred_f = repaired_freq[:, 0]
            pred_d = repaired_damp[:, 0]
        else:
            dv_prev = repaired_mach[:, k - 1] - repaired_mach[:, k - 2]
            dv_curr = cand_v - repaired_mach[:, k - 1]
            safe_dv = np.where(np.abs(dv_prev) > 1e-9, dv_prev, 1.0)

            df_dv = (repaired_freq[:, k - 1] - repaired_freq[:, k - 2]) / safe_dv
            dd_dv = (repaired_damp[:, k - 1] - repaired_damp[:, k - 2]) / safe_dv

            pred_f = repaired_freq[:, k - 1] + df_dv * dv_curr
            pred_d = repaired_damp[:, k - 1] + dd_dv * dv_curr

        cost_matrix = np.zeros((num_modes, num_modes))
        for i in range(num_modes):
            df_pred = (cand_f - pred_f[i]) / f_scale
            dd_pred = (cand_d - pred_d[i]) / d_scale
            df_last = (cand_f - repaired_freq[i, k - 1]) / f_scale
            dd_last = (cand_d - repaired_damp[i, k - 1]) / d_scale

            cost_matrix[i, :] = np.sqrt(
                1.0 * (0.6 * df_pred**2 + 0.4 * df_last**2)
                + 0.15 * (0.6 * dd_pred**2 + 0.4 * dd_last**2)
            )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for track_i, cand_j in zip(row_ind, col_ind):
            repaired_freq[track_i, k] = cand_f[cand_j]
            repaired_damp[track_i, k] = cand_d[cand_j]
            repaired_mach[track_i, k] = cand_v[cand_j]
            #repaired_keas[track_i, k] = cand_v[cand_j]
            #repaired_ktas[track_i, k] = cand_v[cand_j]

    repaired_dict = {}
    for i, mode_key in enumerate(mode_keys):
        repaired_dict[mode_key] = {
            "mach": repaired_mach[i],
            "damping": repaired_damp[i],
            "frequency": repaired_freq[i],
            "keas": keas_raw[i],
            "ktas": ktas_raw[i]
        }

    return repaired_dict


def load_flutter_csv(csv_path: str | Path) -> dict:
    """Reads a flutter CSV export and returns a nested dictionary of modes.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing separate 'Frequency (Hz)' and 'Damping'
        tables.

    Returns
    -------
    dict
        Dictionary structured as:
        {
            "Mode 1": {
                "velocity": [Mach Number, ...],
                "frequency": [freq_Hz, ...],
                "damping": [damping, ...]
            },
            ...
        }
    """
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # The CSV consists of a "Frequency (Hz)" section and a "Damping" section
    parts = content.split("Damping")
    freq_raw = (
        parts[0].replace("Frequency (Hz)", "").strip()
    )  # remove section header
    damp_raw = parts[1].strip()

    # Load both tables into pandas DataFrames
    df_freq = pd.read_csv(io.StringIO(freq_raw))
    df_damp = pd.read_csv(io.StringIO(damp_raw))

    # Strip whitespace from column headers and drop trailing unnamed columns
    df_freq.columns = [c.strip() for c in df_freq.columns]
    df_damp.columns = [c.strip() for c in df_damp.columns]
    df_freq = df_freq.loc[:, ~df_freq.columns.str.contains(r"^Unnamed")]
    df_damp = df_damp.loc[:, ~df_damp.columns.str.contains(r"^Unnamed")]

    # Extract all mode columns present in the file
    mode_cols = [c for c in df_freq.columns if c.startswith("Mode")]

    # Build the dictionary
    flutter_dict = {}
    mach_list = df_freq["Mach Number"].tolist()
    keas_list = df_freq["Equivalent Airspeed (knots)"].tolist()
    ktas_list = df_freq["True Airspeed (knots)"].tolist()

    for mode in mode_cols:
        flutter_dict[mode] = {
            "mach": mach_list,
            "keas": keas_list,
            "ktas": ktas_list,
            "frequency": df_freq[mode].tolist(),
            "damping": df_damp[mode].tolist(),
        }

    return flutter_dict


def get_flutter_onset(vf_vg_data: dict, damping_threshold: float):

    threshold_crossings = []

    for key, data in vf_vg_data.items():
        
        mach = data["mach"]
        keas = data["keas"]
        ktas = data["ktas"]
        damping = data["damping"]
        frequency = data["frequency"]

        # 1. Identify frequency at lowest velocity (v = 0)
        m0_idx = np.argmin(np.abs(mach))
        f_v0 = frequency[m0_idx]
        
        for i in range(len(mach) - 1):
            g1, g2 = damping[i], damping[i + 1]
            m1, m2 = mach[i], mach[i + 1]
            v1, v2 = ktas[i], ktas[i + 1]
            ve1, ve2 = keas[i], keas[i + 1]

            # Upward crossing into instability or crossing through threshold
            if (g1 - damping_threshold) * (g2 - damping_threshold) <= 0 and g1 != g2:
                m_cross = (m1 + (damping_threshold - g1) / (g2 - g1) * (m2 - m1))
                v_cross = (v1 + (damping_threshold - g1) / (g2 - g1) * (v2 - v1))
                ve_cross = (ve1 + (damping_threshold - g1) / (g2 - g1) * (ve2 - ve1))
                threshold_crossings.append(
                    {
                        "mode": key,
                        "mach_cross": m_cross,
                        "ktas_cross": v_cross,
                        "keas_cross": ve_cross,
                        "f_v0": f_v0,
                        "upward": g2 > g1,
                    }
                )

    return threshold_crossings


def make_vf_vg_plot(
    vf_vg_data: dict,
    suptitle: str = None,
    mode_floor_Hz: float = 0.0,
    damping_threshold: float = None,
    xaxes:list = ["keas", "ktas", "MACH"],
    xaxes_labels:list = ["KEAS", "KTAS", "MACH"],
    figure_savepath: str | Path = None,
):
    """Generates standard V-g and V-f subplots with threshold scanning and annotations.

    Parameters
    ----------
    vf_vg_data : dict
        Flutter results keyed by mode name containing velocity, damping, and frequency arrays.
    suptitle : str, optional
        Super-title for the figure.
    mode_floor_Hz : float, optional
        Minimum initial frequency at v=0 required to display a mode. Modes below this floor
        (e.g., rigid body modes) are filtered out. Default is 0.0.
    damping_threshold : float, optional
        Damping value (e.g., 0.03 or 0.0) to scan for flutter onset. The lowest-velocity crossing
        will be marked with a vertical line and an arrow annotation.
    """
    assert len(xaxes) == len(xaxes_labels)

    def make_converter(x_from, x_to):
        """Creates an interpolation function between two monotonic 1D arrays."""
        if x_from[0] > x_from[-1]:  # np.interp requires x_from to be increasing
            x_from, x_to = x_from[::-1], x_to[::-1]
        return lambda val: np.interp(val, x_from, x_to)

    fig, (ax_vg, ax_vf) = plt.subplots(
        2, 1, figsize=(9, 8), 
        sharex=True, 
        constrained_layout=True
    )

    v_all = []

    for key, data in vf_vg_data.items():
        damping = data["damping"]
        frequency = data["frequency"]
        v_datas = [data[x.lower()] for x in xaxes]
        

        # 1. Identify frequency at lowest velocity (v = 0)
        v0_idx = np.argmin(np.abs(v_datas[0]))
        f_v0 = frequency[v0_idx]

        # Filter out modes below mode_floor_Hz
        if f_v0 < mode_floor_Hz:
            continue

        v_all.append(v_datas[0])

        # 2. Add frequency to legend label: f"{key} = {f_v0:.2f} Hz"
        mode_label = f"{key} = {f_v0:.2f} Hz"

        ax_vg.plot(
            v_datas[0],
            damping,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=mode_label,
        )
        ax_vf.plot(
            v_datas[0],
            frequency,
            marker="s",
            markersize=3,
            linewidth=1.2,
            label=mode_label,
        )


        # 3. Check for threshold crossing via linear interpolation
        threshold_crossings = []
        if damping_threshold is not None:
            threshold_crossings = get_flutter_onset(vf_vg_data=vf_vg_data,
                                                    damping_threshold=damping_threshold)

            #trim threshold_crossings for modes below mode_floor_Hz
            threshold_crossings = [x for x in threshold_crossings if vf_vg_data[x["mode"]]["frequency"][0] >= mode_floor_Hz]

    ax_vf.set_xlabel(xaxes_labels[0])

    if len(xaxes) > 1: 
        for i,_ in enumerate(xaxes):
            if i == 0: continue  
            ax = ax_vf.secondary_xaxis(-0.22*i, functions=(make_converter(v_datas[0], v_datas[i]), make_converter(v_datas[i], v_datas[0])))
            ax.set_xlabel(xaxes_labels[i])
    
    # Reference lines on V-g plot
    if damping_threshold is not None and damping_threshold != 0.0:
        ax_vg.axhline(
            damping_threshold,
            color="orange",
            linestyle=":",
            linewidth=1.2,
            label=f"Threshold ($g={damping_threshold}$)",
        )
    

    # Handle lowest-velocity crossing annotation
    if damping_threshold is not None and len(threshold_crossings) > 0:
        threshold_crossings.sort(key=lambda x: x[f"{xaxes[0].lower()}_cross"])
        crit = threshold_crossings[0]
        v_crit = crit[f"{xaxes[0].lower()}_cross"]
        m_crit = crit["mode"]

        # Vertical line indicating onset speed on both subplots
        ax_vg.axvline(
            v_crit,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Crossing: {v_crit:.2f}",
        )
        ax_vf.axvline(v_crit, color="black", linestyle="--", linewidth=1.2)

        # Arrow callout pointing at crossing coordinate (v_crit, damping_threshold)
        v_span = (
            np.max([v.max() for v in v_all]) - np.min([v.min() for v in v_all])
            if v_all
            else 1.0
        )
        x_text_offset = 0.05 * v_span

        ax_vg.annotate(
            f"{m_crit} unstable at $g={damping_threshold}$\nat {xaxes_labels[0]} = ${v_crit:.2f}$",
            xy=(v_crit, damping_threshold),
            xytext=(v_crit + x_text_offset, damping_threshold + 0.05),
            arrowprops=dict(
                facecolor="black", shrink=0.08, width=1.2, headwidth=6
            ),
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.75,
            ),
        )
    

    # Axis formatting
    ax_vf.set_xlim([np.nanmin(v_datas[0]), np.nanmax(v_datas[0])])
    ax_vg.set_ylabel("Damping, $g$")
    ax_vg.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax_vf.set_ylabel("Frequency (Hz)")

    if suptitle:
        fig.suptitle(suptitle)

    if figure_savepath is not None: 
        try: 
            fig.savefig(figure_savepath._str.split(".")[0] + ".png")
        except: 
            print(f"Failed to save figure to {figure_savepath._str}")