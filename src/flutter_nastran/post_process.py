from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from pyNastran.f06.parse_flutter import make_flutter_response
from scipy.optimize import linear_sum_assignment

plt.style.use("bmh")

# Copyright (C) <2026>, Streamline Aerospace. All rights reserved.


def extract_vf_vg_data(f06_path: str) -> dict:
    """Extracts raw flutter data from a NASTRAN SOL 145 .f06 file using pyNastran."""
    flutters = make_flutter_response(f06_path)

    if isinstance(flutters, dict):
        subcase_id = list(flutters.keys())[0]
        flutter = flutters[subcase_id]
    else:
        flutter = flutters

    i_vel = flutter.ivelocity  # Index 4
    i_dmp = flutter.idamping  # Index 5
    i_frq = flutter.ifreq  # Index 6

    results = flutter.results
    modes = flutter.modes

    res_dict = {}
    for mode in modes:
        res_dict[f"Mode {mode}"] = {
            "velocity": results[mode - 1][:, i_vel].copy(),
            "damping": results[mode - 1][:, i_dmp].copy(),
            "frequency": results[mode - 1][:, i_frq].copy(),
        }

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

    num_steps = len(vf_vg_data[mode_keys[0]]["velocity"])

    freq_raw = np.array([vf_vg_data[k]["frequency"] for k in mode_keys])
    damp_raw = np.array([vf_vg_data[k]["damping"] for k in mode_keys])
    vel_raw = np.array([vf_vg_data[k]["velocity"] for k in mode_keys])

    repaired_freq = np.zeros_like(freq_raw)
    repaired_damp = np.zeros_like(damp_raw)
    repaired_vel = np.zeros_like(vel_raw)

    repaired_freq[:, 0] = freq_raw[:, 0]
    repaired_damp[:, 0] = damp_raw[:, 0]
    repaired_vel[:, 0] = vel_raw[:, 0]

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
        cand_v = vel_raw[:, k]

        if k == 1:
            pred_f = repaired_freq[:, 0]
            pred_d = repaired_damp[:, 0]
        else:
            dv_prev = repaired_vel[:, k - 1] - repaired_vel[:, k - 2]
            dv_curr = cand_v - repaired_vel[:, k - 1]
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
            repaired_vel[track_i, k] = cand_v[cand_j]

    repaired_dict = {}
    for i, mode_key in enumerate(mode_keys):
        repaired_dict[mode_key] = {
            "velocity": repaired_vel[i],
            "damping": repaired_damp[i],
            "frequency": repaired_freq[i],
        }

    return repaired_dict


def make_vf_vg_plot(
    vf_vg_data: dict,
    suptitle: str = None,
    mode_floor_Hz: float = 0.0,
    damping_threshold: float = None,
    save_plot: bool = False,
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
    fig, (ax_vg, ax_vf) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, constrained_layout=True
    )

    threshold_crossings = []
    v_all = []

    for key, data in vf_vg_data.items():
        velocity = data["velocity"]
        damping = data["damping"]
        frequency = data["frequency"]

        # 1. Identify frequency at lowest velocity (v = 0)
        v0_idx = np.argmin(np.abs(velocity))
        f_v0 = frequency[v0_idx]

        # Filter out modes below mode_floor_Hz
        if f_v0 < mode_floor_Hz:
            continue

        v_all.append(velocity)

        # 2. Add frequency to legend label: f"{key} = {f_v0:.2f} Hz"
        mode_label = f"{key} = {f_v0:.2f} Hz"

        ax_vg.plot(
            velocity,
            damping,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=mode_label,
        )
        ax_vf.plot(
            velocity,
            frequency,
            marker="s",
            markersize=3,
            linewidth=1.2,
            label=mode_label,
        )

        # 3. Check for threshold crossing via linear interpolation
        if damping_threshold is not None:
            for i in range(len(velocity) - 1):
                g1, g2 = damping[i], damping[i + 1]
                v1, v2 = velocity[i], velocity[i + 1]

                # Upward crossing into instability or crossing through threshold
                if (g1 - damping_threshold) * (
                    g2 - damping_threshold
                ) <= 0 and g1 != g2:
                    v_cross = (
                        v1
                        + (damping_threshold - g1) / (g2 - g1) * (v2 - v1)
                    )
                    threshold_crossings.append(
                        {
                            "mode": key,
                            "v_cross": v_cross,
                            "f_v0": f_v0,
                            "upward": g2 > g1,
                        }
                    )

    # Reference lines on V-g plot
    ax_vg.axhline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="Flutter Limit ($g=0$)",
    )

    if damping_threshold is not None and damping_threshold != 0.0:
        ax_vg.axhline(
            damping_threshold,
            color="orange",
            linestyle=":",
            linewidth=1.2,
            label=f"Threshold ($g={damping_threshold}$)",
        )

    # 4. Handle lowest-velocity crossing annotation
    if damping_threshold is not None and len(threshold_crossings) > 0:
        threshold_crossings.sort(key=lambda x: x["v_cross"])
        crit = threshold_crossings[0]
        v_crit = crit["v_cross"]
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
            f"{m_crit} crosses $g={damping_threshold}$\nat $M={v_crit:.2f}$",
            xy=(v_crit, damping_threshold),
            xytext=(v_crit + x_text_offset, damping_threshold + 0.05),
            arrowprops=dict(
                facecolor="black", shrink=0.08, width=1.2, headwidth=6
            ),
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="#fff3cd",
                edgecolor="#856404",
                alpha=0.9,
            ),
        )

    # Axis formatting
    ax_vg.set_ylabel("Damping, $g$")
    ax_vg.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    ax_vf.set_xlabel("Mach Number")
    ax_vf.set_ylabel("Frequency (Hz)")

    savetitle=None
    if suptitle:
        fig.suptitle(suptitle)
        savetitle = suptitle + ".png"

    if save_plot: 
        fig.savefig(savetitle)


if __name__ == "__main__":
    search_path = Path("f06")
    matching_files = sorted(list(search_path.glob("*.f06")))

    for f06_file in matching_files:
        raw_data = extract_vf_vg_data(str(f06_file))
        clean_data = repair_vf_vg_curves(raw_data)

        # Plots structural modes (filtering rigid body modes < 1.0 Hz)
        # and annotates the lowest flutter crossing at g = 0.03
        make_vf_vg_plot(
            clean_data,
            suptitle=f06_file.name,
            mode_floor_Hz=0.001,
            damping_threshold=0.03,
            save_plot=True)