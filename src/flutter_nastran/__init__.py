"""
flutter_nastran: Automation and post-processing tools for NASTRAN SOL 145 flutter analysis.
"""

from flutter_nastran.bdf import flutter_card, generate_sol145_master_bdf, split_nastran_deck, add_whirl_flutter_dmig_to_bdf, update_struct_bdf_hub_grid
from flutter_nastran.post_process import extract_vf_vg_data, make_vf_vg_plot, repair_vf_vg_curves, get_flutter_onset, load_flutter_csv
from flutter_nastran.runner import run_nastran
from flutter_nastran.gui import select_directory, find_bdf_files, get_subdirectories
from flutter_nastran.stdatm1976 import air_density, air_temp, sonic_speed
from flutter_nastran.rotor import get_propeller_coeffs_bet, generate_whirl_flutter_dictionaries

__all__ = [
    "split_nastran_deck",
    "flutter_card",
    "generate_sol145_master_bdf",
    "add_whirl_flutter_dmig_to_bdf",
    "run_nastran",
    "extract_vf_vg_data",
    "repair_vf_vg_curves",
    "make_vf_vg_plot",
    "select_directory",
    "find_bdf_files",
    "get_subdirectories",
    "get_flutter_onset",
    "load_flutter_csv",
    "air_density",
    "air_temp",
    "sonic_speed"
    "get_propeller_coeffs_bet",
    "generate_whirl_flutter_dictionaries",
    "update_struct_bdf_hub_grid"
]