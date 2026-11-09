"""
flutter_nastran: Automation and post-processing tools for NASTRAN SOL 145 flutter analysis.
"""

from flutter_nastran.bdf import flutter_card, generate_sol145_master_bdf, split_nastran_deck
from flutter_nastran.post_process import extract_vf_vg_data, make_vf_vg_plot, repair_vf_vg_curves
from flutter_nastran.runner import run_nastran

__all__ = [
    "split_nastran_deck",
    "flutter_card",
    "generate_sol145_master_bdf",
    "run_nastran",
    "extract_vf_vg_data",
    "repair_vf_vg_curves",
    "make_vf_vg_plot",
]