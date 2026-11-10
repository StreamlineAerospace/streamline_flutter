from flutter_nastran import load_flutter_csv, select_directory, make_vf_vg_plot
from pathlib import Path
import matplotlib.pyplot as plt
"""
Run this script to generate matplotlib figures of .csv data exported from FlutterPost.exe
"""

if __name__ == "__main__":

    damping_threshold = 0.03
    
    working_dir = select_directory(prompt_title="Select Top Level Directory Containing .csv Files (OK if files are nested)")
    matching_files = list(Path(working_dir).rglob("*.csv"))

    for csv_file in matching_files: 
        try:
            res = load_flutter_csv(csv_path=csv_file)
            make_vf_vg_plot(
                    res,
                    suptitle=csv_file.name,
                    mode_floor_Hz=0.001,
                    damping_threshold=damping_threshold,
                    figure_savepath=csv_file)
        except: 
            print(f"Unable to plot data in {csv_file.name}. Continuing")
            continue

    plt.show()