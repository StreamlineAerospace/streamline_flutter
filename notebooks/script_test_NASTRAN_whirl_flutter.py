#Import Statements
from flutter_nastran import (
    split_nastran_deck,
    flutter_card2,
    generate_sol145_master_bdf2,
    select_directory,
    find_bdf_files,
    get_subdirectories,
    add_whirl_flutter_dmig_to_bdf2,
    get_propeller_coeffs_bet,
    sonic_speed,
    generate_whirl_flutter_dictionaries,
    extract_vf_vg_data2,
    get_flutter_onset,
    repair_vf_vg_curves,
    make_vf_vg_plot,
    update_struct_bdf_hub_grid,
    air_density)

import numpy as np
import warnings
from pathlib import Path
import os
import math

"""
Script setups up NASTRAN run files for SOL 145 flutter analysis and uses DMIG to add matrices 
corresponding to rotating propeller with a flexible mounting
"""

if __name__ == "__main__":

    nastran_path = Path(r"C:\FEMAPv1101\nastran\bin\nastran.exe")

    #declare flutter sweep parameters 
    damping_threshold = 0.03
    hub_grid_id = 44240 #ID of grid point corresponding to hub
    altitude_ft = 15000 #altitude to survey flutter over 
    Mach_min = 0.025
    Mach_max = 0.8
    Mach_num_pts = 32
    spc_id = 2 #! make sure this is set (can actually be read off of input .bdf file)
    ref_chord_in = 55
    flutter_freq_interval = [0.002, 50] #interval to calculate reduced frequencies from
    flutter_freq_max = 40 #maximum frequency for NASTRAN SOL 145 to survey up to  

    #### PROPELLER MODELING ####
    eta_sweep = [0.17, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.27, 1.0]
    c_cr_sweep = [0.25, 0.33, 0.40, 0.50, 0.70, 0.80, 0.90, 1.0, 1.0]
    eta_0 = 0.17
    c_r_ft = 0.75
    R_ft = 3
    Omega_rad_sec = 2000.0 * 2.0 * math.pi / 60.0
    N_b = 4
    c_cr_func = lambda eta: np.interp(eta, eta_sweep, c_cr_sweep)
    I_xx_slug_ft2 = 175
    I_xx_slinch_in2 = 12*I_xx_slug_ft2
    S_ref_ft2 = math.pi*R_ft**2

    #### Generating Propeller Aero, Gyro Matrices
    Mach_sweep = np.linspace(Mach_min, Mach_max, Mach_num_pts)
    a_ft_s = sonic_speed(h_ft=altitude_ft)
    rho_slug_ft3 = air_density(h_ft=altitude_ft)
    rho_slinch_in3 = rho_slug_ft3*4.82253e-5
    V_inf_ft_s_sweep = Mach_sweep*a_ft_s

    gyro_dicts, aero_damping_dicts, aero_stiffness_dicts = [],[],[]
    mach_matrices = {}

    for i,V_inf in enumerate(V_inf_ft_s_sweep): 
        res = get_propeller_coeffs_bet(
                N_b=N_b,
                Omega_rad_s=Omega_rad_sec,
                V_inf=V_inf,
                R=R_ft,
                c_r=c_r_ft,
                c_cr_func=c_cr_func,
                M=V_inf/a_ft_s,
                eta_0=min(eta_sweep),
                unsteady=True)

        #convert units
        V_inf_in_s = V_inf*12
        D_in = 2*R_ft*12

        #Propeller Stiffness Derivatives:
        C_k = {
            (2,5): res["C_y_theta"],
            (2,6): -res["C_z_theta"],
            (3,5): res["C_z_theta"],
            (3,6): res["C_y_theta"],
            (5,5): res["C_m_theta"],
            (5,6): -res["C_n_theta"],
            (6,5): res["C_n_theta"],
            (6,6): res["C_m_theta"]
        }

        #Propelelr Damping Derivatives
        C_b = {
            (2,5): res["C_y_q"],
            (2,6): -res["C_z_q"],
            (3,5): res["C_z_q"],
            (3,6): res["C_y_q"],
            (5,5): res["C_m_q"],
            (5,6): -res["C_n_q"],
            (6,5): res["C_n_q"],
            (6,6): res["C_m_q"]
        }

        #Propeller Damping Matrix

        gyro_dict, aero_damping_dict, aero_stiffness_dict = generate_whirl_flutter_dictionaries(omega=Omega_rad_sec,
                                                                                            Ixx=I_xx_slinch_in2,
                                                                                            rho=rho_slinch_in3,
                                                                                            V=V_inf_in_s,
                                                                                            D=D_in,
                                                                                            C_k=C_k,
                                                                                            C_b=C_b)

        mach_matrices[Mach_sweep[i]] = (gyro_dict, aero_damping_dict, aero_stiffness_dict)



    if True: 

        #FEMAP export bdf model (Must contain femap-exported *.bdf files)
        working_dir = select_directory(prompt_title="Select Local Working Directory")

        os.chdir(working_dir)

        analysis_model_bdfs = find_bdf_files(directory=working_dir)
        print(f"Found {len(analysis_model_bdfs)} BDF files: {analysis_model_bdfs}")

        #check if this working directory contains any subfolders 
        subdirs = get_subdirectories(directory=working_dir)
        if len(subdirs)>0:
            warnings.warn("Working Directory contains Subdirectories. Continuing may overwrite them")

        folder_paths = [] #store a list of paths to each sub-folder

        for model_bdf in analysis_model_bdfs:

            #get analysis model bdf base name
            base_name = model_bdf.name.split(".")[0] #remove file extension

            #create a folder inside the working directory
            folder_path = working_dir + "/" + base_name + f"_{int(altitude_ft/1000)}kft"
            os.makedirs(folder_path, exist_ok=True)
            folder_paths.append(folder_path)

            #generate *_STRUCT.bdf and *_AERO.bdf files 
            struc_path, aero_path = split_nastran_deck(input_bdf_path=model_bdf.name, output_dir=folder_path)

            #update structural bdf with custom coordinate system for hub grid node
            struc_path = update_struct_bdf_hub_grid(struct_bdf_filepath=struc_path,
                                                    hub_grid_id=hub_grid_id,
                                                    x_vector=[1,0,0],
                                                    z_vector=[0,0,1])
            #generate flutter cards 
            flutter_path = flutter_card2(altitude_ft=altitude_ft,
                            Mach_sweep=np.linspace(Mach_min, Mach_max, Mach_num_pts), 
                            freq_interval_Hz=flutter_freq_interval,
                            reference_chord_in=ref_chord_in,
                            output_dir=folder_path)

            #generate master run cards 
            run_file = generate_sol145_master_bdf2(flutter_card_path=flutter_path,
                                struct_card_path=struc_path,
                                aero_card_path=aero_path,
                                base_title="RV10_WING w/ Propeller",
                                output_dir=folder_path,
                                spc_id=spc_id,
                                max_freq_Hz=flutter_freq_max,
                                mach_list=Mach_sweep)


            run_file = add_whirl_flutter_dmig_to_bdf2(xRUN_bdf_filepath=run_file,
                                                    hub_grid_id = hub_grid_id, 
                                                    mach_matrices = mach_matrices)

    working_dir = select_directory(prompt_title="Select Top-Level Directory Containing .f06 File")
    matching_files = list(Path(working_dir).rglob("*.f06"))
    Mach_numbers = [] #mach numbers where onset of flutter is expected. Obtained from Vf-Vg diagrams

    for filepath in matching_files: 

        #check if file is a single-point complex results set
        if "complex" in filepath.name.lower(): continue 

        #process data in .f06 file 
        raw_data = extract_vf_vg_data2(filepath, altitude_ft=altitude_ft)
        crossings = get_flutter_onset(vf_vg_data=raw_data, damping_threshold=damping_threshold)
        crossings.sort(key=lambda x: x["mach_cross"])
        try: 
            Mach_numbers.append(crossings[0]["mach_cross"])
        except: Mach_numbers.append(None)

        #fix errors in vf-vg curves due to mode crossings 
        clean_data = repair_vf_vg_curves(raw_data)
        
        # Plots structural modes
        make_vf_vg_plot(
            clean_data,
            suptitle=filepath.name,
            mode_floor_Hz=1,
            damping_threshold=damping_threshold,
            figure_savepath=filepath,
            xaxes=["keas", "ktas"],
            xaxes_labels=["KEAS", "KTAS"])

    print(f"Unstable flutter onset detected at: Mach = {Mach_numbers}")