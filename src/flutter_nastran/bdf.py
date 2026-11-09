
import numpy as np
from flutter_nastran import stdatm1976 as atm 
from pyNastran.bdf.bdf import BDF
from datetime import datetime
import os

"""
Copyright (C) <2026>, Streamline Aerospace. All rights reserved.
module contains functions for generating NASTRAN .bdf files for SOL 145 flutter analysis
"""

def flutter_card(altitude_ft:float, Mach_sweep:list, freq_interval_Hz:list, reference_chord_in:float, output_dir:str, complex_mode_output:bool=False) -> str: 
    """Generates a NASTRAN .bdf file containing commands and parameters for SOL 145 flutter sweep

    Args:
        altitude_ft (float): MSL altitude in feet
        Mach_sweep (list): Linear spaced, increasing, array of Mach numbers to sweep over 
        freq_interval_Hz (list): List of min and max frequency (in Hz) to survey flutter over
        reference_chord_in (float): Reference chord length (typically mean aero chord) in inches
        complex_mode_output (bool, optional): If True, sets up flutter card for single-point complex mode shape output for subsquent flutter animation. Defaults to False.

    Returns:
        str: filename of generated .bdf file
    """

    if complex_mode_output: 
        assert np.size(Mach_sweep) == 1

    #calculate velocity in in/sec 
    a_ft_s = atm.sonic_speed(h_ft=altitude_ft)
    rho_slinch = 0.000578704*atm.air_density(h_ft=altitude_ft)
    dens_factor = atm.air_density(h_ft=altitude_ft)/atm.air_density(h_ft=0)
    v_in_s = Mach_sweep*a_ft_s*12

    #calculate min and max reduced frequencies
    freq_red_min_Hz = min(freq_interval_Hz)*math.pi*reference_chord_in/v_in_s
    freq_red_max_Hz = max(freq_interval_Hz)*math.pi*reference_chord_in/v_in_s
    
    model = BDF()

    # Custom Header Comment
    header_comment = f"""$=======================================================================
$Flutter Solution Data:          {int(altitude_ft/1000)}kft - KEAS Sweep
$=======================================================================
$
$Altitude:             {altitude_ft:,.0f} ft
$Air density:          {rho_slinch:.3E} slinch/in^3
$Speed of sound:       {a_ft_s:,.1f} ft/sec
$                        {a_ft_s*12:,.1f} in/sec
$
$-----------------------------------------------------------------------
$Input deck created:   {datetime.now().strftime('%d-%b-%y')}
$-----------------------------------------------------------------------"""

    # -------------------------------------------------------------------------
    # PARAM entries
    # -------------------------------------------------------------------------
    model.add_param("VREF", [a_ft_s*12], comment="Convert flutter output velocities from in/sec to Mach number")#[cite: 1]
    model.add_param("SUPAERO", ["CPM"], comment="Use the CPM method for supersonic conditions")#[cite: 1]
    if complex_mode_output:
        model.add_param(
            "OPPHIPA",
            [0],
            comment="Output mode shapes to check the spline interpolation (also need DISP=ALL)",)

    # -------------------------------------------------------------------------
    # AERO entry
    # -------------------------------------------------------------------------
    model.add_aero(
        cref=float(reference_chord_in),
        rho_ref=1.146E-7, #! not sure why this is constant, but reference code has it this way
        velocity=None,
        acsid=0,
        sym_xz=0,
        sym_xy=0,
        comment="Aerodynamic physical data\n AeroCS Veloc RefC RhoRef SymXZ SymXY",)

    # -------------------------------------------------------------------------
    # FLUTTER entry
    # -------------------------------------------------------------------------
    flutter_sid = 1000
    dens_sid = 1001
    mach_sid = 1002
    veloc_sid = 1003

    model.add_flutter(
        sid=flutter_sid,
        method="PKNL",
        density=dens_sid,
        mach=mach_sid,
        reduced_freq_velocity=veloc_sid,
        comment="Flutter solution data\n Use PK non-looping (PKNL) flutter solution method",)

    # -------------------------------------------------------------------------
    # FLFACT entries
    # -------------------------------------------------------------------------
    mach_factors = [round(m, 4) for m in Mach_sweep]
    dens_factors = [round(dens_factor, 3)] * len(mach_factors)
    veloc_factors = [round(v, 1) for v in v_in_s]

    if complex_mode_output:
        veloc_factors = [v*-1 for v in veloc_factors]

    model.add_flfact(dens_sid, dens_factors, comment="Density ratios for match point flutter solutions")
    model.add_flfact(mach_sid, mach_factors, comment="Mach numbers for match point flutter solutions")
    model.add_flfact(veloc_sid, veloc_factors, comment="Velocities for match point flutter solutions")

    # -------------------------------------------------------------------------
    # MKAERO1 entries
    # -------------------------------------------------------------------------
    for i,m in enumerate(mach_factors):
        # Scale k values inversely with Mach (or supply arbitrary dynamic lists)
        k_scaled = np.logspace(np.log10(freq_red_min_Hz[i]), np.log10(freq_red_max_Hz[i]), num=8)
        model.add_mkaero1(machs=[m], reduced_freqs=k_scaled)

    # -------------------------------------------------------------------------
    # Write Out and Re-parse to Validate
    # -------------------------------------------------------------------------
    output_path = f"{output_dir}/flutter_{int(altitude_ft/1000)}kft"
    if complex_mode_output: 
        output_path += "_complex"
    output_path += ".bdf"

    model.write_bdf(output_path, size=8, is_double=False)
    print(f"Deck successfully generated: {output_path}")

    # Self-validation check
    verify_model = BDF()
    verify_model.read_bdf(output_path, xref=False)

    #add header comment and delete unnecessary comments from pyNastran
    with open(output_path, "r") as f:
        deck_lines = [
            line for line in f if not line.strip().startswith("$pyNastran")
        ]
    with open(output_path, "w") as f:
        f.write(header_comment + "".join(deck_lines))

    return output_path


def generate_sol145_master_bdf(flutter_card_path:str, struct_card_path:str, aero_card_path:str, output_dir:str, spc_id:int, base_title:str, max_freq_Hz:float, min_freq_Hz:float=None, complex_mode_output:bool=False, aerosymflag:str="SYMMETRIC") -> str:
    """Generates a master run .bdf fil SOL 145 flutter analysis

    Args:
        flutter_card_path (str): filepath of flutter .bdf file generated by flutter_card()
        struct_card_path (str): filepath of structural .bdf file generated by split_nastran_deck()
        aero_card_path (str): filepath of aero .bdf file generated by split_nastran_deck()
        spc_id (int): index of structural model constraint set to be applied
        base_title (str): short title describing analysis: Structure configuration, altitude, velocities, etc. 
        max_freq_Hz (float): Maximum frequency, in Hz, to instruct solver to survey up to
        min_freq_Hz (float, optional): frequency, in Hz, to ignore modes below - Use for filtering out rigid body mode
        complex_mode_output (bool, optional): If True, sets up flutter card for single-point complex mode shape output for subsquent flutter animation. Defaults to False.
        aerosymflag (str, optional): _description_. Defaults to "SYMMETRIC".

    Returns:
        str: _description_
    """
    fid = 30 #flutter identification number #!Hardcoded

    flutter_card_name = flutter_card_path.split("/")[-1]
    output_file = f"{output_dir}/xRUN_{base_title}_{flutter_card_name.split(".")[0]}.bdf"
 
    pre_bulk_lines = ["INIT MASTER(S)\n",
                        "NASTRAN SYSTEM(442)=-1,SYSTEM(319)=1\n",
                        "SOL 145\n", #NASTRAN solver
                        "TIME 10000\n", #maximum CPU time (in minutes) allowed for the job
                        "$DIAG 12\n", #Requests diagnostic output or special options.
                        "GEOMCHECK, NONE\n",
                        "CEND\n",
                        f"    TITLE = {base_title}\n",
                        "    ECHO = NONE\n", #NONE Suppresses the printing of grid points, elements, and bulk data listings in the output file to keep the .f06 file compact
                        f"    METHOD = {fid}\n", #The METHOD command can be used to select a method for real eigenvalue extraction.
                        "    FMETHOD = 1000\n", #Used to specify the parameters for aerodynamic flutter analysis, by referencing a FLUTTER Bulk Data Entry.
                        f"    AESYMXZ = {aerosymflag}\n", #aerodynamic xz plane of symmetry flag [SYMMETRIC/ANTISYMMETRIC/ASYMMETRIC]
                        f"    SPC={spc_id}\n",] #Identification number of single-point constraint set.
    if complex_mode_output: 
        pre_bulk_lines.append("    DISP=ALL\n")

    pre_bulk_lines.append("BEGIN BULK\n")

    model = BDF()
    # PARAM Cards
    model.add_param("POST", [-1]) #Generates an OUTPUT2 results file.
    model.add_param("OGEOM", ["NO"]) #output model data to .op2 file? 
    model.add_param("AUTOSPC", ["NO"]) #Automatically constrain degrees-of-freedom with no stiffness.
    model.add_param("AUTOSPRT", ["NO"]) #Activate inertia relief and auto-support degree-of-freedom generation.
    model.add_param("WTMASS", [0.00259]) #Converts weights to masses using this multiplier.

    if min_freq_Hz is not None: 
        assert min_freq_Hz > 0 and min_freq_Hz < max_freq_Hz
        model.add_param("LFREQ", [min_freq_Hz]) #Used to remove the Rigid Body Modes from the Modal Space.

    # EIGRL Card
    model.add_eigrl(
        sid=fid,
        v1=min_freq_Hz,
        v2=float(max_freq_Hz),
        msglvl=0,
        norm="MASS",
        comment="EIGRL card for eigenvalue extraction",
    )

    # Write the cards to a temporary file via pyNastran
    temp_bulk_path = output_file + ".tmp"
    model.write_bdf(temp_bulk_path, size=8, is_double=False)


    with open(temp_bulk_path, "r") as f:
        bulk_cards = [line for line in f if not line.strip().startswith("$pyNastran") and not line.strip().startswith("ENDDATA")]

    if os.path.exists(temp_bulk_path):
        os.remove(temp_bulk_path)

    # Comments and INCLUDE statements for remaining bulk data
    include_section = (
        "$\n"
        "$Structural Model\n"
        f"INCLUDE '{struct_card_path.split("/")[-1]}'\n"
        "$\n"
        "$Aero Model\n"
        f"INCLUDE '{aero_card_path.split("/")[-1]}'\n"
        "$\n"
        "$Flutter Card   \n"
        f"INCLUDE '{flutter_card_path.split("/")[-1]}'\n"
        "$\n"
        "ENDDATA\n")

    with open(output_file, "w") as f:
        [f.write(line) for line in pre_bulk_lines]
        
        f.writelines(bulk_cards)
        f.write(include_section)

    print(f"Master SOL 145 deck generated: {output_file}")

    return output_file


def split_nastran_deck(input_bdf_path: str, output_dir: str) -> list:
    """Splits a combined Nastran BDF into structural and aerodynamic models, discarding solution and executive control directives.
    Args:
        input_bdf_path (str): filepath of NASTRAN .bdf model from FEMAP file>export>analysis model

    Returns:
        list (str): [structural_model_bdf, aero_model_bdf]
    """
    # Aerodynamic card types (add keywords as needed)
    aero_cards = {
        "AERO",
        "AEROS",
        "CAERO1",
        "CAERO2",
        "CAERO3",
        "CAERO4",
        "CAERO5",
        "PAERO1",
        "PAERO2",
        "PAERO3",
        "PAERO4",
        "PAERO5",
        "SPLINE1",
        "SPLINE2",
        "SPLINE3",
        "SPLINE4",
        "SPLINE5",
        "AEFACT",
        "AESTAT",
        "AESURF",
        "MKAERO1",
        "MKAERO2",
        "FLUTTER",
        "FLFACT",
        "GUST",
        "SET1",
    }

    # Solution/control keywords to discard (Executive, Case Control, Solution Params)
    control_keywords = {
        "INIT",
        "NASTRAN",
        "ID",
        "SOL",
        "TIME",
        "GEOMCHECK",
        "CEND",
        "TITLE",
        "ECHO",
        "SPC",
        "METHOD",
        "FMETHOD",
        "AESYMXZ",
        "AESYMXY",
        "BEGIN",
        "PARAM",
        "ENDDATA",
    }

    with open(input_bdf_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = f.readlines()

    # 1. Group multi-line continuation entries into complete logical cards
    cards = []
    current_card = []

    for line in raw_lines:
        line_clean = line.rstrip("\r\n")

        # Comment line: attach to current block or preserve as single-line card
        if line_clean.strip().startswith("$"):
            if current_card:
                cards.append(current_card)
                current_card = []
            cards.append([line_clean])
            continue

        # Blank line
        if not line_clean.strip():
            if current_card:
                cards.append(current_card)
                current_card = []
            continue

        # Continuation line (+ or *)
        if line_clean.startswith("+") or line_clean.startswith("*"):
            current_card.append(line_clean)
        else:
            # Start of a new card
            if current_card:
                cards.append(current_card)
            current_card = [line_clean]

    if current_card:
        cards.append(current_card)

    # 2. Categorize cards
    struc_deck = []
    aero_deck = []
    pending_comments = []

    for card in cards:
        first_line = card[0].strip()

        # Buffer comments so they stay attached to the card beneath them
        if first_line.startswith("$"):
            pending_comments.append(card[0])
            continue

        # Determine card keyword (handles comma-free or column fixed-field format)
        if "," in first_line:
            keyword = first_line.split(",")[0].strip().upper()
        else:
            keyword = first_line[:8].strip().upper()

        if len(keyword.split(" ")) > 1:
            keyword = keyword.split(" ")[0]

        # Ignore Executive / Case Control / PARAM cards
        #if keyword in control_keywords:
        #if any(key in keyword for key in control_keywords):
        if keyword in control_keywords:
            pending_comments = []  # Discard associated comments
            continue

        # Route to Aero or Structural
        if keyword in aero_cards:
        #if any(key in keyword for key in aero_cards):
            if pending_comments:
                aero_deck.extend(pending_comments)
                pending_comments = []
            aero_deck.extend(card)
        else:
            if pending_comments:
                struc_deck.extend(pending_comments)
                pending_comments = []
            struc_deck.extend(card)

    #filter out AERO command land that may have ended up in aero deck
    aero_deck = [line for line in aero_deck if not "AERO " in line]

    #make file names 
    base_file_name = input_bdf_path.split(".")[0]
    output_struc_path = output_dir + "/" + base_file_name + "_STRUCT.bdf"
    output_aero_path = output_dir + "/" + base_file_name + "_AERO.bdf"

    # 3. Write out both bdf files
    with open(output_struc_path, "w", encoding="utf-8") as f_struc:
        f_struc.write("$ ===================================================\n")
        f_struc.write("$ STRUCTURAL MODEL DECK\n")
        f_struc.write("$ ===================================================\n")
        f_struc.write("\n".join(struc_deck) + "\n")

    with open(output_aero_path, "w", encoding="utf-8") as f_aero:
        f_aero.write("$ ===================================================\n")
        f_aero.write("$ AERODYNAMIC MODEL DECK\n")
        f_aero.write("$ ===================================================\n")
        f_aero.write("\n".join(aero_deck) + "\n")

    print(
        f"Separation complete:\n  - Structural: {output_struc_path}\n  - Aero: {output_aero_path}"
    )

    return output_struc_path, output_aero_path