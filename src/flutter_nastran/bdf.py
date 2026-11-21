
import numpy as np
from flutter_nastran import stdatm1976 as atm 
from pyNastran.bdf.bdf import BDF
from datetime import datetime
import math
import os

"""
Copyright (C) <2026>, Streamline Aerospace. All rights reserved.
module contains functions for generating NASTRAN .bdf files for SOL 145 flutter analysis
"""


def flutter_card(
        altitude_ft:float, 
        Mach_sweep:list, 
        freq_interval_Hz:list, 
        reference_chord_in:float, 
        output_dir:str, 
        complex_mode_output:bool=False) -> str: 
    """Generates a NASTRAN .bdf file containing commands and parameters for SOL 145 flutter sweep.
    Generates distinct FLUTTER/FLFACT cards for each Mach number to support DMIG subcases.
    """
    if complex_mode_output: 
        assert np.size(Mach_sweep) == 1

    # Calculate velocity in in/sec 
    a_ft_s = atm.sonic_speed(h_ft=altitude_ft)
    a_in_s = a_ft_s * 12

    rho_slinch = 0.000578704 * atm.air_density(h_ft=altitude_ft)
    dens_factor = atm.air_density(h_ft=altitude_ft) / atm.air_density(h_ft=0)
    
    # Pre-calculate velocity bounds for each mach number based on frequency intervals
    freq_red_min_Hz = []
    freq_red_max_Hz = []
    velocities_in_s = []
    
    for m in Mach_sweep:
        v = m * a_in_s
        velocities_in_s.append(v)
        freq_red_min_Hz.append(min(freq_interval_Hz) * math.pi * reference_chord_in / v)
        freq_red_max_Hz.append(max(freq_interval_Hz) * math.pi * reference_chord_in / v)
    
    model = BDF(debug=False, log=None)

    # Custom Header Comment
    header_comment = f"""$=======================================================================
$Flutter Solution Data:          {int(altitude_ft/1000)}kft - KEAS Sweep
$=======================================================================
$
$Altitude:             {altitude_ft:,.0f} ft
$Air density:          {rho_slinch:.3E} slinch/in^3
$Speed of sound:       {a_ft_s:,.1f} ft/sec
$                        {a_in_s:,.1f} in/sec
$
$-----------------------------------------------------------------------
$Input deck created:   {datetime.now().strftime('%d-%b-%y')}
$-----------------------------------------------------------------------"""

    # -------------------------------------------------------------------------
    # PARAM & AERO entries
    # -------------------------------------------------------------------------
    model.add_param("VREF", [a_in_s], comment="Convert flutter output velocities from in/sec to Mach number")
    model.add_param("SUPAERO", ["CPM"], comment="Use the CPM method for supersonic conditions")
    if complex_mode_output:
        model.add_param("OPPHIPA", [0], comment="Output mode shapes to check spline interpolation")

    model.add_aero(
        cref=float(reference_chord_in),
        rho_ref=1.146E-7,
        velocity=None,
        acsid=0,
        sym_xz=0,
        sym_xy=0,
        comment="Aerodynamic physical data",
    )

    dens_sid = 1001
    model.add_flfact(dens_sid, [round(dens_factor, 3)], comment="Density ratios for flutter solutions")

    # -------------------------------------------------------------------------
    # Generate Subcase-Specific FLUTTER & FLFACT entries
    # -------------------------------------------------------------------------
    for i, m in enumerate(Mach_sweep):
        # Create a unique ID for each Mach number (e.g., Mach 0.3 -> 1030)
        fmethod_id = 1000 + int(round(m * 100))
        mach_sid = fmethod_id + 1
        veloc_sid = fmethod_id + 2
        
        mach_factor = round(m, 4)
        
        # Generate a narrow velocity band around the matched point velocity
        v_base = velocities_in_s[i]
        v_band = [v_base]
        # Example narrow band: 0.8x to 1.2x matched velocity
        if complex_mode_output:
            v_band = [-v for v in v_band]
            
        veloc_factors = [round(v, 1) for v in v_band]

        model.add_flutter(
            sid=fmethod_id,
            method="PKNL",
            density=dens_sid,
            mach=mach_sid,
            reduced_freq_velocity=veloc_sid,
            comment=f"Flutter solution data for Mach {mach_factor}",
        )
        
        model.add_flfact(mach_sid, [mach_factor])
        model.add_flfact(veloc_sid, veloc_factors)
        
        # MKAERO1 entries (Global)
        k_scaled = np.logspace(np.log10(freq_red_min_Hz[i]), np.log10(freq_red_max_Hz[i]), num=8)
        model.add_mkaero1(machs=[mach_factor], reduced_freqs=k_scaled)

    # -------------------------------------------------------------------------
    # Write Out 
    # -------------------------------------------------------------------------
    output_path = f"{output_dir}/flutter_{int(altitude_ft/1000)}kft"
    if complex_mode_output: 
        output_path += "_complex"
    output_path += ".bdf"

    model.write_bdf(output_path, size=8, is_double=False)
    
    with open(output_path, "r") as f:
        deck_lines = [line for line in f if not line.strip().startswith("$pyNastran")]
    with open(output_path, "w") as f:
        f.write(header_comment + "\n" + "".join(deck_lines))

    return output_path


def generate_sol145_master_bdf(
        flutter_card_path:str, 
        struct_card_path:str, 
        aero_card_path:str, 
        output_dir:str, 
        spc_id:int, 
        base_title:str,
        mach_list:list,  
        max_freq_Hz:float, 
        min_freq_Hz:float=None, 
        complex_mode_output:bool=False, 
        aerosymflag:str="SYMMETRIC") -> str:
    """Generates a master run .bdf file for SOL 145 flutter analysis.
    Adds subcase blocks if mach_list is provided to support DMIG matching.
    """
    fid = 1000 
    flutter_card_name = flutter_card_path.split("/")[-1]
    title = struct_card_path.split("/")[-1]
    title = title.split("_STRUCT")[0]
    output_file = f"{output_dir}/xRUN_{title}_{flutter_card_name.split('.')[0]}.bdf"
 
    pre_bulk_lines = ["INIT MASTER(S)\n",
                        "NASTRAN SYSTEM(442)=-1,SYSTEM(319)=1\n",
                        "SOL 145\n",
                        "TIME 10000\n",
                        "$DIAG 12\n", 
                        "GEOMCHECK, NONE\n",
                        "CEND\n",
                        f"    TITLE = {base_title}\n",
                        "    ECHO = NONE\n",
                        f"    METHOD = {fid}\n", 
                        f"    AESYMXZ = {aerosymflag}\n",
                        f"    SPC={spc_id}\n"]
    
    if complex_mode_output: 
        pre_bulk_lines.append("    DISP=ALL\n")

    # If no explicit mach list is passed, write the global FMETHOD parameter
    if mach_list is None:
        pre_bulk_lines.insert(-2, "    FMETHOD = 1000\n")
    # If mach list is passed, define separate subcases for DMIG mapping
    else:
        for idx, mach in enumerate(mach_list):
            fmethod_id = 1000 + int(round(mach * 100))
            pre_bulk_lines.append(f"\nSUBCASE {idx + 1}\n")
            pre_bulk_lines.append(f"    LABEL = Mach {mach:.3f} Matched Point\n")
            pre_bulk_lines.append(f"    FMETHOD = {fmethod_id}\n")

    pre_bulk_lines.append("\nBEGIN BULK\n")

    model = BDF(debug=False, log=None)
    model.add_param("POST", [-1])
    model.add_param("OGEOM", ["NO"])
    model.add_param("AUTOSPC", ["NO"])
    model.add_param("AUTOSPRT", ["NO"])
    model.add_param("WTMASS", [0.00259])

    if min_freq_Hz is not None: 
        model.add_param("LFREQ", [min_freq_Hz])

    model.add_eigrl(
        sid=fid,
        v1=min_freq_Hz,
        v2=float(max_freq_Hz),
        msglvl=0,
        norm="MASS",
    )

    temp_bulk_path = output_file + ".tmp"
    model.write_bdf(temp_bulk_path, size=8, is_double=False)

    with open(temp_bulk_path, "r") as f:
        bulk_cards = [line for line in f if not line.strip().startswith("$pyNastran") and not line.strip().startswith("ENDDATA")]

    if os.path.exists(temp_bulk_path):
        os.remove(temp_bulk_path)

    include_section = (
        "$\n"
        "$Structural Model\n"
        f"INCLUDE '{struct_card_path.split('/')[-1]}'\n"
        "$\n"
        "$Aero Model\n"
        f"INCLUDE '{aero_card_path.split('/')[-1]}'\n"
        "$\n"
        "$Flutter Card   \n"
        f"INCLUDE '{flutter_card_path.split('/')[-1]}'\n"
        "$\n"
        "ENDDATA\n")

    with open(output_file, "w") as f:
        [f.write(line) for line in pre_bulk_lines]
        f.writelines(bulk_cards)
        f.write(include_section)

    return output_file


def split_nastran_deck(
        input_bdf_path: str, 
        output_dir: str) -> list:
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


def add_whirl_flutter_dmig_to_bdf(
    xRUN_bdf_filepath: str,
    hub_grid_id: int,
    mach_matrices: dict,
    num_complex_roots: int = 30
):
    """
    Applies whirl flutter DMIG matrices across multiple Mach numbers, adds the EIGC complex
    eigenvalue solver, and links the unique matrices to pre-existing SUBCASE blocks.

    Args:
        xRUN_bdf_filepath (str): Path to the master xRUN .bdf file.
        hub_grid_id (int): Structural GRID ID of the propeller hub.
        mach_matrices (dict): {mach_float: (gyro_dict, aero_damping_dict, aero_stiffness_dict)}
        num_complex_roots (int): Number of desired complex roots (ND) on the EIGC card.
    """
    if not isinstance(hub_grid_id, int):
        raise ValueError(f"hub_grid_id must be an integer, got {type(hub_grid_id)}: {hub_grid_id}")

    cont_counter = 1

    def next_cont() -> str:
        nonlocal cont_counter
        tag = f"*C{cont_counter:<6d}"  # Exactly 8 characters
        cont_counter += 1
        return tag

    def format_dmig_large_field(matrix_name: str, matrix_dict: dict) -> list[str]:
        """Formats a square unsymmetric double-precision DMIG matrix in 16-char large-field format."""
        if not matrix_dict:
            return []

        lines = [
            "$" + "-" * 79 + "\n",
            f"$ DMIG Matrix {matrix_name}\n"
        ]

        # Header entry: GJ=0, IFO=1 (Square), TIN=2 (Double Prec), TOUT=2 (Double Prec), POLAR=0
        c_hdr = next_cont()
        lines.append(f"{'DMIG*':<8}{matrix_name:>16}{0:>16d}{1:>16d}{2:>16d}{c_hdr}\n")
        lines.append(f"{c_hdr}{2:>16d}{0:>16d}{'':>16}{'':>16}\n")

        # Column and Row entries
        for col in sorted(matrix_dict.keys()):
            rows = sorted(matrix_dict[col].keys())
            if not rows:
                continue

            c_curr = next_cont()
            lines.append(f"{'DMIG*':<8}{matrix_name:>16}{hub_grid_id:>16d}{int(col):>16d}{'':>16}{c_curr}\n")

            for idx, row in enumerate(rows):
                val = float(matrix_dict[col][row])
                has_more = idx < (len(rows) - 1)
                c_next = next_cont() if has_more else ""
                lines.append(
                    f"{c_curr}{hub_grid_id:>16d}{int(row):>16d}{val:>16.9E}{'':>16}{c_next}\n"
                )
                c_curr = c_next

        return lines

    # 1. Build all DMIG and EIGC bulk data lines
    dmig_cards = []
    for mach in sorted(mach_matrices.keys()):
        gyro_dict, damping_dict, stiffness_dict = mach_matrices[mach]
        m_id = int(round(float(mach) * 100))
        k_name = f"KPROP{m_id}"
        b_name = f"BPROP{m_id}"

        # Apply Nastran Left-Hand-Side sign conventions:
        # [K_DMIG] = -[K_aero]
        # [B_DMIG] = [G_gyro] - [B_aero]
        k_dmig = {}
        for (r, c), val in stiffness_dict.items():
            if abs(val) > 1e-12:
                k_dmig.setdefault(c, {})[r] = -float(val)

        b_dmig = {}
        all_b_keys = set(gyro_dict.keys()).union(set(damping_dict.keys()))
        for (r, c) in all_b_keys:
            val = float(gyro_dict.get((r, c), 0.0)) - float(damping_dict.get((r, c), 0.0))
            if abs(val) > 1e-12:
                b_dmig.setdefault(c, {})[r] = val

        dmig_cards.extend(format_dmig_large_field(k_name, k_dmig))
        dmig_cards.extend(format_dmig_large_field(b_name, b_dmig))

    # Add EIGC card (Field 8 = ND = num_complex_roots)
    dmig_cards.append("$DYNAMIC\n")
    dmig_cards.append(
        f"{'EIGC':<8}{20:>8d}{'HESS':>8}{'MAX':>8}{'':>8}{'':>8}{'':>8}{num_complex_roots:>8d}\n"
    )

    # 2. Inject Case Control selections and Bulk Data cards into a new _WHIRL deck
    file_dir, file_name = os.path.split(xRUN_bdf_filepath)
    name, ext = os.path.splitext(file_name)
    output_bdf_path = os.path.join(file_dir, f"{name}_WHIRL{ext}")

    with open(xRUN_bdf_filepath, "r") as fin, open(output_bdf_path, "w") as fout:
        case_control = True
        for line in fin:
            if line.strip() == "BEGIN BULK":
                case_control = False

            if line.strip() == "ENDDATA":
                fout.writelines(dmig_cards)
                fout.write(line)
                continue

            if case_control:
                if line.strip().startswith("METHOD"):
                    fout.write(line)
                    fout.write("    CMETHOD = 20\n")
                    continue

                if line.strip().startswith("FMETHOD"):
                    fout.write(line)
                    parts = line.split("=")
                    if len(parts) == 2:
                        try:
                            fmethod_id = int(parts[1].strip())
                            if fmethod_id > 1000:
                                m_id = fmethod_id - 1000
                                fout.write(f"    K2PP = KPROP{m_id}\n")
                                fout.write(f"    B2PP = BPROP{m_id}\n")
                        except ValueError:
                            pass
                    continue

            fout.write(line)

    print(f"Successfully generated DMIG matched-point deck: {output_bdf_path}")
    return output_bdf_path


def update_struct_bdf_hub_grid(
    struct_bdf_filepath: str, 
    hub_grid_id: int, 
    x_vector: list, 
    z_vector: list
) -> str:
    """
    Directly modifies a Nastran structural BDF text file to create a custom local 
    coordinate system and assigns it to the CD field of a specific GRID card.
    """
    
    # 1. Parse the file to find the Hub GRID coordinates and the highest CORD ID
    hub_found = False
    origin = np.zeros(3)
    max_cid = 0
    lines = []
    
    with open(struct_bdf_filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if line.startswith('CORD2') or line.startswith('CORD1'):
            try:
                cid = int(line[8:16].strip())
                if cid > max_cid:
                    max_cid = cid
            except ValueError:
                pass
                
        # Look for the target GRID card
        if line.startswith('GRID    ') or line.startswith('GRID,'):
            # Parse fixed 8-character fields
            if ',' not in line:
                gid_str = line[8:16].strip()
                if gid_str and int(gid_str) == hub_grid_id:
                    hub_found = True
                    # Fields 3, 4, 5 are X, Y, Z
                    x_str = line[24:32].strip()
                    y_str = line[32:40].strip()
                    z_str = line[40:48].strip()
                    
                    # Nastran sometimes omits the 'E' in scientific notation (e.g., 3.6-4)
                    def parse_nastran_float(s):
                        if '-' in s[1:]:
                            s = s[0] + s[1:].replace('-', 'E-')
                        if '+' in s[1:]:
                            s = s[0] + s[1:].replace('+', 'E+')
                        return float(s)
                        
                    origin[0] = parse_nastran_float(x_str)
                    origin[1] = parse_nastran_float(y_str)
                    origin[2] = parse_nastran_float(z_str)

    if not hub_found:
        raise ValueError(f"GRID {hub_grid_id} not found in {struct_bdf_filepath}.")

    # 2. Calculate the new CORD2R geometry
    new_cid = max_cid + 1
    z_point = origin + np.array(z_vector)
    x_point = origin + np.array(x_vector)

    # 3. Format the new CORD2R card (8-character fields)
    cord2r_lines = [
        "$ Propeller Hub Local Coordinate System (DMIG Alignment)\n",
        f"{'CORD2R':<8}{new_cid:>8d}{0:>8d}{origin[0]:>8.2f}{origin[1]:>8.2f}{origin[2]:>8.2f}{z_point[0]:>8.2f}{z_point[1]:>8.2f}+CRD{new_cid:03d}\n",
        f"+CRD{new_cid:03d}{z_point[2]:>8.2f}{x_point[0]:>8.2f}{x_point[1]:>8.2f}{x_point[2]:>8.2f}\n"
    ]

    # 4. Write the modified file back out
    with open(struct_bdf_filepath, 'w') as f:
        # Inject the CORD2R card right after the structural header
        injected_cord = False
        for line in lines:
            if not injected_cord and not line.startswith('$'):
                f.writelines(cord2r_lines)
                injected_cord = True
                
            # Modify the target GRID card
            if line.startswith('GRID    ') and int(line[8:16].strip()) == hub_grid_id:
                # Nastran GRID format:
                # 1:GRID, 2:ID, 3:CP, 4:X1, 5:X2, 6:X3, 7:CD, 8:PS, 9:SEID
                # We need to overwrite characters 48-56 (Field 7, CD)
                prefix = line[:48]
                suffix = line[56:] if len(line) > 56 else "\n"
                f.write(f"{prefix}{new_cid:>8d}{suffix}")
            else:
                f.write(line)
                
    print(f"Successfully added CORD2R {new_cid} and updated GRID {hub_grid_id} in {struct_bdf_filepath}.")
    return struct_bdf_filepath