import math

"""
Module contains functions for rotordynamics calculations
"""

# Copyright (C) <2026>, Streamline Aerospace. All rights reserved.

def generate_whirl_flutter_dictionaries(omega, Ixx, rho, V, D, C_k, C_b):
    """
    Generates dimensional NASTRAN DMIG dictionaries for gyroscopic, aerodynamic 
    stiffness, and aerodynamic damping matrices.
    
    #! CHECK OVER EQUATIONS!

    Parameters:
    - omega: Propeller speed (rad/s). Positive assumes clockwise rotation looking forward.
    - Ixx: Propeller polar moment of inertia about the spin axis.
    - rho: Freestream air density.
    - V: Freestream velocity.
    - D: Propeller diameter.
    - C_k: Dictionary of non-dimensional aero stiffness derivatives. Keys are (row_dof, col_dof).
    - C_b: Dictionary of non-dimensional aero damping derivatives. Keys are (row_dof, col_dof).
    
    Returns:
    - tuple: (gyro_dict, aero_damping_dict, aero_stiff_dict) ready for PyNastran DMIG injection.
    """
    
    # 1. Calculate flight condition variables
    q = 0.5 * rho * V**2           # Dynamic pressure
    S = math.pi * (D / 2.0)**2     # Propeller disk reference area
    
    # 2. Gyroscopic Dictionary
    # Angular momentum couples pitch rate (DOF 5) and yaw rate (DOF 6)
    gyro_dict = {
        (5, 6): Ixx * omega,
        (6, 5): -Ixx * omega
    }
    
    # 3. Aerodynamic Stiffness Dictionary [K_A]
    aero_stiff_dict = {}
    for (row, col), coeff in C_k.items():
        if row in [1, 2, 3]:
            # Forces scale with dynamic pressure and area
            scale = q * S
        elif row in [4, 5, 6]:
            # Moments scale with dynamic pressure, area, and diameter
            scale = q * S * D
        else:
            continue
            
        aero_stiff_dict[(row, col)] = coeff * scale
        
    # 4. Aerodynamic Damping Dictionary [B_A]
    aero_damping_dict = {}
    for (row, col), coeff in C_b.items():
        # A. Damping from Translational Rates (DOFs 1, 2, 3)
        if col in [1, 2, 3]:  
            if row in [1, 2, 3]:
                # Force induced by translational rate
                scale = (q * S) / V
            elif row in [4, 5, 6]:
                # Moment induced by translational rate
                scale = (q * S * D) / V
                
        # B. Damping from Angular Rates (DOFs 4, 5, 6)
        elif col in [4, 5, 6]: 
            if row in [1, 2, 3]:
                # Force induced by angular rate
                scale = (q * S * D) / (2.0 * V)
            elif row in [4, 5, 6]:
                # Moment induced by angular rate
                scale = (q * S * D**2) / (2.0 * V)
        else:
            continue
            
        aero_damping_dict[(row, col)] = coeff * scale
        
    return gyro_dict, aero_damping_dict, aero_stiff_dict