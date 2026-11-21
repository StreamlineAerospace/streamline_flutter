import math
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1, y0, y1


def get_propeller_integrals(
    N_b: int,
    R: float,
    c_r: float,
    mu: float,
    c_cr_func,
    a_0: float = 2 * math.pi,
    a_M: float = 4 * math.pi,
    M: float = 0.0,
    eta_0: float = 0.0,
    unsteady: bool = False,
    A: float | None = None,
) -> dict:
    """Evaluates blade integrals. Rodden & Rose equations (22) thru (29)."""

    # Equation (29): Blade aspect ratio if not user-specified
    if A is None:
        D = 2.0 * R
        A = (D / c_r) * (1.0 - eta_0) ** 2 / quad(c_cr_func, eta_0, 1.0)[0]

    # Steady defaults: C(0) = F(0) + i*G(0) = 1 + 0i
    F = lambda eta: 1.0
    G = lambda eta: 0.0

    if unsteady:
        # Equation (28): Reduced frequency k = c / (2*R*sqrt(mu^2 + eta^2))
        k_func = lambda eta: (c_cr_func(eta) * c_r) / (2.0 * R * np.sqrt(mu**2 + eta**2))

        def F(eta):
            k = k_func(eta)
            denom_k = (j1(k) + y0(k)) ** 2 + (y1(k) - j0(k)) ** 2
            return (j1(k) * (j1(k) + y0(k)) + y1(k) * (y1(k) - j0(k))) / denom_k

        def G(eta):
            k = k_func(eta)
            denom_k = (j1(k) + y0(k)) ** 2 + (y1(k) - j0(k)) ** 2
            return -(y1(k) * y0(k) + j1(k) * j0(k)) / denom_k

    # Mach number cutoff (page 8)
    mach_limit = 1.0 - (a_0 / a_M) ** 2

    def denom(eta):
        m_term = min(M**2 * (1.0 + (eta**2) / (mu**2)), mach_limit) if M > 0 else 0.0
        return np.sqrt(mu**2 + eta**2) * (2.0 + A * np.sqrt(1.0 - m_term))

    scale = (N_b / 4.0) * (a_0 / (2.0 * math.pi)) * A

    I_1 = scale * mu**2 * quad(lambda eta: c_cr_func(eta) * F(eta) / denom(eta), eta_0, 1.0)[0]
    J_1 = scale * mu**2 * quad(lambda eta: c_cr_func(eta) * G(eta) / denom(eta), eta_0, 1.0)[0]
    I_2 = scale * mu * quad(lambda eta: eta**2 * c_cr_func(eta) * F(eta) / denom(eta), eta_0, 1.0)[0]
    J_2 = scale * mu * quad(lambda eta: eta**2 * c_cr_func(eta) * G(eta) / denom(eta), eta_0, 1.0)[0]
    I_3 = scale * quad(lambda eta: eta**4 * c_cr_func(eta) * F(eta) / denom(eta), eta_0, 1.0)[0]
    J_3 = scale * quad(lambda eta: eta**4 * c_cr_func(eta) * G(eta) / denom(eta), eta_0, 1.0)[0]

    return {"I_1": I_1, "J_1": J_1, "I_2": I_2, "J_2": J_2, "I_3": I_3, "J_3": J_3}


def get_propeller_coeffs_bet(
    N_b: int,
    Omega_rad_s: float,
    V_inf: float,
    R: float,
    c_r: float,
    c_cr_func,
    a_0: float = 2 * math.pi,
    a_M: float = 4 * math.pi,
    M: float = 0.0,
    eta_0: float = 0.0,
    unsteady: bool = False,
    A: float | None = None,
) -> dict:
    """Generates propeller derivatives using blade element theory (Eqs. 6-21)."""
    mu = V_inf / (Omega_rad_s * R)
    prop = get_propeller_integrals(
        N_b=N_b, R=R, c_r=c_r, mu=mu, c_cr_func=c_cr_func,
        a_0=a_0, a_M=a_M, M=M, eta_0=eta_0, unsteady=unsteady, A=A
    )

    factor_force = 4.0 * Omega_rad_s * c_r / V_inf
    factor_moment = 2.0 * Omega_rad_s * c_r / V_inf

    C_z_theta = -factor_force * prop["I_1"]
    C_y_psi = -C_z_theta

    C_y_theta = -factor_force * prop["J_1"]
    C_z_psi = C_y_theta

    C_z_q = factor_force * prop["J_2"]
    C_y_r = -C_z_q

    C_y_q = -factor_force * prop["I_2"]
    C_z_r = C_y_q

    C_m_theta = -factor_moment * prop["J_2"]
    C_n_psi = C_m_theta

    C_n_theta = -factor_moment * prop["I_2"]
    C_m_psi = -C_n_theta

    C_m_q = -factor_moment * prop["I_3"]
    C_n_r = C_m_q

    C_n_q = -factor_moment * prop["J_3"]
    C_m_r = -C_n_q

    return {
        "C_z_theta": C_z_theta, "C_y_psi": C_y_psi,
        "C_y_theta": C_y_theta, "C_z_psi": C_z_psi,
        "C_z_q": C_z_q,         "C_y_r": C_y_r,
        "C_y_q": C_y_q,         "C_z_r": C_z_r,
        "C_m_theta": C_m_theta, "C_n_psi": C_n_psi,
        "C_n_theta": C_n_theta, "C_m_psi": C_m_psi,
        "C_m_q": C_m_q,         "C_n_r": C_n_r,
        "C_n_q": C_n_q,         "C_m_r": C_m_r,
    }


def generate_whirl_flutter_dictionaries(omega, Ixx, rho, V, D, C_k, C_b):
    """
    Generates dimensional NASTRAN DMIG dictionaries for gyroscopic, aerodynamic 
    stiffness, and aerodynamic damping matrices.
    
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


if __name__ == "__main__":
    #Validation test case NASA TN D-1807
    import matplotlib.pyplot as plt 
    plt.style.use("bmh")


    beta_75R_deg_sweep = [25, 35, 46, 52, 58]
    V_ft_s_sweep = [63.28, 91.13, 134.16, 168.08, 212.64]
    a_sound_ft_s = 1116.0
    eta_sweep = [0.17, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.27, 1.0]
    c_cr_sweep = [0.25, 0.33, 0.40, 0.50, 0.70, 0.80, 0.90, 1.0, 1.0]
    eta_0 = 0.17
    c_r_ft = 4.3752 / 12.0
    R_ft = 10.1256 / 12.0
    Omega_rad_sec = 1800.0 * 2.0 * math.pi / 60.0
    N_b = 4

    c_cr_func = lambda eta: np.interp(eta, eta_sweep, c_cr_sweep)

    C_z_theta_sweep, C_m_q_sweep = [],[]

    for beta, V_inf in zip(beta_75R_deg_sweep, V_ft_s_sweep):
        res = get_propeller_coeffs_bet(
            N_b=N_b,
            Omega_rad_s=Omega_rad_sec,
            V_inf=V_inf,
            R=R_ft,
            c_r=c_r_ft,
            c_cr_func=c_cr_func,
            M=V_inf / a_sound_ft_s,
            eta_0=eta_0,
            unsteady=True,
            A=3.47,  # Table 3 / Fig. 3 user-supplied aspect ratio
        )
        C_z_theta_sweep.append(res["C_z_theta"])
        C_m_q_sweep.append(res["C_m_q"])
        print(
            f"beta={beta:2d}: Cz_theta={res['C_z_theta']:7.4f}, "
            f"Cm_theta={res['C_m_theta']:6.4f}, Cm_q={res['C_m_q']:7.4f}, "
            f"Cz_psi={res['C_z_psi']:6.4f}, Cm_psi={res['C_m_psi']:6.4f}"
        )

    fig,ax = plt.subplots(1,1)
    ax.plot(beta_75R_deg_sweep, C_z_theta_sweep, label="C_z_theta")
    ax.plot(beta_75R_deg_sweep, C_m_q_sweep, label="C_m_q")
    ax.set_xlabel("Blade Angle @ 0.75R")
    ax.set_ylabel("Aerodynamic Derivative")
    ax.legend(title="Coefficient")
    plt.show()