"""
Functions for 1976 standard atmosphere calculations
"""
import math 

# Copyright (C) <2026>, Streamline Aerospace. All rights reserved.

def air_temp(h_ft): 
    return 288.15 - 1.9812e-3*h_ft


def air_density(h_ft, dT_C=0):
    T_std = air_temp(h_ft)
    rho_std = (0.24179285 - (1.6624675e-6)*h_ft)**4.2558797
    return rho_std/(1 + dT_C/T_std)


def dynamic_visc(h_ft):
    T = air_temp(h_ft) #get temperature 

    mu_ref = 1.716e-5
    T_ref = 273.15
    S = 110.4
    mu_Ns_m2 = mu_ref*(T/T_ref)**1.5*(T_ref+S)/(T+S)

    mu_lbfs_ft2 = mu_Ns_m2*0.0209
    return mu_lbfs_ft2


def sonic_speed(h_ft):

    T_K = air_temp(h_ft)
    a_m_s = math.sqrt(1.4*278.05*T_K)
    a_ft_s = a_m_s*3.28084
    return a_ft_s