from __future__ import division, print_function
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import numpy as np
import pdb

Omega_earth = 2*np.pi/u.d

def magnetic_dipole(r_0, D, F, rho_0=11e6*u.g/u.m**3, gamma=0.15):
    M = 4*np.pi*r_0**3*gamma*(rho_0/c.mu0)**0.5*(F*D)**(1/3.)
    return M.to(u.A*u.m**2)
    
def core_radius(R_p_in_R_e, M_p_in_M_e):
    r_0 = R_p_in_R_e*c.R_earth*np.sqrt(1/0.21*np.maximum(1.07 - R_p_in_R_e/M_p_in_M_e**0.27,0))
    return r_0.to(u.m)
    
def convective_heat_flux(D, Omega, Rosby_on_Rosby_e=1.0):
    """Convective heat flux (add comments!)
    
    NB Originally had 0.1/0.09 for Rosby number
    """
    D_earth = 0.65*3.86e6*u.m
    Omega_earth = 2*np.pi/u.d
    F_earth = 2e-13*u.m**2/u.s**3
    F = F_earth * Rosby_on_Rosby_e**2 * (D/D_earth)**(2/3.) * (Omega/Omega_earth)**(7/3.)
    return F.to(u.m**2/u.s**3)
    
def magnetic_dipole_errors(R, R_err, M, M_err, Omega, n_samp=1000, M_earth=None, M_earlyearth=None, M_mars=None):
    Rs = R + R_err*np.random.normal(size=n_samp)
    Ms = M + M_err*np.random.normal(size=n_samp)
    #pdb.set_trace()
    r0s = core_radius(Rs,Ms)
    Ds = 0.65*r0s
    Fs = convective_heat_flux(Ds, Omega)
    M_dipoles = magnetic_dipole(r0s, Ds, Fs)
    if M_earth is None:
        return np.mean(M_dipoles), np.std(M_dipoles), np.percentile(M_dipoles, 90)
    else
        probabilities = [np.mean(M_dipoles > M_earth), np.mean(M_dipoles > M_earlyearth), np.mean(M_dipoles > M_mars)]
        return np.mean(M_dipoles), np.std(M_dipoles), np.percentile(M_dipoles, 90), probabilities
    
if __name__ == "__main__":
    r_0 = core_radius(1.0, 1.0)#Should be 3.86e6*u.m
    M = magnetic_dipole(r_0, 2.39e6*u.m, 2e-13*u.m**2/u.s**3)
    F = convective_heat_flux(0.65*3.86e6*u.m, 2*np.pi/u.d)
    print(M)
    print(F)
    M, M_err, M_90, probabilities = magnetic_dipole_errors(1.288, 0.1344, 2.73, 0.553, 0.1*Omega_earth)