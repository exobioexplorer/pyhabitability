from __future__ import division, print_function
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import numpy as np
import pdb
from astropy.table import Table

#Define the rotation rate of earth
Omega_earth = 2*np.pi/u.d

#Define critical magnetic moments
M_earth = 5.71e22*u.A/u.m**2
M_early_earth = 0.5*M_earth
M_mars = 0.1*M_earth

def magnetic_dipole(r_0, D, F, rho_0=11e6*u.g/u.m**3, gamma=0.15):
    """Find the magnetic dipole moment of a rocky planet
    
    Parameters
    ----------
    r_0: astropy length
        Total core radius
    D: astropy length
        Liquid core thickness
    F: astropy quantity
        Convective buoyancy flux
    
    """
    M = 4*np.pi*r_0**3*gamma*(rho_0/c.mu0)**0.5*(F*D)**(1/3.)
    return M.to(u.A*u.m**2)
    
def core_radius(R_p_in_R_e, M_p_in_M_e):
    """Find the core radius
    
    Parameters
    ----------
    R_p_in_R_e: float
        planetary radius in earth radii
    M_p_in_M_e: float
        planetary radius in earth radii
    """
    r_0 = R_p_in_R_e*c.R_earth*np.sqrt(1/0.21*np.maximum(1.07 - R_p_in_R_e/M_p_in_M_e**0.27,0))
    return r_0.to(u.m)
    
def convective_heat_flux(D, Omega, Rosby_on_Rosby_e=1.333):
    """Convective heat flux (add comments!)
    
    NB Originally had 0.1/0.09 for Rosby number
    """
    D_earth = 0.65*3.86e6*u.m
    Omega_earth = 2*np.pi/u.d
    F_earth = 2e-13*u.m**2/u.s**3
    F = F_earth * Rosby_on_Rosby_e**2 * (D/D_earth)**(2/3.) * (Omega)**(7/3.)
    return F.to(u.m**2/u.s**3)
    
def magnetic_dipole_errors(R, R_err, M, M_err, Omega, Omega_err, n_samp=1000, M_earth=None, \
        M_earlyearth=None, M_mars=None, return_mc=False):
    """Compute the histogram of matnetic dipoles for one planet
    
    Parameters
    ----------
    return_mc: bool
        Do we return the full Monte-carlo output?
    """
    Rs = np.maximum(R + R_err*np.random.normal(size=n_samp),1e-6)
    Ms = np.maximum(M + M_err*np.random.normal(size=n_samp),1e-6)
    Omegas = np.maximum(Omega + Omega_err*np.random.normal(size=n_samp),0)
    #pdb.set_trace()
    r0s = core_radius(Rs,Ms)
    Ds = 0.65*r0s
    Fs = convective_heat_flux(Ds, Omegas)
    M_dipoles = magnetic_dipole(r0s, Ds, Fs)
    #Slightly messy code - return different things depending on our inputs.
    if return_mc:
        return M_dipoles
    if M_earth is None:
        return np.mean(M_dipoles), np.std(M_dipoles), np.percentile(M_dipoles, 90)
    else:
        probabilities = [np.mean(M_dipoles > M_earth), np.mean(M_dipoles > M_earlyearth), np.mean(M_dipoles > M_mars)]
        return np.mean(M_dipoles), np.std(M_dipoles), np.percentile(M_dipoles, 90), probabilities
    
if __name__ == "__main__":
    #First, a test calculation.
    r_0 = core_radius(1.0, 1.0)#Should be 3.86e6*u.m
    M = magnetic_dipole(r_0, 2.39e6*u.m, 2e-13*u.m**2/u.s**3)
    F = convective_heat_flux(0.65*3.86e6*u.m, 1.0 ) #Was 2*np.pi/u.d in physical units for earth
    print(M)
    print(F)
    M, M_err, M_90 = magnetic_dipole_errors(1.288, 0.1344, 2.73, 0.553, 0.1, 0.05)
    
    #Import our planetary data
    tab = Table.read('plan-mag-input.csv', format='ascii.csv')
    
    n_planets = len(tab)
    n_samp = 1000
    M_mc_all = np.zeros( (n_planets,n_samp) )
    for ix, (rad, rad_err, mass, mass_err, Omega, Omega_err) in enumerate(zip(tab['radius'], tab['radius err'], tab['mass'], tab['mass err'], tab['omega'], tab['omega err'])):
        M_mc_all[ix]  = magnetic_dipole_errors(rad, rad_err, mass, mass_err, Omega, Omega_err, n_samp=n_samp, return_mc=True)
        
    #Plot a histogram of mean M over all stars
    mean_M_in_M_earth = np.mean(M_mc_all/M_earth.si.value, axis=1)
    plt.hist(np.log10(np.maximum(mean_M_in_M_earth,1e-3)),20)
    plt.xlabel('log10(M/M_earth)')
    
    