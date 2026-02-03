import numpy as np
import matplotlib.pyplot as plt

def calculate_pdot_vs_rin_with_mu(P=0.71, M=1.4, I=1e45, 
                                  Mdot=5.56e18, xi=1.0, gamma=1.0,
                                  rin_over_rc_range=(0.01, 1.1), n_points=500):
    """
    Calculate Pdot as a function of Rin/Rc, with corresponding μ values.
    
    Parameters:
    -----------
    P : float
        Pulse period in seconds (default: 0.71 s for SMC X-1)
    M : float
        Neutron star mass in solar masses (default: 1.4 M_sun)
    I : float
        Moment of inertia in g cm^2 (default: 1e45)
    Mdot : float
        Accretion rate in g/s (default: 5.56e18, from Lx ~ 5e38 erg/s)
    xi : float
        Parameter for material torque (default: 1.0)
    gamma : float
        Parameter for magnetic torque (default: 1.0)
    rin_over_rc_range : tuple
        Range of Rin/Rc values to explore (default: 0.1 to 10)
    n_points : int
        Number of points in the array (default: 500)
    """
    
    # Constants
    G = 6.67430e-8  # cm^3 g^-1 s^-2
    M_sun = 1.989e33  # g
    M_g = M * M_sun  # Convert to grams
    
    # Calculate corotation radius
    R_c = (G * M_g * P**2 / (4 * np.pi**2))**(1/3)
    print(f"Corotation radius R_c = {R_c/1e5:.2f} km")
    print(f"Accretion rate Mdot = {Mdot:.2e} g/s")
    print(f"X-ray luminosity Lx ~ {Mdot * 0.1 * 9e20 / 1e38:.2f} × 10^38 erg/s (assuming 10% efficiency)")
    
    # Create range of Rin/Rc values
    rin_over_rc_array = np.logspace(np.log10(rin_over_rc_range[0]), 
                                    np.log10(rin_over_rc_range[1]), 
                                    n_points)
    
    # Calculate actual Rin values
    Rin_array = rin_over_rc_array * R_c
    
    # Calculate fastness parameter for each Rin
    omega_array = (Rin_array / R_c)**(1.5)
    
    # Calculate required μ for each Rin to be the magnetospheric radius
    # From: R_m = (μ^4 / (2GMṀ^2))^(1/7) => μ = (2GMṀ^2 R_m^7)^(1/4)
    mu_array = (2 * G * M_g * Mdot**2 * Rin_array**7)**(0.25)
    
    # Initialize arrays for torque and Pdot
    N_array = np.zeros_like(Rin_array)
    Pdot_array = np.zeros_like(Rin_array)
    
    # Calculate torque for each Rin
    for i, (Rin, omega) in enumerate(zip(Rin_array, omega_array)):
        # Prefactor
        prefactor = Mdot * np.sqrt(G * M_g * Rin)
        
        if omega <= 1:
            # Accretion regime
            magnetic_term = (np.sqrt(2) * gamma / 3) * (1 - 2*omega + (2/3)*omega**2)
        else:
            # Propeller regime (soft)
            magnetic_term = (np.sqrt(2) * gamma / 3) * (2/(3*omega) - 1)
        
        material_term = xi * (1 - omega)
        N = prefactor * (material_term + magnetic_term)
        N_array[i] = N
        
        # Calculate Pdot from torque
        # From: -2πI Pdot / P^2 = N  =>  Pdot = - (P^2 * N) / (2πI)
        Pdot_array[i] = - (P**2 * N) / (2 * np.pi * I)
# Function returns:
# Rin_array: Array of inner disk radii in cm, corresponding to R_in values that could be the magnetospheric radius
#            These values span a range from ~0.1R_c to ~10R_c
# rin_over_rc_array: Array of R_in normalized to the corotation radius (R_in/R_c)
#                    This dimensionless parameter determines the torque regime
# omega_array: Array of fastness parameters ω = (R_in/R_c)^(3/2)
#              When ω < 1: Accretion regime (spin-up)
#              When ω > 1: Propeller regime (spin-down)
#              Critical value ω_c ≈ 0.884 where torque changes sign
# mu_array: Array of magnetic moments μ (G cm³) REQUIRED for each R_in to be the actual magnetospheric radius R_m
#           Calculated from R_m formula: μ = (2GMṀ²R_m^7)^(1/4)
#           Shows what magnetic field strength would naturally produce each R_in for given Ṁ
# N_array: Array of total torques N (dyn·cm) on the neutron star
#          Calculated from Equation 11 of Dai & Li (2006): N = N_0 + N_mag
#          Positive N → spin-up, Negative N → spin-down
# Pdot_array: Array of spin period derivatives Ṗ (s/s)
#             Calculated from: Ṗ = -(P²N)/(2πI)
#             Negative Ṗ → spin-up (period decreasing)
#             Positive Ṗ → spin-down (period increasing)
# R_c: Corotation radius in cm
#      R_c = (GMP²/4π²)^(1/3), where Keplerian period equals spin period
#      Critical boundary between accretion and propeller regimes
#
# Key insight: For a FIXED accretion rate Ṁ, the system will settle at R_m determined by μ
#              Different μ values correspond to different positions on the R_in/R_c axis
#              This shows why μ matters: it determines where the system operates on the Ṗ curve
    
    return Rin_array, rin_over_rc_array, omega_array, mu_array, N_array, Pdot_array, R_c

rin1, rin2rc1, omega1, mu1, n1, pdot1, rc1=calculate_pdot_vs_rin_with_mu(P=0.71, M=1.4, I=1e45, Mdot=5.56e18, xi=1.0, gamma=1.0,rin_over_rc_range=(0.01, 1.1), n_points=500)


fig, axes = plt.subplots(2, 1, sharex=False, figsize=(6, 8))

# First panel: y1 vs x1
axes[0].plot(mu1, rin2rc1)
axes[0].set_xlabel(r"$\mu\,(\mathrm{G\,cm^{-3}})$")
axes[0].set_ylabel(r"$R_{in}/R_{cor}$")
axes[0].set_yscale("log")

# Second panel: y2 vs x2
axes[1].plot(mu1, pdot1)
axes[1].set_xlabel(r"$\mu\,(\mathrm{G\,cm^{-3}})$")
axes[1].set_ylabel(r'$\dot{P}\,(s\,s^{-1})$')
#axes[1].set_yscale("log")
plt.tight_layout()
plt.show()

