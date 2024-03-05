r"""
Spherical core with Gaussian chain micelle model
"""

import numpy as np
from sasmodels.special import sas_3j1x_x, sas_sinx_x

name = "spherical_micelle"
title = "Spherical core with a Gaussian chain micelle"
description = """ See J. Appl. Cryst. (2000). 33, 637±640
      """
category = "shape:sphere"

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["v_core",    "Ang^3",  4000.0, [0.0, np.inf], "", "Volume of the core (single block)"],
              ["v_corona",      "Ang^3",      4000.0, [0.0, np.inf], "", "Volume of the corona (single block)"],
              ["sld_solvent",  "Ang^-2",     1.0, [0.0, np.inf], "sld", "Solvent scattering length density"],
              ["sld_core",      "Ang^-2", 2.0, [0.0, np.inf], "sld", "Core scattering length density"],
              ["sld_corona",    "Ang^-2", 2.0,  [0.0, np.inf], "sld", "Corona scattering length density"],
              ["radius_core",   "Ang",       40.0,  [0.0, np.inf], "volume", "Radius of core ( must be >> rg )"],
              ["rg",    "Ang",       10.0,  [0.0, np.inf], "volume", "Radius of gyration of chains in corona"],
              ["d_penetration", "",           1.0,  [0.0, np.inf], "", "Factor to mimic non-penetration of Gaussian chains"],
              ["x_solv",      "",           0.0,  [0.0, 1.0], "", "Core solvation fraction"],        
             ]


def Iq(q,
        v_core=4000,
        v_corona=4000,
        sld_solvent=1,
        sld_core=2,
        sld_corona=1,
        radius_core=40,
        rg=10,
        d_penetration=1,
        x_solv = 0.0
        ):
    n_aggreg = ((1-x_solv)*(4/3)*np.pi*(radius_core**3))/v_core
    rho_solv = sld_solvent    
    rho_core = sld_core       
    rho_corona = sld_corona    

    beta_core = v_core * (rho_core - rho_solv)
    beta_corona = v_corona * (rho_corona - rho_solv)

    # Self-correlation term of the core
    bes_core = sas_3j1x_x(q*radius_core)
    term1 = np.power(n_aggreg*beta_core*bes_core, 2)

    # Self-correlation term of the chains
    qrg2 = np.power(q*rg, 2)
    debye_chain = 2.0*(np.vectorize(np.expm1)(-qrg2)+qrg2)/(qrg2**2) 
    debye_chain[qrg2==0.0] = 1.0
    term2 = n_aggreg * beta_corona * beta_corona * debye_chain

    # Interference cross-term between core and chains
    chain_ampl = -np.vectorize(np.expm1)(-qrg2)/qrg2
    chain_ampl[qrg2==0.0] =  1.0 
    bes_corona = sas_sinx_x(q*(radius_core + (d_penetration * rg)))
    term3 = 2.0 * n_aggreg * n_aggreg * beta_core * beta_corona * bes_core * chain_ampl * bes_corona

    # Interference cross-term between chains
    term4 = n_aggreg * (n_aggreg - 1.0)* np.power(beta_corona * chain_ampl * bes_corona, 2)

    # I(q)_micelle : Sum of 4 terms computed above
    i_micelle = term1 + term2 + term3 + term4

    return i_micelle

Iq.vectorized = True  # Iq accepts an array of q values

def random():
    """Return a random parameter set for the model."""
    radius_core = 10**np.random.uniform(1, 3)
    rg = radius_core * 10**np.random.uniform(-2, -0.3)
    d_penetration = np.random.randn()*0.05 + 1
    v_core = np.random.uniform(3, 5)*(10**3)
    v_corona = np.random.uniform(3, 5)*(10**3)
    x_solv = np.random.uniform(0.0, 1.0)
    sld_solvent = np.random.uniform(0.0, 2.0),
    sld_core = np.random.uniform(0.0, 2.0),
    sld_corona = np.random.uniform(0.0, 2.0),

    pars = dict(
        background=0,
        scale=1.0,
        v_core=v_core,
        v_corona=v_corona,
        sld_solvent = sld_solvent,
        sld_core = sld_core,
        sld_corona = sld_corona,
        radius_core=radius_core,
        rg=rg,
        d_penetration=d_penetration,
        x_solv = x_solv
    )
    return pars