#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note that the sed module can be found in the STARDUST git repo in the scripts directory

from sed import sed




# constants
N_grid = 1000
star_mass_max = 500
imf_bins = 50

# --------------------------------------------
# example 1 - medium mass and redshift
# --------------------------------------------

# our 8 parameters
z = 9.0
log_M = 10.
source_age = 10.0

star_mass_min = 100
imf_index = 2.35
f_esc = 0.1

qso_alpha = 1.0
qso_efficiency = 0.1

sed_file = 'sed_example_1_IMF+PL_M%.3f_z%.3f.dat'%(log_M, z)

sed.generate_SED_IMF_PL(haloMass=10**log_M,
                        redshift=z,
                        eLow=10.4, eHigh=1.e4,
                        N=N_grid,  logGrid=True,
                        starMassMin=star_mass_min,
                        starMassMax=star_mass_max,
                        imfBins=imf_bins,
                        imfIndex=imf_index,
                        fEsc=f_esc,
                        alpha=qso_alpha,
                        qsoEfficiency=qso_efficiency,
                        targetSourceAge=source_age,
                        fileName=sed_file)


# --------------------------------------------
# example 2 high mass, low redshift
# --------------------------------------------

# our 8 parameters
z = 6.0
log_M = 13.
source_age = 10.0

star_mass_min = 100
imf_index = 2.35
f_esc = 0.1

qso_alpha = 1.0
qso_efficiency = 0.1

sed_file = 'sed_example_2_IMF+PL_M%.3f_z%.3f.dat'%(log_M, z)

sed.generate_SED_IMF_PL(haloMass=10**log_M,
                        redshift=z,
                        eLow=10.4, eHigh=1.e4,
                        N=N_grid,  logGrid=True,
                        starMassMin=star_mass_min,
                        starMassMax=star_mass_max,
                        imfBins=imf_bins,
                        imfIndex=imf_index,
                        fEsc=f_esc,
                        alpha=qso_alpha,
                        qsoEfficiency=qso_efficiency,
                        targetSourceAge=source_age,
                        fileName=sed_file)


# --------------------------------------------
# example 3 low mass, high redshift
# --------------------------------------------

# our 8 parameters
z = 12.5
log_M = 8.
source_age = 10.0

star_mass_min = 100
imf_index = 2.35
f_esc = 0.1

qso_alpha = 1.0
qso_efficiency = 0.1

sed_file = 'sed_example_3_IMF+PL_M%.3f_z%.3f.dat'%(log_M, z)

sed.generate_SED_IMF_PL(haloMass=10**log_M,
                        redshift=z,
                        eLow=10.4, eHigh=1.e4,
                        N=N_grid,  logGrid=True,
                        starMassMin=star_mass_min,
                        starMassMax=star_mass_max,
                        imfBins=imf_bins,
                        imfIndex=imf_index,
                        fEsc=f_esc,
                        alpha=qso_alpha,
                        qsoEfficiency=qso_efficiency,
                        targetSourceAge=source_age,
                        fileName=sed_file)



# --------------------------------------------
# example 4 QSO only t-evolution
# --------------------------------------------

# our 8 parameters
z = 7.0
log_M = 10.
source_age = 10.0

qso_alpha = 1.0
qso_efficiency = 0.1

sed_file = 'sed_example_4_PL_M%.3f_z%.3f.dat'%(log_M, z)

sed.generate_SED_PL(haloMass=10**log_M,
                    eHigh=1.e4, eLow=10.4,
                    fileName=sed_file,
                    alpha=qso_alpha,
                    N=N_grid,
                    logGrid=True,
                    qsoEfficiency=qso_efficiency,
                    silent=False)
