

column_HII = 3      # 4. column in the profile file is the hydrogen II ionization fraction
column_HeII = 5     # 6. column n the profile file is the helium II ionization fraction
column_HeIII = 6    # 6. column n the profile file is the helium III ionization fraction
column_T = 7        # 8. column in the profile file is the is kinetic temp

EXPECTED_PROFILE_LEN = 15000


import numpy as np
import sys, os
import scipy.ndimage as ndimage


def down_sample_profile(a, reducedNumber):

    # go from len profileGridPoints to profileGridPointsReduced

    skip = int(len(a)/reducedNumber)

    return a[::skip]


def convert_profile_to_npy(profile_path, prefix='', n_grid=1500):


    tmp_profile_HII = np.genfromtxt(profile_path, usecols=(column_HII))
    tmp_profile_HeII = np.genfromtxt(profile_path, usecols=(column_HeII))
    tmp_profile_HeIII = np.genfromtxt(profile_path, usecols=(column_HeIII))
    tmp_profile_T = np.genfromtxt(profile_path, usecols=(column_T))

    if n_grid is not None:
        tmp_profile_HII = down_sample_profile(tmp_profile_HII, n_grid)
        tmp_profile_HeII = down_sample_profile(tmp_profile_HeII, n_grid)
        tmp_profile_HeIII = down_sample_profile(tmp_profile_HeIII, n_grid)
        tmp_profile_T = down_sample_profile(tmp_profile_T, n_grid)


    # smoothing
    tmp_profile_HII = ndimage.gaussian_filter(tmp_profile_HII, sigma=10)
    tmp_profile_HeII = ndimage.gaussian_filter(tmp_profile_HeII, sigma=10)
    tmp_profile_HeIII = ndimage.gaussian_filter(tmp_profile_HeIII, sigma=10)
    tmp_profile_T = ndimage.gaussian_filter(tmp_profile_T, sigma=10)


    if prefix != '':
        prefix = prefix+'_'
        
    np.save('%sprofile_HII.npy'%prefix, tmp_profile_HII)
    np.save('%sprofile_HeII.npy'%prefix, tmp_profile_HeII)
    np.save('%sprofile_HeIII.npy'%prefix, tmp_profile_HeIII)
    np.save('%sprofile_T.npy'%prefix, tmp_profile_T)


if __name__ == '__main__':



    #p1 =  './run_1/run_1_profile_M10.000_z9.000_t10.000.dat'    
    #convert_profile_to_npy(profile_path=p1, prefix="run_1")
    
    
    #p2 =  './run_2/run_2_profile_M13.000_z6.000_t10.000.dat'    
    #convert_profile_to_npy(profile_path=p2, prefix="run_2")   
   
    
    #p3 =  './run_3/run_3_profile_M8.000_z12.500_t10.000.dat'
    #convert_profile_to_npy(profile_path=p3, prefix="run_3")


    # run 4
    #p4_1 ='run_4/run_4_profile_M10.000_z7.000_t4.000.dat'
    #p4_2 ='run_4/run_4_profile_M10.000_z7.000_t8.000.dat'
    #p4_3 ='run_4/run_4_profile_M10.000_z7.000_t12.000.dat'
    #p4_4 ='run_4/run_4_profile_M10.000_z7.000_t16.000.dat'
    #p4_5 ='run_4/run_4_profile_M10.000_z7.000_t19.990.dat'
    
    #convert_profile_to_npy(profile_path=p4_1, prefix="run_4_t4")
    #convert_profile_to_npy(profile_path=p4_2, prefix="run_4_t8")
    #convert_profile_to_npy(profile_path=p4_3, prefix="run_4_t12")
    #convert_profile_to_npy(profile_path=p4_4, prefix="run_4_t16")
    #convert_profile_to_npy(profile_path=p4_5, prefix="run_4_t20")
