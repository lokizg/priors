import sys
from astropy.io import ascii
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import hstack
#import matplotlib.pyplot as plt 
import numpy as np
from scipy import stats
from scipy import optimize
from scipy import interpolate 
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde

import LocusTools as lt
import PlotTools as pt
import param_tool 

import healpy as hp

import time
#import configure


file_params = param_tool.getFileParams()
fitting_params = param_tool.getFittingParams()

print(file_params['OutputFilePath'])
print(file_params['ListFile'])


nside = fitting_params['NSide']

output_dir = file_params['OutputFilePath']
list_file = file_params['ListFile']
f = open(list_file)
healpixfile_lines = f.readlines()
f.close()

healpix_code = []
ra = []
dec = []
num_stars = []
num_maps = []
exit_code = []
run_time = []


i=0
while i < len(healpixfile_lines):

    tri_out_file = healpixfile_lines[i].strip()
    
    ### Temporarily for testing the special three pixels
    #healpix_num = -99
    
    
    ### Temporarily comment out for testing the special three pixels
    healpix_num = int(healpixfile_lines[i].strip().split('/')[8].split('.')[0].split('_')[2])
    
    #### Get RA and Dec of the healpixel
    sky_l, sky_b = hp.pix2ang(nside, healpix_num, lonlat=True, nest=True)
    sky_galactic = SkyCoord(l = sky_l*u.degree, b = sky_b*u.degree, frame='galactic')
    sky_icrs = sky_galactic.transform_to('icrs')
    healpix_ra = sky_icrs.ra.deg
    healpix_dec = sky_icrs.dec.deg
    
    
    if healpix_dec > 30:
        healpix_code.append(healpix_num)
        ra.append(healpix_ra)
        dec.append(healpix_dec)
        num_stars.append(-99)
        num_maps.append(-99)
        exit_code.append('B')
        run_time.append(0)
        
        i+=1
        continue
    ###
    
    
    t_start = time.time()
    trilegal = lt.readTRILEGAL(tri_out_file)
    healpix_stars = len(trilegal) ### Get the number of stars in the healpixel
    
    
    # absolute intrinsic r band magnitude
    trilegal['Mr'] = trilegal['rmag'] - trilegal['Ar'] - trilegal['mu0']
    # 'mu0' is the column name of data module in TRILEGAL file
    # NB TRILEGAL sample is defined by r<27.5
    
    
    rootname = 'healPix_N{}_{}_priors'.format(nside, healpix_num)
    rmagMin = fitting_params['rmag_MIN']
    rmagMax = fitting_params['rmag_MAX']
    rmagBinSZ = fitting_params['r_BinSize']
    rmagNsteps = fitting_params['rmag_Nsteps']
    
    healpix_maps = len(np.linspace(rmagMin, rmagMax, rmagNsteps))

    # NB map parameters are set in dumpPriorMaps()
    lt.dumpPriorMaps(trilegal, rmagMin, rmagMax, rmagNsteps, rootname)
    t_end = time.time()
    
    healpix_code.append(healpix_num)
    ### Temporarily comment out for testing the special three pixels 
    ra.append(healpix_ra)
    dec.append(healpix_dec)
    ###
    
    #ra.append(-99)
    #dec.append(-99)
    
    
    num_stars.append(healpix_stars)
    num_maps.append(healpix_maps)
    exit_code.append('G')
    run_time.append((t_end - t_start))
    
    i+=1


    


data = Table()
data["HealpixNum"] = np.array(healpix_code, dtype = "int32")
data["RA"] = np.array(ra, dtype = "float32")
data["Dec"] = np.array(dec, dtype = "float32")
data["RA"].format = '{0:.6f}'
data["Dec"].format = '{0:.6f}'

data["NumStars"] = np.array(num_stars, dtype = "int64")
data["NumMaps"] = np.array(num_maps, dtype = "int32")
data["Exit"] = np.array(exit_code, dtype = "str")
data["RunTime"] = np.array(run_time, dtype = "float32")
data["RunTime"].format = '{0:.6f}'

out_master = output_dir + '/Summary_rMin-{}_rMax-{}_rBinSZ-{}.dat'.format(rmagMin, rmagMax, rmagBinSZ)
#ascii.write_header(comment_line)
ascii.write(data, out_master, overwrite=True)
#ascii.commented_header.write(comment_line, out_master)
    
    
    
    