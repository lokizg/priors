import configure
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii

####### DC: You should be able to use the priors without this function
#def getFileParams():
#    file_params = configure.getFileConfig()
#    
#    return file_params
#######

def getFittingParams():
    fitting_params = configure.getConfigParam()
    
    return fitting_params


def printMapBin(rmagMin, rmagMax, rmagBinSZ, rmagNsteps):
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)

    i=0
    for rind, r in enumerate(rGrid):
        rMin = r - rmagBinSZ
        rMax = r + rmagBinSZ
    
        print('r=', rMin, 'to', rMax, ', Map index:', i)
        i+=1


def getHealpixNum(obj_ra, obj_dec, nside):
    
    #fitting_params = getFittingParams()
    #nside = fitting_params['NSide']
    
    obj_icrs = SkyCoord(ra = obj_ra*u.degree, dec = obj_dec*u.degree, frame='icrs')
    obj_galactic = obj_icrs.transform_to('galactic')
    obj_l = obj_galactic.l.degree
    obj_b = obj_galactic.b.degree
    
    healpix_num = hp.ang2pix(nside, obj_l, obj_b, lonlat=True, nest=True)
    
    return healpix_num


def getPrior(obj_ra, obj_dec, nside, rmagMin, rmagMax, rmagBinSZ, rmagNsteps):
    
    main_dir = './Priors/NSide_{}'.format(nside)
    
    if ((obj_dec > 0) and (obj_dec <= 45)):
        
        prior_dir = main_dir + '/strip_0_45_n{}'.format(nside)
        
    elif ((obj_dec > -15) and (obj_dec <= 0)):
        
        prior_dir = main_dir + '/strip_-15_0_n{}'.format(nside)
        
    elif ((obj_dec > -30) and (obj_dec <= -15)):
        
        prior_dir = main_dir + '/strip_-30_-15_n{}'.format(nside)
        
    elif ((obj_dec > -45) and (obj_dec <= -30)):
        
        prior_dir = main_dir + '/strip_-45_-30_n{}'.format(nside)
        
    elif ((obj_dec > -90) and (obj_dec <= -45)):
        
        prior_dir = main_dir + '/strip_-90_-45_n{}'.format(nside)
        
    else:
        
        print('Only -90 < Dec< 45 is available. Please check your input Dec.')
        return 0

    #print(prior_dir)
    summary_file = prior_dir + '/Summary_rMin-{}_rMax-{}_rBinSZ-{}.dat'.format(rmagMin, rmagMax, rmagBinSZ)
    summary = ascii.read(summary_file)
    all_healpixels = summary['HealpixNum']
    #print(all_healpixels)
    
    healpix_num = getHealpixNum(obj_ra, obj_dec, nside)
    #print(healpix_num)
    
    if healpix_num in all_healpixels:
        ind = np.where(all_healpixels == healpix_num)[0][0]
    
        num_maps = summary['NumMaps'][ind]
        print('Number of prior maps:', num_maps)
        
        printMapBin(rmagMin, rmagMax, rmagBinSZ, rmagNsteps)
        
        rootname = prior_dir + '/healPix_N{}_{}_priors'.format(nside, healpix_num)
        
        return rootname
    
    else:
        print('Prior map is not available for this sky area.')
        return 0
    
    
def getMap(mapInd, rootname):
    
    prior_file = rootname + '-{}.npz'.format(mapInd)
    priors = np.load(prior_file)
        
    return priors
        
        
        
        
        
    
