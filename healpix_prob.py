from astropy.io import ascii
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats
from sklearn.mixture import GaussianMixture

import healpy as hp
from astropy import units as u
from astropy.coordinates import SkyCoord

from os.path import exists


class HealSky:
    def __init__(self, n_side, healpix_num, r_binsize):
        self.n_side = n_side
        self.healpix_num = healpix_num
        self.r_binsize = r_binsize
        
        
    def get_data_file(self):
        sky_l, sky_b = hp.pix2ang(self.n_side, self.healpix_num, lonlat=True, nest=True)
        
        sky_galactic = SkyCoord(l = sky_l*u.degree, b = sky_b*u.degree, frame='galactic')
        sky_icrs = sky_galactic.transform_to('icrs')
        
        if (sky_icrs.dec.degree > -90) & (sky_icrs.dec.degree < -45):
            return -90, -45
        
        elif (sky_icrs.dec.degree > -45) & (sky_icrs.dec.degree < -30):
            return -45, -30
        
        elif (sky_icrs.dec.degree > -30) & (sky_icrs.dec.degree < -15):
            return -30, -15
        
        elif (sky_icrs.dec.degree > -15) & (sky_icrs.dec.degree < 0):
            return -15, 0
        
        elif (sky_icrs.dec.degree > 0) & (sky_icrs.dec.degree < 45):
            return 0, 45
        
        
    def read_data(self):
        
        dec_small, dec_large = self.get_data_file()
        
        file = "/data/epyc/projects/trilegal/lsstsim/strip_{}_{}/OutDir/triout_{}_{}.dat".format(dec_small, dec_large, self.n_side, self.healpix_num)
        
        if exists(file) == False:
            return f"Please check Healpix resolution, n_side, and Healpix number, healpix_num."
        
        data = ascii.read(file)

        self.dm = np.array(data["mu0"], dtype = "float32")
        self.g_mag = np.array(data["gmag"], dtype = "float32")
        self.r_mag = np.array(data["rmag"], dtype = "float32")
        self.i_mag = np.array(data["imag"], dtype = "float32")
        self.u_mag = np.array(data["umag"], dtype = "float32")
        self.z_mag = np.array(data["zmag"], dtype = "float32")

        self.feh = np.array(data["M_H"], dtype = "float32")
        self.av = np.array(data["Av"], dtype = "float32")
        
        self.ar = 2.75 * self.av / 3.10 
        self.mr = self.r_mag - self.ar - self.dm - 5
        

class Stars(HealSky):            
    def select_star(self, test_r):
        bin_number = (27.5-13.5) / self.r_binsize
        r_bin = self.r_binsize * np.arange(bin_number + 1) + 13.5

        loc_bin = np.where(r_bin < test_r)[0][-1]
        
        self.mr_select = self.mr[np.where((self.r_mag > r_bin[loc_bin]) & (self.r_mag < r_bin[loc_bin+1]))]
        self.ar_select = self.ar[np.where((self.r_mag > r_bin[loc_bin]) & (self.r_mag < r_bin[loc_bin+1]))]
        self.feh_select = self.feh[np.where((self.r_mag > r_bin[loc_bin]) & (self.r_mag < r_bin[loc_bin+1]))]
        
        
    def mr_fitting(self):
        mr_reshape = self.mr_select.reshape(-1, 1)
        mr_result = GaussianMixture(n_components=3, covariance_type='spherical').fit(mr_reshape)
        
        return mr_result
    
    
    def feh_fitting(self):
        feh_reshape = self.feh_select.reshape(-1, 1)
        feh_result = GaussianMixture(n_components=3, covariance_type='spherical').fit(feh_reshape)
        
        return feh_result
    
    
    def ar_fitting(self):
        ar_reshape = self.ar_select.reshape(-1, 1)
        ar_result = GaussianMixture(n_components=3, covariance_type='spherical').fit(ar_reshape)
        
        return ar_result
    
    
    def mr_prob(self, mr_result, mr_sample_bins):
        mr_logprob = mr_result.score_samples(mr_sample_bins.reshape(-1,1))
        mr_pdf = np.exp(mr_logprob)
        
        return mr_pdf
        
    
    def plot_mr(self, mr_pdf, mr_sample_bins, filepath):
        mr_plt_bins = 0.5*np.arange(25) - 6
        
        plt.hist(self.mr_select, bins = mr_plt_bins, density=True, histtype='stepfilled', alpha=0.4)
        plt.plot(mr_sample_bins, mr_pdf, '-k', label="Best-fit Mixture")
    
        plt.xlabel("$M_\mathrm{r}$")
        
        plt.savefig(filepath, bbox_inches="tight")
        
        
    def feh_prob(self, feh_result, feh_sample_bins):
        feh_logprob = feh_result.score_samples(feh_sample_bins.reshape(-1,1))
        feh_pdf = np.exp(feh_logprob)
        
        return feh_pdf
        
    
    def plot_feh(self, feh_pdf, feh_sample_bins, filepath):
        feh_plt_bins = 0.25*np.arange(16) - 3
        
        plt.hist(self.feh_select, bins = feh_plt_bins, density=True, histtype='stepfilled', alpha=0.4)
        plt.plot(feh_sample_bins, feh_pdf, '-k', label="Best-fit Mixture")
    
        plt.xlabel("FeH")
        
        plt.savefig(filepath, bbox_inches="tight")
        
        
    def ar_prob(self, ar_result, ar_sample_bins):
        ar_logprob = ar_result.score_samples(ar_sample_bins.reshape(-1,1))
        ar_pdf = np.exp(ar_logprob)
        
        return ar_pdf
    
    
    def plot_ar(self, ar_pdf, ar_sample_bins, filepath):
        ar_plt_bins = 0.25*np.arange(21)
        
        plt.hist(self.ar_select, bins = ar_plt_bins, density=True, histtype='stepfilled', alpha=0.4)
        plt.plot(ar_sample_bins, ar_pdf, '-k', label="Best-fit Mixture")
    
        plt.xlabel("$A_\mathrm{r}$")
        
        plt.savefig(filepath, bbox_inches="tight")
    
        
        
        
        
        
    