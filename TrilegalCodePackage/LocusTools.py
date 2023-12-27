import numpy as np
from astropy.table import Table
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
import PlotTools as pt

# modify by DC
from astropy.io import ascii
import param_tool
#import configure

# def MSlocus(gi, FeH):
# def RGlocus(gi, FeH): 
# def GClocusGiants(Mr, FeH):
# def CoveyMSlocus(): 
# def readTRILEGAL():
# def readTRILEGALLSST():
# def readTRILEGALLSSTestimates():
# def LSSTsimsLocus(fixForStripe82=True): 
# def BHBlocus(): 
# def photoFeH(ug, gr): 
# def getMr(gi, FeH):
# def extcoeff(): 
# def getColorsFromMrFeH(L, Lvalues):
# def getColorsFromGivenMrFeH(myMr, myFeH, L, gp, colors=""):
# def getLocusChi2colorsSLOW(colorNames, Mcolors, Ocolors, Oerrors):
# def getLocusChi2colors(colorNames, Mcolors, Ocolors, Oerrors):
# def logLfromObsColors(Mr, FeH, ObsColors, ObsColErrors, gridProperties, SDSScolors):
# def getFitResults(i, gridStride, Ntest, colors, locusGrid, ObsColorErr, locusFitMS, locusFitRG, results):
# def extractBestFit(chi2vals, locusGrid):
# def fitPhotoD(i, gridStride, colors, data2fit, locusFitMS, locusFitRG, results):
# def getPhotoDchi2map(i, colors, data2fit, locus):
# def fitMedians(x, y, xMin, xMax, Nbin, verbose=1): 
# def getLSSTm5(data, depth='coadd'):
# def get2Dmap(sample, labels, metadata):
# def dumpPriorMaps(sample, rmagMin, rmagMax, rmagNsteps, fileRootname):
# def readPrior(rmagMin, rmagMax, rmagNsteps,rootname):
# def getMetadataPriors(priorMap=""):
# def getMetadataLikelihood(locusData=""):
# def pnorm(pdf):
# def getStats(x,pdf):
# def showPosterior(iS): 


def getFileParams():
    file_params = param_tool.getFileParams()
    
    return file_params


def getFittingParams():
    fitting_params = param_tool.getFittingParams()
    
    return fitting_params


def MSlocus(gi, FeH):
        ## main sequence SDSS color locus, as a function of [Fe/H], from
        ## Yuan et al (2015, ApJ 799, 134): "Stellar loci I. Metalicity dependence and intrinsic widths" 
        ## https://iopscience.iop.org/article/10.1088/0004-637X/799/2/134/pdf
        ## 
        ## f(x,y) = a0 + a1y+ a2y2 + a3y3 + a4x + a5xy + a6xy2 + a7x2 + a8yx2 + a9x3,
        ##     where x≡g−i, y≡[Fe/H], and f = g-r, r-i, i-z (see below for u-g)
        ## 
        ## for u-g color, there are five more terms (and a different expression) 
        ## f(x,y) =  b0    + b1y    + b2y2    + b3y3  + b4y4 +
        ##           b5x   + b6yx   + b7y2x   + b8y3x +
        ##           b9x2  + b10yx2 + b11y2x2 +
        ##           b12x3 + b13yx3 +
        ##           b14x4
        ## 
        ##  valid for 0.3 < g-i < 1.6 and −2.0 < [Fe/H] < 0.0  
        ## 
        ## populate the arrays of coefficients (Table 1 in Yuan et al) 
        # u−g
        b = {}  # renamed from a here for clarity wrt other three colors below 
        b[0] =  1.5003
        b[1] =  0.0011
        b[2] =  0.1741
        b[3] =  0.0910
        b[4] =  0.0181
        b[5] = -3.2190
        b[6] =  1.1675
        b[7] =  0.0875
        b[8] =  0.0031
        b[9] =  7.9725
        b[10] = -0.8190
        b[11] = -0.0439
        b[12] = -5.2923
        b[13] =  0.1339
        b[14] =  1.1459
        # g-r
        a = {}
        a[1,0] =  0.0596
        a[1,1] =  0.0348
        a[1,2] =  0.0239
        a[1,3] =  0.0044
        a[1,4] =  0.6070
        a[1,5] =  0.0261
        a[1,6] = -0.0044
        a[1,7] =  0.1532
        a[1,8] = -0.0136
        a[1,9] = -0.0613
        # r-i
        a[2,0] = -0.0596
        a[2,1] = -0.0348
        a[2,2] = -0.0239
        a[2,3] = -0.0044
        a[2,4] =  0.3930
        a[2,5] = -0.0261
        a[2,6] =  0.0044
        a[2,7] = -0.1532
        a[2,8] =  0.0136
        a[2,9] =  0.0613        
        # i-z
        a[3,0] = -0.1060
        a[3,1] = -0.0357
        a[3,2] = -0.0123
        a[3,3] = -0.0017
        a[3,4] =  0.2543
        a[3,5] = -0.0010
        a[3,6] = -0.0050
        a[3,7] = -0.0381
        a[3,8] = -0.0071
        a[3,9] =  0.0030

        ## evaluate colors
        SDSScolors = ('ug', 'gr', 'ri', 'iz')
        color = {}
        for c in SDSScolors:
            color[c] = 0*(gi+FeH)
            j = SDSScolors.index(c)
            x = gi
            y = FeH
            if (c=='ug'):
                color[c] += b[0] + b[1]*y + b[2]*y**2 + b[3]*y**3 + b[4]*y**4 
                color[c] += b[5]*x + b[6]*x*y + b[7]*x*y**2 + b[8]*x*y**3 + b[9]*x**2
                color[c] += b[10]*x**2*y + b[11]*x**2*y**2 + b[12]*x**3 + b[13]*x**3*y + b[14]*x**4
            else:
                color[c] += a[j,0] + a[j,1]*y + a[j,2]*y**2 + a[j,3]*y**3 + a[j,4]*x + a[j,5]*x*y
                color[c] += a[j,6]*x*y**2 + a[j,7]*x**2 + a[j,8]*y*x**2 + a[j,9]*x**3
        color['gi'] = gi
        color['FeH'] = 0*gi + FeH       
        return color


def RGlocus(gi, FeH): 
        ## red giant SDSS color locus, as a function of [Fe/H], from
        ## Zhang et al (2021, RAA, 21, 12, 319): "Stellar loci IV. red giant stars" 
        ## https://iopscience.iop.org/article/10.1088/1674-4527/21/12/319/pdf
        ## 
        ## f(x,y) = a0 + a1y+ a2y2 + a3y3 + a4x + a5xy + a6xy2 + a7x2 + a8yx2 + a9x3,
        ##     where x≡g−i, y≡[Fe/H], and f = ug, gr, ri, iz
        ##  valid for 0.55 < g-i < 1.2 and −2.5 < [Fe/H] < -0.3 
        ## 
        ## populate the array of coefficients (Table 1 in Zhang et al) 
        a = {}
        # u−g
        a[0,0] =  1.4630
        a[0,1] =  0.3132
        a[0,2] = -0.0105
        a[0,3] = -0.0224
        a[0,4] = -1.5851
        a[0,5] = -0.2423
        a[0,6] = -0.0372
        a[0,7] =  2.8655
        a[0,8] =  0.0958
        a[0,9] = -0.7469
        # g-r
        a[1,0] =  0.0957
        a[1,1] =  0.0370
        a[1,2] =  0.0120
        a[1,3] =  0.0020
        a[1,4] =  0.5272
        a[1,5] = -0.0026
        a[1,6] =  0.0019
        a[1,7] =  0.1645
        a[1,8] =  0.0057
        a[1,9] = -0.0488        
        # r-i
        a[2,0] = -0.0957
        a[2,1] = -0.0370
        a[2,2] = -0.0120
        a[2,3] = -0.0020
        a[2,4] =  0.4728
        a[2,5] =  0.0026
        a[2,6] = -0.0019
        a[2,7] = -0.1645
        a[2,8] = -0.0057
        a[2,9] =  0.0488        
        # i-z
        a[3,0] = -0.0551
        a[3,1] = -0.0229
        a[3,2] = -0.0165
        a[3,3] = -0.0033
        a[3,4] =  0.0762
        a[3,5] = -0.0365
        a[3,6] = -0.0006
        a[3,7] =  0.1899
        a[3,8] =  0.0244
        a[3,9] = -0.0805
        ## evaluate colors
        SDSScolors = ('ug', 'gr', 'ri', 'iz')
        color = {}
        for c in SDSScolors:
            color[c] = 0*(gi+FeH)
            j = SDSScolors.index(c)
            x = gi
            y = FeH
            color[c] += a[j,0] + a[j,1]*y + a[j,2]*y**2 + a[j,3]*y**3 + a[j,4]*x + a[j,5]*x*y
            color[c] += a[j,6]*x*y**2 + a[j,7]*x**2 + a[j,8]*y*x**2 + a[j,9]*x**3
        color['gi'] = gi
        color['FeH'] = 0*gi + FeH       
        return color


def GClocusGiants(Mr, FeH):
    ## fits to the RGB branch for globular clusters observed
    ## by SDSS: M2, M3, M5, M13, M15, and M53
    ## the median RGB color is determined for each cluster 
    ## using two Mr bins: 1.5 < Mr < 2 and -0.5 < Mr < 0
    ## for each bin, color vs FeH dependence is fit by a 
    ## linear function
    ## the results are roughly valid for 2.5 < Mr < -1 and
    ## are accurate to a few hundreths of a mag (better 
    ## accuracy for redder colors and brighther bin 
    ## fits as functions of Mr and FeH 
    ##  color = A + B * FeH + C * Mr + D * FeH * Mr
    color = {}
    color['ug'] =  1.9805 +0.2865*FeH -0.2225*Mr -0.0425*FeH*Mr
    color['gr'] =  0.7455 +0.0835*FeH -0.1225*Mr -0.0325*FeH*Mr
    color['ri'] =  0.3100 +0.0280*FeH -0.0500*Mr -0.0100*FeH*Mr
    color['iz'] =  0.0960 +0.0000*FeH -0.0200*Mr +0.0000*FeH*Mr
    color['Mr'] = 0*color['iz'] + Mr
    color['FeH'] = 0*color['iz'] + FeH
    return color

def CoveyMSlocus(): 
        ## STELLAR LOCUS IN THE SDSS-2MASS PHOTOMETRIC SYSTEM
        ## read and return colors from
        ## http://faculty.washington.edu/ivezic/sdss/catalogs/tomoIV/Covey_Berry.txt
        ## for more details see the header (note that 2MASS is on Vega system!)
        ## according to Yuan et al (2015, ApJ 799, 134), this locus approximately corresponds to FeH = -0.5 
        colnames = ['Nbin', 'gi', 'Ns', 'ug', 'Eug', 'gr', 'Egr', 'ri', 'Eri', 'iz', 'Eiz', \
                     'zJ', 'EzJ', 'JH', 'EJH', 'HK', 'EHK']
        coveyMS = Table.read('./Covey_Berry.txt', format='ascii', names=colnames)
        return coveyMS 


def readTRILEGAL(file):
        ## read TRILEGAL simulation (per healpix, as extracted by Dani, ~1-2M stars)
        # Dani: gall galb Gc logAge M_H mu0 Av logg gmag rmag imag umag zmag
        
        ### Comment out by DC
        #colnames = ['glon', 'glat', 'comp', 'logage', 'FeH', 'DM', 'Av', 'logg', 'gmag', 'rmag', 'imag', 'umag', 'zmag']
        ### DC is a much nicer researcher
        
        # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
        # logage with age in years
        # DM = m-M is called true distance modulus in DalTio+(2022), so presumably extinction is not included
        # and thus Mr = rmag - Ar - DM - 5  
        
        ### Comment out by DC
        #trilegal = Table.read(file, format='ascii', names=colnames)
        ### DC is a much nicer person
        
        trilegal = ascii.read(file)
        # dust extinction: Berry+ give Ar = 2.75E(B-V) and DalTio+ used Av=3.10E(B-V)
        trilegal['Ar'] = 2.75 * trilegal['Av'] / 3.10   
        C = extcoeff()
        # correcting colors for extinction effects 
        trilegal['ug'] = trilegal['umag'] - trilegal['gmag'] - (C['u'] - C['g'])*trilegal['Ar']  
        trilegal['gr'] = trilegal['gmag'] - trilegal['rmag'] - (C['g'] - C['r'])*trilegal['Ar']   
        trilegal['ri'] = trilegal['rmag'] - trilegal['imag'] - (C['r'] - C['i'])*trilegal['Ar']   
        trilegal['iz'] = trilegal['imag'] - trilegal['zmag'] - (C['i'] - C['z'])*trilegal['Ar']   
        trilegal['gi'] = trilegal['gr'] + trilegal['ri']  
        return trilegal 


def readTRILEGALLSST():
        
        ## read TRILEGAL simulation augmented with LSST colors, see TRILEGAL_makeTestFile.ipynb 
        colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
        colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr']
        # comp: Galactic component the star belongs to: 1 → thin disk; 2 → thick disk; 3 → halo; 4 → bulge; 5 → Magellanic Clouds.
        # Mr = rmag - Ar - DM - 5
        # rmag0, ug0...iz0: intrinsic values without dust extinction (but include photometric noise)
        # rmag, ug...iz: dust extinction included
        # uErr...zErr: photometric noise
        sims = Table.read('./TRILEGAL_three_pix_triout_V1.txt', format='ascii', names=colnames)
         
        
        sims['gi0'] = sims['gr0'] + sims['ri0']  
        sims['gi'] = sims['gr'] + sims['ri']
        print(np.size(sims), 'read from TRILEGAL_three_pix_triout_V1.txt')
        return sims


def readTRILEGALLSSTestimates():
        ## read FeH and Mr estimates and their uncertainties 
        colnames = ['glon', 'glat', 'FeHEst', 'FeHUnc', 'MrEst', 'MrUnc', 'chi2min']
        simsEst = Table.read('./TRILEGAL_three_pix_triout_BayesEstimates.txt', format='ascii', names=colnames)
        print(np.size(simsEst), 'read from TRILEGAL_three_pix_triout_BayesEstimates.txt')
        return simsEst


def readKarloMLestimates(multiMethod=True):
        file = 'ML_predictions_TRILEGAL_three_pix_triout_V1.txt'
        ## file includes:
        colnames = ['glon', 'glat', 'comp', 'logg', 'FeH', 'Mr', 'DM', 'Ar', 'rmag0', 'ug0', 'gr0', 'ri0', 'iz0']
        colnames = colnames + ['rmag', 'ug', 'gr', 'ri', 'iz', 'uErr', 'gErr', 'rErr', 'iErr', 'zErr', 'gi0', 'gi']
        colnames = colnames + ['test_set', 'naive_single_Mr', 'naive_single_Ar', 'naive_single_FeH']
        colnames = colnames + ['naive_multi_Mr', 'naive_multi_Ar', 'naive_multi_FeH']
        colnames = colnames + ['sampling_single_Mr', 'sampling_single_MrErr', 'sampling_single_Ar']
        colnames = colnames + ['sampling_single_ArErr', 'sampling_single_FeH', 'sampling_single_FeHErr']
        colnames = colnames + ['sampling_multi_Mr', 'sampling_multi_MrErr', 'sampling_multi_Ar']
        colnames = colnames + ['sampling_multi_ArErr', 'sampling_multi_FeH', 'sampling_multi_FeHErr']
        ## read FeH and Mr estimates and fake their uncertainties 
        simsML = Table.read(file, format='ascii', names=colnames)
        simsML['MrUnc'] = -1
        simsML['FeHUnc'] = -1
        # single vs. multi
        if (multiMethod):
           simsML['MrEstML'] = simsML['naive_multi_Mr']
           simsML['FeHEstML'] = simsML['naive_multi_FeH']
        else:
           simsML['MrEstML'] = simsML['naive_single_Mr']
           simsML['FeHEstML'] = simsML['naive_single_FeH']
                
        print(np.size(simsML), 'read from', file)
        return simsML


        
def LSSTsimsLocus(fixForStripe82=True): 
        ## Mr, as function of [Fe/H], along the SDSS/LSST stellar 
        ## for more details see the file header
        ## [Fe/H] ranges from -2.50 to 0.0 in steps of 0.5 dex 
        colnames = ['Mr', 'FeH', 'ug', 'gr', 'ri', 'iz', 'zy']
        LSSTlocus = Table.read('./MSandRGBcolors_v1.3.txt', format='ascii', names=colnames)
        LSSTlocus['gi'] = LSSTlocus['gr'] + LSSTlocus['ri']
        if (fixForStripe82):
            print('Fixing input Mr-FeH-colors grid to agree with the SDSS v4.2 catalog')
            # implement empirical corrections for u-g and i-z colors to make it better agree with the SDSS v4.2 catalog
            # fix u-g: slightly redder for progressively redder stars and fixed for gi>giMax
            ugFix = LSSTlocus['ug']+0.02*(2+LSSTlocus['FeH'])*LSSTlocus['gi']
            giMax = 1.8
            ugMax = 2.53 + 0.13*(1+LSSTlocus['FeH'])
            LSSTlocus['ug'] = np.where(LSSTlocus['gi']>giMax, ugMax , ugFix)
            # fix i-z color: small offsets as functions of r-i and [Fe/H]
            off0 = 0.08
            off2 = -0.09
            off5 = 0.008
            offZ = 0.01
            Z0 = 2.5
            LSSTlocus['iz'] += off0*LSSTlocus['ri']+off2*LSSTlocus['ri']**2+off5*LSSTlocus['ri']**5
            LSSTlocus['iz'] += offZ*(Z0+LSSTlocus['FeH'])
        return LSSTlocus

def BHBlocus(): 
        ## Mr, [Fe/H] and SDSS/LSST colors for BHB stars (Sirko+2004)
        ## [Fe/H] ranges from -2.50 to 0.0 in steps of 0.5 dex 
        colnames = ['Mr', 'FeH', 'ug', 'gr', 'ri', 'iz', 'zy']
        BHBlocus = Table.read('./BHBcolors_v1.0.dat', format='ascii', names=colnames)
        BHBlocus['gi'] = BHBlocus['gr'] + BHBlocus['ri'] 
        return BHBlocus

def photoFeH(ug, gr): 
        x = np.array(ug)
        y = np.array(gr)
        ## photometric metallicity introduced in Ivezic et al. (2008), Tomography II
        ## and revised in Bond et al. (2012), Tomography III (see Appendix A.1) 
        ## valid for SDSS bands and F/G stars defined by 
        ## 0.2 < g−r < 0.6 and −0.25 + 0.5*(u−g) < g−r < 0.05 + 0.5*(u−g)
        A, B, C, D, E, F, G, H, I, J = (-13.13, 14.09, 28.04, -5.51, -5.90, -58.68, 9.14, -20.61, 0.0, 58.20)
        return A + B*x + C*y + D*x*y + E*x**2 + F*y**2 + G*x**2*y + H*x*y**2 + I*x**3 + J*y**3 

def getMr(gi, FeH):
        ## Mr(g-i, FeH) introduced in Ivezic et al. (2008), Tomography II
        MrFit = -5.06 + 14.32*gi -12.97*gi**2 + 6.127*gi**3 -1.267*gi**4 + 0.0967*gi**5
        ## offset for metallicity, valid for -2.5 < FeH < 0.2
        FeHoffset = 4.50 -1.11*FeH -0.18*FeH**2
        return MrFit + FeHoffset

def extcoeff():
        ## coefficients to correct for ISM dust (for S82 from Berry+2012, Table 1)
        ## extcoeff(band) = A_band / A_r 
        extcoeff = {}
        extcoeff['u'] = 1.810
        extcoeff['g'] = 1.400
        extcoeff['r'] = 1.000  # by definition
        extcoeff['i'] = 0.759 
        extcoeff['z'] = 0.561 
        return extcoeff 
        
def getColorsFromMrFeH(L, Lvalues):
    SDSScolors = ['ug', 'gr', 'ri', 'iz']
    # grid properties
    MrMin = np.min(L['Mr'])
    MrMax = np.max(L['Mr'])
    dMr = L['Mr'][1]-L['Mr'][0]
    i = (Lvalues['Mr']-MrMin)/dMr
    i = i.astype(int)
    imax = (MrMax-MrMin)/dMr
    imax = imax.astype(int)
    FeHmin = np.min(L['FeH'])
    dFeH = L['FeH'][imax+2]-L['FeH'][0]
    j = (Lvalues['FeH']-FeHmin)/dFeH
    j = j.astype(int)
    k = i + j*(imax+2) + 1
    for color in SDSScolors:
        Lvalues[color] = L[color][k]
 
        
def getColorsFromGivenMrFeH(myMr, myFeH, L, gp, colors=""):
    SDSScolors = ['ug', 'gr', 'ri', 'iz'] 
    if (colors==""): colors = SDSScolors
    i = (myMr-gp['MrMin'])/gp['dMr']
    i = i.astype(int)
    j = (myFeH-gp['FeHmin'])/gp['dFeH']
    j = j.astype(int)
    k = i + j*(gp['imax']+2) + 1
    myColors = {}
    for color in colors:
        myColors[color] = L[color][k]
    return myColors


### numerical analysis ### 

# given a grid of model colors, Mcolors, compute chi2
# for a given set of observed colors Ocolors, with errors Oerrors 
# colors to be used in chi2 computation are listed in colorNames
def getLocusChi2colorsSLOW(colorNames, Mcolors, Ocolors, Oerrors):
    chi2 = 0*Mcolors[colorNames[0]]
    for i in range(0,np.size(chi2)):
        for color in colorNames:   
            chi2[i] += ((Ocolors[color]-Mcolors[color][i])/Oerrors[color])**2 
    return chi2


def getLocusChi2colors(colorNames, Mcolors, Ocolors, Oerrors):
    chi2 = 0*Mcolors[colorNames[0]]
    for color in colorNames:   
        chi2 += ((Ocolors[color]-Mcolors[color])/Oerrors[color])**2 
    return chi2


def logLfromObsColors(Mr, FeH, ObsColors, ObsColErrors, gridProperties, SDSScolors):
    myModelColors = getColorsFromGivenMrFeH(Mr, FeH, LSSTlocus, gridProperties)
    chi2 = getLocusChi2colors(SDSScolors, myModelColors, ObsColors, ObsColErrors)
    # note that logL = -1/2 * chi2 (for gaussian errors)
    return (-2*chi2)



## for testing procedures 
def getFitResults(i, gridStride, Ntest, colors, locusGrid, ObsColorErr, locusFitMS, locusFitRG, results):
    tempResults = {}
    for Q in ('chi2', 'Mr', 'FeH'):
        for Stype in ('MS', 'RG'):
            tempResults[Q, Stype] = np.zeros(Ntest)   
    
    for j in range(0,Ntest):
        #print('       test #', j)
        ObsColor = {}
        for color in colors:
            ## this is where random noise is generated and added to input magnitudes ##
            ObsColor[color] = locusGrid[color][i*gridStride] + np.random.normal(0, ObsColorErr[color])
            # print(color, locusFeH[color][i], ObsColor[color], ObsColorErr[color])
        ## chi2 for each grid point
        chi2MS = getLocusChi2colors(colors, locusFitMS, ObsColor, ObsColorErr)
        chi2RG = getLocusChi2colors(colors, locusFitRG, ObsColor, ObsColorErr)
        ## store Mr and FeH values corresponding to the minimum chi2
        # MS: 
        kmin = np.argmin(chi2MS)
        tempResults['chi2', 'MS'][j] = chi2MS[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'MS'][j] = locusFitMS[Q][kmin] 
        # RG: 
        kmin = np.argmin(chi2RG)
        tempResults['chi2', 'RG'][j] = chi2RG[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'RG'][j] = locusFitRG[Q][kmin] 

    for Q in ('chi2', 'Mr', 'FeH'):
        results[Q, 'MSmean'][i] = np.mean(tempResults[Q,'MS'])
        results[Q, 'MSstd'][i] = np.std(tempResults[Q, 'MS'])
        results[Q, 'RGmean'][i] = np.mean(tempResults[Q,'RG'])
        results[Q, 'RGstd'][i] = np.std(tempResults[Q, 'RG']) 
    return 

# given chi2 values for every grid point, return the grid point with the smallest chi2
def extractBestFit(chi2vals, locusGrid):
    kmin = np.argmin(chi2vals)
    return locusGrid[kmin], kmin


## similar to getFitResults, but for fitting a sample where each entry has DIFFERENT color errors
def fitPhotoD(i, gridStride, colors, data2fit, locusFitMS, locusFitRG, results):
    Ntest = 1
    tempResults = {}
    for Q in ('chi2', 'Mr', 'FeH'):
        for Stype in ('MS', 'RG'):
            tempResults[Q,Stype] = np.zeros(Ntest)   
    
    for j in range(0,Ntest):
        #print('       test #', j)
        ObsColor = {}
        ObsColorErr = {}
        for color in colors:
            errname = color + 'Err'
            ObsColorErr[color] = data2fit[errname][i*gridStride]
            if (0): 
                ## this is where random noise is generated and added to input magnitudes ##
                ObsColor[color] = data2fit[color][i*gridStride] + np.random.normal(0, ObsColorErr[color])
                # print(color, locusFeH[color][i], ObsColor[color], ObsColorErr[color])
            else:
                ## assuming that colors already have random noise and are corrected for extinction 
                ObsColor[color] = data2fit[color][i*gridStride] 
        ## chi2 for each grid point
        chi2MS = getLocusChi2colors(colors, locusFitMS, ObsColor, ObsColorErr)
        chi2RG = getLocusChi2colors(colors, locusFitRG, ObsColor, ObsColorErr)
        ## store Mr and FeH values corresponding to the minimum chi2
        # MS: 
        kmin = np.argmin(chi2MS)
        tempResults['chi2', 'MS'][j] = chi2MS[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'MS'][j] = locusFitMS[Q][kmin] 
        # RG: 
        kmin = np.argmin(chi2RG)
        tempResults['chi2', 'RG'][j] = chi2RG[kmin] 
        for Q in ('Mr', 'FeH'):
            tempResults[Q, 'RG'][j] = locusFitRG[Q][kmin]

        ## NB: above needs to be changed so that instead of minimum chi2
        ##     and no scatter estimate, all chi2 < chi2max are selected
        ##     and then 2D Gauss is fit (as in Berry+2012)
        ##     right not std is 0 for all cases 

    for Q in ('chi2', 'Mr', 'FeH'):
        results[Q, 'MSmean'][i] = np.mean(tempResults[Q,'MS'])
        results[Q, 'MSstd'][i] = np.std(tempResults[Q, 'MS'])
        results[Q, 'RGmean'][i] = np.mean(tempResults[Q,'RG'])
        results[Q, 'RGstd'][i] = np.std(tempResults[Q, 'RG']) 
    return 



## similar to fitPhotoD, but simply returning chi2 mag for one isochrone family, without any processing 
def getPhotoDchi2map(i, colors, data2fit, locus):

        # set up colors for fitting
        ObsColor = {}
        ObsColorErr = {}
        for color in colors:
            errname = color + 'Err'
            ObsColorErr[color] = data2fit[errname][i]
            # assuming that colors already have random noise and are corrected for extinction 
            ObsColor[color] = data2fit[color][i]
            
        ## return chi2map for each grid point
        return getLocusChi2colors(colors, locus, ObsColor, ObsColorErr)




# given vectors x and y, fit medians in bins from xMin to xMax, with Nbin steps,
# and return xBin, medianBin, medianErrBin 
def fitMedians(x, y, xMin, xMax, Nbin, verbose=1): 

    # first generate bins
    xEdge = np.linspace(xMin, xMax, (Nbin+1)) 
    xBin = np.linspace(0, 1, Nbin)
    nPts = 0*np.linspace(0, 1, Nbin)
    medianBin = 0*np.linspace(0, 1, Nbin)
    sigGbin = -1+0*np.linspace(0, 1, Nbin) 
    for i in range(0, Nbin): 
        xBin[i] = 0.5*(xEdge[i]+xEdge[i+1]) 
        yAux = y[(x>xEdge[i])&(x<=xEdge[i+1])]
        if (yAux.size > 0):
            nPts[i] = yAux.size
            medianBin[i] = np.median(yAux)
            # robust estimate of standard deviation: 0.741*(q75-q25)
            sigmaG = 0.741*(np.percentile(yAux,75)-np.percentile(yAux,25))
            # uncertainty of the median: sqrt(pi/2)*st.dev/sqrt(N)
            sigGbin[i] = np.sqrt(np.pi/2)*sigmaG/np.sqrt(nPts[i])
        else:
            nPts[i] = 0 
            medianBin[i] = 0 
            sigGbin[i] = 0 
            # nPts[i], medianBin[i], sigGBin[i] = 0 
        
    if (verbose):
        print('median:', np.median(medianBin[nPts>0]))

    return xBin, nPts, medianBin, sigGbin


def basicStats(df, colName):
    yAux = df[colName]
    # robust estimate of standard deviation: 0.741*(q75-q25)
    sigmaG = 0.741*(np.percentile(yAux,75)-np.percentile(yAux,25))
    median = np.median(yAux)
    return [np.size(yAux), np.min(yAux), np.max(yAux), median, sigmaG]


def getMedianSigG(basicStatsLine):
    med = "%.3f" % basicStatsLine[3]
    sigG = "%.2f" % basicStatsLine[4]
    return [med, sigG, basicStatsLine[0]]

def makeStatsTable0(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):
    
    # split along Mr sequence: giants, main-sequence blue and red stars
    dfG = df[df['Mr']<=Mr1]
    dfB = df[(df['Mr']>Mr1)&(df['Mr']<=Mr2)]
    dfR = df[df['Mr']>Mr2]
    # and finally split by metallicity, both because FeH sensitivity to u-g, and because halo vs. disk distinction
    #    "h" is for "halo", not "high" 
    dfGh = dfG[dfG['FeH']<=FeHthresh]
    dfGd = dfG[dfG['FeH']>FeHthresh]
    dfBh = dfB[dfB['FeH']<=FeHthresh]
    dfBd = dfB[dfB['FeH']>FeHthresh]
    dfRh = dfR[dfR['FeH']<=FeHthresh]
    dfRd = dfR[dfR['FeH']>FeHthresh]
    # and for all (without Mr split):
    dfh = df[df['FeH']<=FeHthresh]
    dfd = df[df['FeH']>FeHthresh]
 
    print('---------------------------------------------------------------------')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfh, dMrname)), getMedianSigG(basicStats(dfh, dFeHname)))
    print('          giants:', getMedianSigG(basicStats(dfGh, dMrname)), getMedianSigG(basicStats(dfGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBh, dMrname)), getMedianSigG(basicStats(dfBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfRh, dMrname)), getMedianSigG(basicStats(dfRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('             all:', getMedianSigG(basicStats(dfd, dMrname)), getMedianSigG(basicStats(dfd, dFeHname)))
    print('          giants:', getMedianSigG(basicStats(dfGd, dMrname)), getMedianSigG(basicStats(dfGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBd, dMrname)), getMedianSigG(basicStats(dfBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfRd, dMrname)), getMedianSigG(basicStats(dfRd, dFeHname)))
    print('---------------------------------------------------------')

    return


def makeStatsTable(df, dMrname='dMr', dFeHname='dFeH', magName='umag', magThresh=25.0, FeHthresh=-1.0, Mr1=4.0, Mr2=8.0):
    # first split by SNR 
    dfBright = df[df[magName]<=magThresh]
    dfFaint = df[df[magName]>magThresh]
    # then split along Mr sequence: giants, main-sequence blue and red stars
    dfBrightG = dfBright[dfBright['Mr']<=Mr1]
    dfBrightB = dfBright[(dfBright['Mr']>Mr1)&(dfBright['Mr']<=Mr2)]
    dfBrightR = dfBright[dfBright['Mr']>Mr2]
    dfFaintG = dfFaint[dfFaint['Mr']<=Mr1]
    dfFaintB = dfFaint[(dfFaint['Mr']>Mr1)&(dfFaint['Mr']<=Mr2)]
    dfFaintR = dfFaint[dfFaint['Mr']>Mr2]
    # and finally split by metallicity, both because FeH sensitivity to u-g, and because halo vs. disk distinction
    dfBrightGh = dfBrightG[dfBrightG['FeH']<=FeHthresh]
    dfBrightGd = dfBrightG[dfBrightG['FeH']>FeHthresh]
    dfBrightBh = dfBrightB[dfBrightB['FeH']<=FeHthresh]
    dfBrightBd = dfBrightB[dfBrightB['FeH']>FeHthresh]
    dfBrightRh = dfBrightR[dfBrightR['FeH']<=FeHthresh]
    dfBrightRd = dfBrightR[dfBrightR['FeH']>FeHthresh]
    dfFaintGh = dfFaintG[dfFaintG['FeH']<=FeHthresh]
    dfFaintGd = dfFaintG[dfFaintG['FeH']>FeHthresh]
    dfFaintBh = dfFaintB[dfFaintB['FeH']<=FeHthresh]
    dfFaintBd = dfFaintB[dfFaintB['FeH']>FeHthresh]
    dfFaintRh = dfFaintR[dfFaintR['FeH']<=FeHthresh]
    dfFaintRd = dfFaintR[dfFaintR['FeH']>FeHthresh]

    print('---------------------------------------------------------------------')
    print(' -- high SNR  ---')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfBrightGh, dMrname)), getMedianSigG(basicStats(dfBrightGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBrightBh, dMrname)), getMedianSigG(basicStats(dfBrightBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfBrightRh, dMrname)), getMedianSigG(basicStats(dfBrightRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfBrightGd, dMrname)), getMedianSigG(basicStats(dfBrightGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfBrightBd, dMrname)), getMedianSigG(basicStats(dfBrightBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfBrightRd, dMrname)), getMedianSigG(basicStats(dfBrightRd, dFeHname)))
    print(' --  low SNR  ---')
    print('       low [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfFaintGh, dMrname)), getMedianSigG(basicStats(dfFaintGh, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfFaintBh, dMrname)), getMedianSigG(basicStats(dfFaintBh, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfFaintRh, dMrname)), getMedianSigG(basicStats(dfFaintRh, dFeHname)))
    print('      high [FeH]:          ', dMrname, '                ', dFeHname)
    print('          giants:', getMedianSigG(basicStats(dfFaintGd, dMrname)), getMedianSigG(basicStats(dfFaintGd, dFeHname)))
    print('         blue MS:', getMedianSigG(basicStats(dfFaintBd, dMrname)), getMedianSigG(basicStats(dfFaintBd, dFeHname)))
    print('          red MS:', getMedianSigG(basicStats(dfFaintRd, dMrname)), getMedianSigG(basicStats(dfFaintRd, dFeHname)))
    print('---------------------------------------------------------')

    return





def getLSSTm5(data, depth='coadd'):
    # temporary: only use SDSS colors
    colors = ['u', 'g', 'r', 'i', 'z']
    # from https://iopscience.iop.org/article/10.3847/1538-4365/ac3e72
    coaddm5 = {}
    coaddm5['u'] = 25.73
    coaddm5['g'] = 26.86
    coaddm5['r'] = 26.88
    coaddm5['i'] = 26.34
    coaddm5['z'] = 25.63
    coaddm5['y'] = 24.87
    singlem5 = {}
    singlem5['u'] = 23.50
    singlem5['g'] = 24.44 
    singlem5['r'] = 23.98 
    singlem5['i'] = 23.41
    singlem5['z'] = 22.77
    singlem5['y'] = 22.01
    gg = {}
    gg['u'] = 0.038 
    gg['g'] = 0.039 
    gg['r'] = 0.039 
    gg['i'] = 0.039 
    gg['z'] = 0.039 
    gg['y'] = 0.039   
    m5 = {}
    for b in colors:
        if (depth=='coadd'):
            m5[b] = coaddm5[b] 
        else:
            m5[b] = singlem5[b] 
    mags = {}
    mags['u'] = data['umag']
    mags['g'] = data['gmag']
    mags['r'] = data['rmag']
    mags['i'] = data['imag']
    mags['z'] = data['zmag']
    errors = {}
    for b in colors:
        x = 10**(0.4*(mags[b]-m5[b]))
        errors[b] = np.sqrt(0.005**2 + (0.04-gg[b])*x + gg[b]*x**2)
    return errors





### IMPLEMENTATION OF MAP-BASED PRIORS 

def get2Dmap(sample, labels, metadata):
    data = np.vstack([sample[labels[0]], sample[labels[1]]])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = int(metadata[2])
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = int(metadata[5])
    xgrid = np.linspace(xMin, xMax, nXbin)
    ygrid = np.linspace(yMin, yMax, nYbin)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    return (Xgrid, Ygrid, Z)

def dumpPriorMaps(sample, rmagMin, rmagMax, rmagNsteps, fileRootname, make_plot=False):
    #### probably astrophysically ok, but it should be read from a configuration file...
    fitting_params = getFittingParams()
    
    FeHmin = fitting_params['FeH_MIN']
    FeHmax = fitting_params['FeH_MAX']
    FeHNpts = fitting_params['FeH_NPTS']
    MrFaint = fitting_params['Mr_FAINT']
    MrBright = fitting_params['Mr_BRIGHT']
    MrNpts = fitting_params['Mr_NPTS']
    rmagBinWidth = fitting_params['r_BinSize']  # should be dependent on rmag and larger for bright mags
    # -------
    labels = ['M_H', 'Mr']
    plot_labels = ['FeH', 'Mr']
    metadata = np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    
    file_params = getFileParams()
    output_dir = file_params['OutputFilePath']
    
    for rind, r in enumerate(rGrid):
        # r magnitude limits for this subsample
        rMin = r - 0.5*rmagBinWidth
        rMax = r + 0.5*rmagBinWidth
        # select subsample
        tS = sample[(sample['rmag']>rMin)&(sample['rmag']<rMax)]
        tSsize = np.size(tS)
        print('r=', rMin, 'to', rMax, 'N=', np.size(sample), 'Ns=', np.size(tS))
        # this is work horse, where data are binned and map arrays produced
        # it takes about 2 mins on Mac OS Apple M1 Pro
        # so about 70 hours for 2,000 healpix samples 
        # maps per healpix are about 2 MB, total 4 GB

        try:
            xGrid, yGrid, Z = get2Dmap(tS, labels, metadata)
        
            if make_plot:
                # display for sanity tests, it can be turned off
                filename = output_dir + '/{}_{}.png'.format(fileRootname, rind)
                pt.show2Dmap(xGrid, yGrid, Z, metadata, plot_labels[0], plot_labels[1], filename, logScale=True)
        
            # store this particular map (given healpix and r band magnitude slice) 
            extfile = "-%02d" % (rind)
            # it is assumed that fileRootname knows which healpix is being processed,
            # as well as the magnitude range specified by rmagMin and rmagMax
            outfile = output_dir + '/' + fileRootname + extfile + '.npz' 
            Zrshp = Z.reshape(xGrid.shape)
            mdExt = np.concatenate((metadata, np.array([rmagMin, rmagMax, rmagNsteps, rmagBinWidth, tSsize, r])))
            np.savez(outfile, xGrid=xGrid, yGrid=yGrid, kde=Zrshp, metadata=mdExt, labels=labels)

        except:
            print('r=', rMin, 'to', rMax, 'Ns=', tSsize, '=> Not enough stars in the bin')
        

def readPrior(rmagMin, rmagMax, rmagNsteps,rootname):
    # read all maps, index by rmag index 
    rGrid = np.linspace(rmagMin, rmagMax, rmagNsteps)
    priors = {}
    
    file_params = getFileParams()
    output_dir = file_params['OutputFilePath']
    
    for rind, r in enumerate(rGrid):
        extfile = "-%02d" % (rind)
        infile = output_dir + '/' + rootname + extfile + '.npz' 
        priors[rind] = np.load(infile)
    rmagBinWidth = priors[0]['metadata'][9] # volatile
    return priors, rmagBinWidth



def getMetadataPriors(priorMap=""):
    if (priorMap==""):
        FeHmin = -2.5
        FeHmax = 1.0
        FeHNpts = 35
        MrFaint = 15.0
        MrBright = -2.0 
        MrNpts = 85
    else:
        FeHmin = priorMap['metadata'][0]
        FeHmax = priorMap['metadata'][1]
        FeHNpts = priorMap['metadata'][2]
        MrFaint = priorMap['metadata'][3]
        MrBright = priorMap['metadata'][4]
        MrNpts = priorMap['metadata'][5]
    return np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])


def getMetadataLikelihood(locusData=""):
    if (locusData==""):
        raise Exception("you must specify locus as it is not uniquely determined!")
    else:
        FeHmin = np.min(locusData['FeH'])
        FeHmax = np.max(locusData['FeH'])
        FeHNpts = np.size(np.unique(locusData['FeH']))  
        MrFaint = np.max(locusData['Mr'])
        MrBright = np.min(locusData['Mr'])
        MrNpts = np.size(np.unique(locusData['Mr']))
    return np.array([FeHmin, FeHmax, FeHNpts, MrFaint, MrBright, MrNpts])


def pnorm(pdf):
    pdf = pdf/np.sum(pdf)
    
def getStats(x,pdf):
    mean = np.sum(x*pdf)/np.sum(pdf)
    V = np.sum((x-mean)**2*pdf)/np.sum(pdf)
    return mean, np.sqrt(V)

   
    
def showPosterior(iS): 

    # grid (same for all stars)
    xLabel = 'FeH'
    yLabel = 'Mr'
    FeHGrid = locusRG[xLabel]
    MrGrid = locusRG[yLabel]
    FeH1d = np.sort(np.unique(FeHGrid))
    Mr1d = np.sort(np.unique(MrGrid)) 
    
    ## for selecting prior
    rmagStar = rmagTrue[iS]

    ## for comparison
    MrStar = MrTrue[iS]
    FeHStar = FeHTrue[iS]
    print('rmagStar =', rmagStar, 'true Mr=', MrStar, 'true FeH=',FeHStar)

    ## likelihood info     
    # chi2/likelihood info for this (iS) star
    chi2Grid = chi2RG[iS]
    likeGrid = np.exp(-0.5*chi2Grid)

    # axis limits for likelihood maps
    mdLocusRG = getMetadataLikelihood(locusRG)
    
    # get map indices for observed r band mags
    rObs = np.array([rmagStar, rmagStar]) 
    rind = np.arange(rmagNsteps)
    zint = np.interp(rObs, rGrid, rind) + rmagBinWidth 
    priorind = zint.astype(int)
    
    # interpolate a prior map on locus Mr-FeH grid and show the values as points color-coded by prior
    Z = priors[priorind[0]]
    Zval = Z['kde']
    X = Z['xGrid']
    Y = Z['yGrid']
    points = np.array((X.flatten(), Y.flatten())).T
    values = Zval.flatten()
    # actual (linear) interpolation
    Z0 = griddata(points, values, (locusRG['FeH'], locusRG['Mr']), method='linear')
    
    ## posterior
    posterior = likeGrid * Z0 
    post2d = posterior.reshape(np.size(FeH1d), np.size(Mr1d))
    
    # marginalize and get stats 
    margpostMr = np.sum(post2d, axis=0)
    margpostFeH = np.sum(post2d, axis=1)
    pnorm(margpostMr)
    pnorm(margpostFeH)
    # stats
    meanFeH, stdFeH = getStats(FeH1d, margpostFeH)
    meanMr, stdMr = getStats(Mr1d, margpostMr)
    print('Mr=', meanMr,'+-', stdMr)
    print('FeH=', meanFeH,'+-', stdFeH)
    
    # show plots
    pt.show3Flat2Dmaps(Z0, likeGrid, posterior, mdLocusRG, xLabel, yLabel, logScale=True, x0=FeHStar, y0=MrStar)
    pt.showMargPosteriors(Mr1d, margpostMr, 'Mr', 'p(Mr)', FeH1d, margpostFeH, 'FeH', 'p(FeH)', MrStar, FeHStar)
    
    return  
