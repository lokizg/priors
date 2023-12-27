import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde

import configure

# def makeDMhistogram(x, xMin=1, xMax=22):
# def plot2Dmap(x, y, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):
# def replot2Dmap(Xgrid, Ygrid, Z, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):
# def restore2Dmap(npz, logScale=False):
# def show2Dmap(Xgrid, Ygrid, Z, metadata, xLabel, yLabel, logScale=False):
# def showFlat2Dmap(Z, metadata, xLabel, yLabel, logScale=False):
# def show3Flat2Dmaps(Z1, Z2, Z3, md, xLab, yLab, x0=-99, y0=-99, logScale=False, minFac=1000, cmap='Blues'):
# def showMargPosteriors(x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, trueX1, trueX2): 



    
# make more sanity plots 
def makeDMhistogram(x, xMin=1, xMax=22):

    ### PLOTTING ###
    plot_kwargs = dict(color='k', linestyle='none', marker='.', markersize=1)
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.98, wspace=0.4, hspace=0.4)

    hist, bins = np.histogram(x, bins=50)
    center = (bins[:-1]+bins[1:])/2
    ax1 = plt.subplot(3,1,1)
    ax1.plot(center, hist, drawstyle='steps')   
    ax1.set_xlim(xMin, xMax)
    ax1.set_xlabel(r'$\mathrm{Distance Modulus}$')
    ax1.set_ylabel(r'$\mathrm{dN/dDM}$')

    # save
    plt.savefig('DMhistogram.png')
    plt.show() 

    return


def plot2Dmap(x, y, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):

    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    xgrid = np.linspace(xMin, xMax, nXbin)
    ygrid = np.linspace(yMin, yMax, nYbin)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(vmin=Z.min(), vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    if (0):     
        xBin, nPts, medianBin, sigGbin = lt.fitMedians(x, y, 0, 12, 60)
        plt.scatter(xBin, medianBin, s=40, c='red')   
        plt.plot([-1,13], [0,0], c='black')   
        
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    return (Xgrid, Ygrid, Z)


def replot2Dmap(Xgrid, Ygrid, Z, xMin, xMax, nXbin, yMin, yMax, nYbin, xLabel, yLabel, logScale=False):

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    return 



def restore2Dmap(npz, mid_rMag, logScale=False, save_plot=False):
    
    im = npz['kde'].reshape(npz['xGrid'].shape)
    xMin = npz['metadata'][0]
    xMax = npz['metadata'][1]
    nXbin = npz['metadata'][2]
    yMin = npz['metadata'][3]
    yMax = npz['metadata'][4]
    nYbin = npz['metadata'][5]
    xLabel = npz['labels'][0]
    yLabel = npz['labels'][1]
    
    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(im,
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=im.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(im,
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    
    if save_plot:
        filename =  './restorePriors_{}.png'.format(mid_rMag)
        plt.savefig(filename)
    
    return


def show2Dmap(Xgrid, Ygrid, Z, metadata, xLabel, yLabel, filename, logScale=False):

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = metadata[2]
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = metadata[5]

    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig(filename)
    #plt.show()


def showFlat2Dmap(Z, metadata, xLabel, yLabel, logScale=False):

    # evaluate on a regular grid
    xMin = metadata[0]
    xMax = metadata[1]
    nXbin = metadata[2]
    yMin = metadata[3]
    yMax = metadata[4]
    nYbin = metadata[5]
    
    Xpts = nXbin.astype(int)
    Ypts = nYbin.astype(int)
    im = Z.reshape((Xpts, Ypts))

    fig, ax = plt.subplots(1,1,figsize=(6,4.5))

    # plot the result as an image
    if (logScale):
        from matplotlib.colors import LogNorm
        plt.imshow(im.T,
               origin='upper', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues', 
               norm=LogNorm(0.001, vmax=Z.max()))
        cb = plt.colorbar()
        cb.set_label("density on log scale")
    else: 
        plt.imshow(im.T,
               origin='upper', aspect='auto',
               extent=[xMin, xMax, yMin, yMax],
               cmap='Blues') 
        cb = plt.colorbar()
        cb.set_label("density on lin scale")

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('x.png')
    plt.show() 
    
  
    
def show3Flat2Dmaps(Z1, Z2, Z3, md, xLab, yLab, x0=-99, y0=-99, logScale=False, minFac=1000, cmap='Blues'):

    # unpack metadata
    xMin = md[0]
    xMax = md[1]
    nXbin = md[2]
    yMin = md[3]
    yMax = md[4]
    nYbin = md[5]
    # set local variables and
    myExtent=[xMin, xMax, yMin, yMax]
    Xpts = nXbin.astype(int)
    Ypts = nYbin.astype(int)
    # reshape flattened input arrays to get "images"
    im1 = Z1.reshape((Xpts, Ypts))
    im2 = Z2.reshape((Xpts, Ypts))
    im3 = Z3.reshape((Xpts, Ypts))
    
    showTrue = False
    if ((x0>-99)&(y0>-99)):
        showTrue = True
        
    def oneImage(ax, image, extent, minFactor, title, showTrue, x0, y0, logScale=True, cmap='Blues'):
        im = image/image.max()
        ImMin = im.max()/minFactor
        if (logScale):
            cmap = ax.imshow(im.T,
               origin='upper', aspect='auto', extent=extent,
               cmap=cmap, norm=LogNorm(ImMin, vmax=im.max()))
            ax.set_title(title)
        else:
            cmap = ax.imshow(im.T,
               origin='upper', aspect='auto', extent=extent,
               cmap=cmap)
            ax.set_title(title)
        if (showTrue):
            ax.scatter(x0, y0, s=150, c='red') 
            ax.scatter(x0, y0, s=40, c='yellow') 
        return cmap
 
    fig, axs = plt.subplots(1,3,figsize=(14,4))

    # plot  
    from matplotlib.colors import LogNorm
    cmap = oneImage(axs[0], im1, myExtent, minFac, 'Prior', showTrue, x0, y0, logScale=logScale)
    cmap = oneImage(axs[1], im2, myExtent, minFac, 'Likelihood', showTrue, x0, y0, logScale=logScale)
    cmap = oneImage(axs[2], im3, myExtent, minFac, 'Posterior', showTrue, x0, y0, logScale=logScale)

    cax = fig.add_axes([0.84, 0.1, 0.1, 0.75])
    cax.set_axis_off()
    cb = fig.colorbar(cmap, ax=cax)
    if (logScale):
        cb.set_label("density on log scale")
    else:
        cb.set_label("density on linear scale")

    for ax in axs.flat:
        ax.set(xlabel=xLab, ylabel=yLab)

    plt.savefig('bayesPanels.png')
    plt.show() 
    

def showMargPosteriors(x1d1, margp1, xLab1, yLab1, x1d2, margp2, xLab2, yLab2, trueX1, trueX2): 
 
    fig, axs = plt.subplots(1,2,figsize=(9,4))
    # plot 
    axs[0].plot(x1d1, margp1)
    axs[0].set(xlabel=xLab1, ylabel=yLab1)
    axs[0].plot([trueX1, trueX1], [np.min(margp1), 1.05*np.max(margp1)], '--r')
    
    axs[1].plot(x1d2, margp2)
    axs[1].set(xlabel=xLab2, ylabel=yLab2)
    axs[1].plot([trueX2, trueX2], [np.min(margp2), 1.05*np.max(margp2)], '--r')


    plt.savefig('margPosteriors.png')
    plt.show() 
