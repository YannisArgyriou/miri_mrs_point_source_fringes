"""Functions file used for calibration of MIRI data.

:History:

Created on Thu Mar 01 10:58:50 2018

@author: Ioannis Argyriou (KULeuven, Belgium, ioannis.argyriou@kuleuven.be)
"""

# import python modules
import os
import pickle
import itertools
import numpy as np
from scipy import arcsin
import scipy.special as sp
from scipy.optimize import curve_fit, least_squares
from astropy.io import fits
from astropy.modeling.functional_models import Moffat1D
import scipy.interpolate as scp_interpolate
import matplotlib.pyplot as plt

# Definition
#--auxilliary data
def mrs_aux(band):
    allbands = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
    allchannels = ['1','2','3','4']
    allsubchannels = ['A','B','C']

    # slice IDs on detector
    sliceid1=[111,121,110,120,109,119,108,118,107,117,106,116,105,115,104,114,103,113,102,112,101]
    sliceid2=[201,210,202,211,203,212,204,213,205,214,206,215,207,216,208,217,209]
    sliceid3=[316,308,315,307,314,306,313,305,312,304,311,303,310,302,309,301]
    sliceid4=[412,406,411,405,410,404,409,403,408,402,407,401]

    MRS_bands = {'1A':[4.885,5.751],
        '1B':[5.634,6.632],
        '1C':[6.408,7.524],
        '2A':[7.477,8.765],
        '2B':[8.711,10.228],
        '2C':[10.017,11.753],
        '3A':[11.481,13.441],
        '3B':[13.319,15.592],
        '3C':[15.4,18.072],
        '4A':[17.651,20.938],
        '4B':[20.417,24.22],
        '4C':[23.884,28.329]} # microns

    MRS_R = {'1A':[3320.,3710.],
        '1B':[3190.,3750.],
        '1C':[3100.,3610.],
        '2A':[2990.,3110.],
        '2B':[2750.,3170.],
        '2C':[2860.,3300.],
        '3A':[2530.,2880.],
        '3B':[1790.,2640.],
        '3C':[1980.,2790.],
        '4A':[1460.,1930.],
        '4B':[1680.,1770.],
        '4C':[1630.,1330.]} # R = lambda / delta_lambda

    MRS_lambpix = {'1A':0.0008,
        '1B':0.0009,
        '1C':0.001,
        '2A':0.0014,
        '2B':0.0017,
        '2C':0.0020,
        '3A':0.0023,
        '3B':0.0026,
        '3C':0.0030,
        '4A':0.0036,
        '4B':0.0042,
        '4C':0.0048} # average pixel spectral size

    MRS_nslices = {'1':21,'2':17,'3':16,'4':12} # number of slices

    MRS_alphapix = {'1':0.196,'2':0.196,'3':0.245,'4':0.273} # arcseconds

    MRS_slice = {'1':0.176,'2':0.277,'3':0.387,'4':0.645} # arcseconds

    MRS_FOV = {'1':[3.70,3.70],'2':[4.51,4.71],'3':[6.13,6.19],'4':[7.74,7.74]} # arcseconds along and across slices

    MRS_FWHM = {'1':0.423,'2':0.647,'3':0.99,'4':1.518} # MRS PSF

    return allbands,allchannels,allsubchannels,MRS_bands[band],MRS_R[band],MRS_alphapix[band[0]],MRS_slice[band[0]],MRS_FOV[band[0]],MRS_FWHM[band[0]],MRS_lambpix[band]

def band_to_subband(band):
    if band[1] == 'A':   subband_id = 'SHORT'
    elif band[1] == 'B': subband_id = 'MEDIUM'
    elif band[1] == 'C': subband_id = 'LONG'

    return subband_id

def band_to_det(band):
    if band[0] in ['1','2']:   det_id = 'SHORT'
    elif band[0] in ['3','4']: det_id = 'LONG'
    return det_id

def band_to_channel(band):
    if band[0] in ['1','2']:   channel_id = '12'
    elif band[0] in ['3','4']: channel_id = '34'
    return channel_id

def point_source_centroiding(band,sci_img,d2cMaps,spec_grid=None,fit='2D',center=None,offset_slice=0):
    # distortion maps
    sliceMap  = d2cMaps['sliceMap']
    lambdaMap = d2cMaps['lambdaMap']
    alphaMap  = d2cMaps['alphaMap']
    betaMap   = d2cMaps['betaMap']
    nslices   = d2cMaps['nslices']
    MRS_alphapix = {'1':0.196,'2':0.196,'3':0.245,'4':0.273} # arcseconds
    MRS_FWHM = {'1':2.16*MRS_alphapix['1'],'2':3.30*MRS_alphapix['2'],
                '3':4.04*MRS_alphapix['3'],'4':5.56*MRS_alphapix['4']} # MRS PSF
    mrs_fwhm  = MRS_FWHM[band[0]]
    lambcens,lambfwhms = spec_grid[0],spec_grid[1]
    unique_betas = np.sort(np.unique(betaMap[(sliceMap>100*int(band[0])) & (sliceMap<100*(int(band[0])+1))]))
    fov_lims  = [alphaMap[np.nonzero(lambdaMap)].min(),alphaMap[np.nonzero(lambdaMap)].max()]

    print('STEP 1: Rough centroiding')
    if center is None:
        # premise> center of point source is located in slice with largest signal
        # across-slice center:
        sum_signals = np.zeros(nslices)
        for islice in range(1,1+nslices):
            sum_signals[islice-1] = sci_img[(sliceMap == 100*int(band[0])+islice) & (~np.isnan(sci_img))].sum()
        source_center_slice = np.argmax(sum_signals)+1
        source_center_slice+=offset_slice

        # along-slice center:
        det_dims = (1024,1032)
        img = np.full(det_dims,0.)
        sel = (sliceMap == 100*int(band[0])+source_center_slice)
        img[sel]  = sci_img[sel]

        source_center_alphas = []
        for row in range(det_dims[0]):
            source_center_alphas.append(alphaMap[row,img[row,:].argmax()])
        source_center_alphas = np.array(source_center_alphas)
        source_center_alpha  = np.average(source_center_alphas[~np.isnan(source_center_alphas)])
    else:
        source_center_slice,source_center_alpha = center[0],center[1]
    # summary:
    print( 'Slice {} has the largest summed flux'.format(source_center_slice))
    print( 'Source position: beta = {}arcsec, alpha = {}arcsec \n'.format(round(unique_betas[source_center_slice-1],2),round(source_center_alpha,2)))

    if fit == '0D':
        return source_center_slice,unique_betas[source_center_slice-1],source_center_alpha

    print( 'STEP 2: 1D Gaussian fit')

    # Fit Gaussian distribution to along-slice signal profile
    sign_amp,alpha_centers,alpha_fwhms,bkg_signal = [np.full((len(lambcens)),np.nan) for j in range(4)]
    failed_fits = []
    for ibin in range(len(lambcens)):
        coords = np.where((sliceMap == 100*int(band[0])+source_center_slice) & (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img)))
        if len(coords[0]) == 0: failed_fits.append(ibin); continue
        try:popt,pcov = curve_fit(gauss1d_wBaseline, alphaMap[coords], sci_img[coords], p0=[sci_img[coords].max(),source_center_alpha,mrs_fwhm/2.355,0],method='lm')
        except: failed_fits.append(ibin); continue
        sign_amp[ibin]      = popt[0]+popt[3]
        alpha_centers[ibin] = popt[1]
        alpha_fwhms[ibin]   = 2.355*np.abs(popt[2])
        bkg_signal[ibin]    = popt[3]

    # omit outliers
    for i in range(len(np.diff(sign_amp))):
        if np.abs(np.diff(alpha_centers)[i]) > 0.05:
            sign_amp[i],sign_amp[i+1],alpha_centers[i],alpha_centers[i+1],alpha_fwhms[i],alpha_fwhms[i+1] = [np.nan for j in range(6)]

    print( '[Along-slice fit] The following bins failed to converge:')
    print( failed_fits)

    # Fit Gaussian distribution to across-slice signal profile (signal brute-summed in each slice)
    summed_signal,beta_centers,beta_fwhms = [np.full((len(lambcens)),np.nan) for j in range(3)]
    failed_fits = []
    for ibin in range(len(lambcens)):
        if np.isnan(alpha_centers[ibin]): failed_fits.append(ibin);continue
        sel = (np.abs(lambdaMap-lambcens[ibin])<=lambfwhms[ibin]/2.) & (~np.isnan(sci_img))
        try:signals = np.array([sci_img[(sliceMap == 100*int(band[0])+islice) & sel][np.abs(alphaMap[(sliceMap == 100*int(band[0])+islice) & sel]-alpha_centers[ibin]).argmin()] for islice in range(1,1+nslices)])
        except ValueError: failed_fits.append(ibin); continue
        try:popt,pcov = curve_fit(gauss1d_wBaseline, unique_betas, signals, p0=[signals.max(),unique_betas[source_center_slice-1],mrs_fwhm/2.355,0],method='lm')
        except: failed_fits.append(ibin); continue
        summed_signal[ibin] = popt[0]+popt[3]
        beta_centers[ibin]  = popt[1]
        beta_fwhms[ibin]    = 2.355*np.abs(popt[2])

    # # omit outliers
    # for i in range(len(np.diff(summed_signal))):
    #     if np.abs(np.diff(beta_centers)[i]) > 0.05:
    #         summed_signal[i],summed_signal[i+1],beta_centers[i],beta_centers[i+1],beta_fwhms[i],beta_fwhms[i+1] = [np.nan for j in range(6)]

    print( '[Across-slice fit] The following bins failed to converge:')
    print( failed_fits)
    print( '')

    if fit == '1D':
        sigma_alpha, sigma_beta = alpha_fwhms/2.355, beta_fwhms/2.355
        return sign_amp,alpha_centers,beta_centers,sigma_alpha,sigma_beta,bkg_signal

    elif fit == '2D':
        print( 'STEP 3: 2D Gaussian fit')
        sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D = [np.full((len(lambcens)),np.nan) for j in range(6)]
        failed_fits = []

        for ibin in range(len(lambcens)):
            # initial guess for fitting, informed by previous centroiding steps
            amp,alpha0,beta0  = sign_amp[ibin],alpha_centers[ibin],beta_centers[ibin]
            sigma_alpha, sigma_beta = alpha_fwhms[ibin]/2.355, beta_fwhms[ibin]/2.355
            base = 0
            guess = [amp, alpha0, beta0, sigma_alpha, sigma_beta, base]
            bounds = ([0,-np.inf,-np.inf,0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

            # data to fit
            coords = (np.abs(lambdaMap-lambcens[ibin])<lambfwhms[ibin]/2.)
            alphas, betas, zobs   = alphaMap[coords],betaMap[coords],sci_img[coords]
            alphabetas = np.array([alphas,betas])

            # perform fitting
            try:popt,pcov = curve_fit(gauss2d, alphabetas, zobs, p0=guess,bounds=bounds)
            except: failed_fits.append(ibin); continue

            sign_amp2D[ibin]      = popt[0]
            alpha_centers2D[ibin] = popt[1]
            beta_centers2D[ibin]  = popt[2]
            sigma_alpha2D[ibin]   = popt[3]
            sigma_beta2D[ibin]    = popt[4]
            bkg_amp2D[ibin]       = popt[5]

        print( 'The following bins failed to converge:')
        print( failed_fits)

        return sign_amp2D,alpha_centers2D,beta_centers2D,sigma_alpha2D,sigma_beta2D,bkg_amp2D

#--fit
# 1d
def gauss1d_wBaseline(x, A, mu, sigma, baseline):
    """1D Gaussian distribution function"""
    G_nu = (1./sigma) * np.sqrt(4*np.log(2)/np.pi) * np.exp(-4*np.log(2)*((x-mu)/sigma)**2)
    return  (A * G_nu / np.max(G_nu))+baseline

def gauss1d_woBaseline(x, A, mu, sigma):
    """1D Gaussian distribution function"""
    G_nu = (1./sigma) * np.sqrt(4*np.log(2)/np.pi) * np.exp(-4*np.log(2)*((x-mu)/sigma)**2)
    return  A * G_nu / np.max(G_nu)

def FPfunc_noPhaseShift(wavenumber,R,D,theta=0):
    # T = 1-R-A
    return (1 + (4*R/(1-R)**2) * np.sin(2*np.pi*D*wavenumber*np.cos(theta))**2 )**-1

# 2d
def gauss2d(xy, amp, x0, y0, sigma_x, sigma_y, base):
    # assert that values are floats
    amp, x0, y0, sigma_x, sigma_y, base = float(amp),float(x0),float(y0),float(sigma_x),float(sigma_y),float(base)
    x, y = xy
    a = 1/(2*sigma_x**2)
    b = 1/(2*sigma_y**2)
    inner = a * (x - x0)**2
    inner += b * (y - y0)**2
    return amp * np.exp(-inner) + base

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    # 1D output
    return g.ravel()

#--normalize fringes
def norm_fringe(sci_data,thres=0,min_dist=2,k=3,ext=3):
    # determine peaks
    sci_data_noNaN = sci_data.copy()
    sci_data_noNaN[np.isnan(sci_data_noNaN)] = 0.
    peaks = find_peaks(sci_data_noNaN,thres=thres,min_dist=min_dist)
    # determine fringe continuum
    if len(peaks)!=0:
        # omit peak at boundary of array (false positive)
        if peaks[0] == np.nonzero(sci_data_noNaN)[0][0]:
            peaks = np.delete(peaks,0)

        if len(peaks)>1:
            arr_interpolator = scp_interpolate.InterpolatedUnivariateSpline(peaks,sci_data_noNaN[peaks],k=k,ext=ext)
            sci_data_profile = arr_interpolator(range(len(sci_data_noNaN)))
        elif len(peaks)==1:
            sci_data_profile = sci_data_noNaN[peaks]*np.ones(len(sci_data_noNaN))
        elif len(peaks)==0:
            sci_data_profile = np.zeros(len(sci_data))

        return sci_data_noNaN,peaks,sci_data_profile

    else:
        return sci_data_noNaN,peaks,np.zeros(len(sci_data))

#--find
def find_nearest(array,value):
    return np.abs(array-value).argmin()

def find_peaks(ydata, thres=0.3, min_dist=1):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude ydata to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(ydata, np.ndarray) and np.issubdtype(ydata.dtype, np.unsignedinteger):
        raise ValueError("ydata must be signed")

    y = ydata.copy()
    y[np.isnan(y)] = 0

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)

    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks

def detpixel_trace(band,d2cMaps,sliceID=None,alpha_pos=None):
    # detector dimensions
    det_dims=(1024,1032)
    # initialize placeholders
    ypos,xpos = np.arange(det_dims[0]),np.zeros(det_dims[0])
    slice_img,alpha_img = [np.full(det_dims,0.) for j in range(2)]
    # create pixel masks
    sel_pix = (d2cMaps['sliceMap'] == 100*int(band[0])+sliceID) # select pixels with correct slice number
    slice_img[sel_pix] = d2cMaps['sliceMap'][sel_pix]           # image containing single slice
    alpha_img[sel_pix] = d2cMaps['alphaMap'][sel_pix]           # image containing alpha positions in single slice

    # find pixel trace
    for row in ypos:
        if band[0] in ['1','4']:
            xpos[row] = np.argmin(alpha_img[row,:])+find_nearest(alpha_img[row,:][(slice_img[row,:]!=0)],alpha_pos)
        elif band[0] in ['2','3']:
            xpos[row] = np.argmax(alpha_img[row,:])+find_nearest(alpha_img[row,:][(slice_img[row,:]!=0)],alpha_pos)
    xpos = xpos.astype(int)

    return ypos,xpos

def detpixel_isolambda(band,d2cMaps,sliceID=None,lambda_pos=None):
    # initialize placeholders
    det_dims = (1024,1032)
    lambda_img = np.full(det_dims,0.)
    # create pixel masks
    sel_pix = (d2cMaps['sliceMap'] == 100*int(band[0])+sliceID) # select pixels with correct slice number
    lambda_img[sel_pix] = d2cMaps['lambdaMap'][sel_pix]           # image containing alpha positions in single slice

    lower = np.where(lambda_img[0,:]!=0)[0][0]-1
    upper = np.where(lambda_img[512,:]!=0)[0][-1]+1
    check_columns = np.arange(lower,upper+1)

    # find pixel trace
    ypos,xpos = np.zeros(len(check_columns)),check_columns
    for i,column in enumerate(check_columns):
        ypos[i] = find_nearest(lambda_img[:,column],lambda_pos)
    ypos = ypos.astype(int)

    # choose range
    sel = np.where((ypos != 0))
    ypos = ypos[sel]
    xpos = xpos[sel]

    sel = np.where((ypos != 1023))
    ypos = ypos[sel]
    xpos = xpos[sel]

    sel = np.append(np.where(abs(np.diff(ypos)) < 2)[0],np.where(abs(np.diff(ypos)) < 2)[0][-1]+1)
    ypos = ypos[sel]
    xpos = xpos[sel]

    return ypos,xpos

def detpixel_trace_compactsource(sci_img,band,d2cMaps,offset_slice=0,verbose=False):
    # detector dimensions
    det_dims  = (1024,1032)
    nslices   = d2cMaps['nslices']
    sliceMap  = d2cMaps['sliceMap']
    lambdaMap = d2cMaps['lambdaMap']
    ypos,xpos = np.arange(det_dims[0]),np.zeros(det_dims[0])

    sum_signals = np.zeros(nslices)
    for islice in range(1+nslices):
        sum_signals[islice-1] = sci_img[(sliceMap == 100*int(band[0])+islice) & (~np.isnan(sci_img))].sum()
    source_center_slice = np.argmax(sum_signals)+1
    if verbose==True:
        print( 'Source center slice ID: {}'.format(source_center_slice))

    signal_img = np.full(det_dims,0.)
    sel_pix = (sliceMap == 100*int(band[0])+source_center_slice+offset_slice)
    signal_img[sel_pix] = sci_img[sel_pix]
    for row in ypos:
        signal_row = signal_img[row,:].copy()
        signal_row[np.isnan(signal_row)] = 0.
        xpos[row] = np.argmax(signal_row)
        if (row>1) & (row<=512) & (xpos[row]<xpos[row-1]):
            xpos[row] = xpos[row-1]
        elif (row>1) & (row>512) & (xpos[row]>xpos[row-1]):
            xpos[row] = xpos[row-1]
    xpos = xpos.astype(int)
    # correct edge effects
    xpos[:2] = xpos[2]
    xpos[-2:] = xpos[-3]
    # there can be no jumps/discontinuities of more than 3 pixels, run loops twice
    if len(np.where(abs(np.diff(xpos))>1)[0]) > 0:
        xpos[np.where(abs(np.diff(xpos))>1)[0][0]+1:] = xpos[1023-ypos[np.where(abs(np.diff(xpos))>1)[0][0]+1:]]
    if len(np.where(abs(np.diff(xpos))>1)[0]) > 0:
        xpos[np.where(abs(np.diff(xpos))>1)[0][0]+1:] = xpos[1023-ypos[np.where(abs(np.diff(xpos))>1)[0][0]+1:]]

    return ypos,xpos

#--save and load objects
def save_obj(obj,name,path='' ):
    with open(path+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,path='' ):
    with open(path+name + '.pkl', 'rb') as f:
        return pickle.load(f)

def find_nearest_value(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

#--other
def sort_lists(Y,X):
    # sort first list (Y) based on values of second list (X)
    """
    From: https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    I have a list of strings like this:

    Y = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    X = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
    What is the shortest way of sorting Y using values from X to get the following output?

    ["a", "d", "h", "b", "c", "e", "i", "f", "g"]
    """
    # answer:
    return np.sort(X),np.array([x for _,x in sorted(zip(X,Y))])
