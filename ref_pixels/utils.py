import numpy as np
from tqdm.auto import trange

from .fast_poly import jl_poly_fit, jl_poly

def var_ex_model(ng, nf, params):
    """ Variance Excess Model

    Measured pixel variance shows a slight excess above the measured values.
    The input `params` describes this excess variance. This function can be 
    used to fit the excess variance for a variety of different readout patterns.
    """
    return 12. * (ng - 1.)/(ng + 1.) * params[0]**2 - params[1] / nf**0.5

def pix_noise(ngroup=2, nf=1, nd2=0, tf=10.73677, rn=15.0, ktc=29.0, p_excess=(0,0),
    fsrc=0.0, idark=0.003, fzodi=0, fbg=0, ideal_Poisson=False,
    ff_noise=False, **kwargs):
    """Noise per pixel

    Theoretical noise calculation of a generalized MULTIACCUM ramp in terms of e-/sec.
    Includes flat field errors from JWST-CALC-003894.

    Parameters
    ----------
    ngroup : int
        Number of groups in integration ramp
    nf : int
        Number of frames in each group
    nd2 : int
        Number of dropped frames in each group
    tf : float
        Frame time
    rn : float
        Read Noise per pixel (e-).
    ktc : float
        kTC noise (in e-). Only valid for single frame (n=1)
    p_excess : array-like
        An array or list of two elements that holds the parameters
        describing the excess variance observed in effective noise plots.
        By default these are both 0. For NIRCam detectors, recommended
        values are [1.0,5.0] for SW and [1.5,10.0] for LW.
    fsrc : float
        Flux of source in e-/sec/pix.
    idark : float
        Dark current in e-/sec/pix.
    fzodi : float
        Zodiacal light emission in e-/sec/pix.
    fbg : float
        Any additional background (telescope emission or scattered light?)
    ideal_Poisson : bool
        If set to True, use total signal for noise estimate,
        otherwise MULTIACCUM equation is used.
    ff_noise : bool
        Include flat field errors in calculation? From JWST-CALC-003894.
        Default=False.

    Notes
    -----
    Various parameters can either be single values or numpy arrays.
    If multiple inputs are arrays, make sure their array sizes match.
    Variables that need to have the same array shapes (or a single value):

        - n, m, s, & tf
        - rn, idark, ktc, fsrc, fzodi, & fbg

    Array broadcasting also works.

    Example
    -------

    >>> n = np.arange(50)+1  # An array of different ngroups to test out

    >>> # Create 2D Gaussian PSF with FWHM = 3 pix
    >>> npix = 20  # Number of pixels in x and y direction
    >>> fwhm = 3.0
    >>> x = np.arange(0, npix, 1, dtype=float)
    >>> y = x[:,np.newaxis]
    >>> x0 = y0 = npix // 2  # Center position
    >>> fsrc = np.exp(-4*np.log(2.) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    >>> fsrc /= fsrc.max()
    >>> fsrc *= 10  # 10 counts/sec in peak pixel
    >>> fsrc = fsrc.reshape(npix,npix,1)  # Necessary for broadcasting

    >>> # Represents pixel array w/ slightly different RN/pix
    >>> rn = 15 + np.random.normal(loc=0, scale=0.5, size=(1,npix,npix))
    >>> # Results is a 50x(20x20) showing the noise in e-/sec/pix at each group
    >>> noise = pix_noise(ngroup=n, rn=rn, fsrc=fsrc)
    """

    # Convert everything to arrays
    n = np.array(ngroup)
    m = np.array(nf)
    s = np.array(nd2)
    tf = np.array(tf)

    # Total flux (e-/sec/pix)
    ftot = fsrc + idark + fzodi + fbg

    # Special case if n=1
    # To be inserted at the end
    if (n==1).any():
        # Variance after averaging m frames
        var = ktc**2 + (rn**2 + ftot*tf) / m
        noise = np.sqrt(var)
        noise /= tf # In terms of e-/sec

        if (n==1).all(): return noise
        noise_n1 = noise

    ind_n1 = (n==1)
    temp = np.array(rn+ktc+ftot)
    temp_bool = np.zeros(temp.shape, dtype=bool)
    ind_n1_all = (temp_bool | ind_n1)

    # Group time
    tg = tf * (m + s)
    # Effective integration time
    tint = tg * (n - 1)

    # Read noise, group time, and frame time variances
    # This is the MULTIACCUM eq from Rauscher et al. (2007).
    # This equation assumes that the slope-fitting routine uses
    # incorrect covariance matrix that doesn't take into account
    # the correlated Poisson noise up the ramp.
    var_rn = rn**2       * 12.               * (n - 1.) / (m * n * (n + 1.))
    var_gp = ftot * tint * 6. * (n**2. + 1.) / (5 * n * (n + 1.))
    var_fm = ftot   * tf * 2. * (m**2. - 1.) * (n - 1.) / (m * n * (n + 1.))

    # Functional form for excess variance above theoretical
    # Empirically measured formulation
    # var_ex = 12. * (n - 1.)/(n + 1.) * p_excess[0]**2 - p_excess[1] / m**0.5
    var_ex = var_ex_model(n, m, p_excess)

    # Variance of total signal
    var_poisson = (ftot * tint) if ideal_Poisson else (var_gp - var_fm)

    # Total variance
    var = var_rn + var_poisson + var_ex
    sig = np.sqrt(var)

    # Noise in e-/sec
    noise = sig / tint
    # Make sure to copy over ngroup=1 cases
    if (n==1).any():
        noise[ind_n1_all] = noise_n1[ind_n1_all]
    #print(ind_n1_all.shape,noise.shape,noise_n1.shape)

    # Include flat field noise
    # JWST-CALC-003894
    if ff_noise:
        noise_ff = 1E-4 # Uncertainty in the flat field
        factor = 1 + noise_ff*np.sqrt(ftot)
        noise *= factor

    return noise


#######################################
# Linearity and Gain
#######################################

# Determine saturation level in ADU (relative to bias)
def find_sat(data, bias=None, ref_info=[4,4,4,4], bit_depth=16):
    """
    Given a data cube, find the values in ADU in which data
    reaches hard saturation.
    """

    # Maximum possible value corresponds to bit depth
    ref_vals = 2**bit_depth-1
    # sat_max can be larger than bit depth if linearity or gain correction has occurred
    sat_max = np.max([2**bit_depth-1, data.max()])
    sat_min = 0

    # Subtract bias?
    nz, ny, nx = data.shape
    imarr = data if bias is None else data - bias

    # Data can be characterized as large differences at start,
    # followed by decline and then difference of 0 at hard saturation

    # Determine difference between samples
    diff_arr = imarr[1:] - imarr[0:-1]

    # Select pixels to determine individual saturation values
    diff_max = np.median(diff_arr[0]) / 10
    diff_min = 100

    # Ensure a high rate at the beginning and a flat rate at the end
    sat_mask = (diff_arr[0]>diff_max) & (np.abs(diff_arr[-1]) < diff_min)

    # Median value to use for pixels that didn't reach saturation
    # sat_med = np.median(imarr[-1, sat_mask])
    
    # Initialize saturation array with median
    # sat_arr = np.ones([ny,nx]) * sat_med

    # Initialize saturation as max-min
    sat_arr = imarr[-1] - imarr[0]
    sat_arr[sat_mask] = imarr[-1, sat_mask]

    # Bound between 0 and bit depth
    sat_arr[sat_arr<sat_min] = sat_min
    sat_arr[sat_arr>sat_max] = sat_max

    # Reference pixels don't saturate
    # [bottom, upper, left, right]
    br, ur, lr, rr = ref_info
    ref_mask = np.zeros([ny,nx], dtype=bool)
    if br>0: ref_mask[0:br,:] = True
    if ur>0: ref_mask[-ur:,:] = True
    if lr>0: ref_mask[:,0:lr] = True
    if rr>0: ref_mask[:,-rr:] = True
    sat_arr[ref_mask] = ref_vals
    
    return sat_arr

# Fit unsaturated data and return coefficients
def cube_fit(tarr, data, bias=None, sat_vals=None, sat_frac=0.95, 
             deg=1, fit_zero=False, verbose=False, ref_info=[4,4,4,4],
             use_legendre=False, lxmap=None, return_lxmap=False, **kwargs):
        
    nz, ny, nx = data.shape
    
    # Subtract bias?
    imarr = data if bias is None else data - bias
    
    # Get saturation levels
    if sat_vals is None:
        sat_vals = find_sat(imarr, ref_info=ref_info, **kwargs)
        
    # Array of masked pixels (saturated)
    mask_good = imarr < sat_frac*sat_vals
    
    # Reshape for all pixels in single dimension
    imarr = imarr.reshape([nz, -1])
    mask_good = mask_good.reshape([nz, -1])

    # Initial 
    cf = np.zeros([deg+1, nx*ny])
    if return_lxmap:
        lx_min = np.zeros([nx*ny])
        lx_max = np.zeros([nx*ny])

    # For each ramp size
    npix_sum = 0
    i0 = 0 if fit_zero else 1
    for i in np.arange(i0,nz)[::-1]:
        ind = (cf[1] == 0) & (mask_good[i])
        npix = np.sum(ind)
        npix_sum += npix
        
        if verbose:
            print(i+1,npix,npix_sum, 'Remaining: {}'.format(nx*ny-npix_sum))
            
        if npix>0:
            if fit_zero:
                x = np.concatenate(([0], tarr[0:i+1]))
                y = np.concatenate((np.zeros([1, np.sum(ind)]), imarr[0:i+1,ind]), axis=0)
            else:
                x, y = (tarr[0:i+1], imarr[0:i+1,ind])

            if return_lxmap:
                lx_min[ind] = np.min(x) if lxmap is None else lxmap[0]
                lx_max[ind] = np.max(x) if lxmap is None else lxmap[1]
                
            # Fit line if too few points relative to polynomial degree
            if len(x) <= deg+1:
                cf[0:2,ind] = jl_poly_fit(x,y, deg=1, use_legendre=use_legendre, lxmap=lxmap)
            else:
                cf[:,ind] = jl_poly_fit(x,y, deg=deg, use_legendre=use_legendre, lxmap=lxmap)

    imarr = imarr.reshape([nz,ny,nx])
    mask_good = mask_good.reshape([nz,ny,nx])
    
    cf = cf.reshape([deg+1,ny,nx])
    if return_lxmap:
        lxmap_arr = np.array([lx_min, lx_max]).reshape([2,ny,nx])
        return cf, lxmap_arr
    else:
        return cf


def hist_indices(values, bins=10, return_more=False):
    """Histogram indices
    
    This function bins an input of values and returns the indices for
    each bin. This is similar to the reverse indices functionality
    of the IDL histogram routine. It's also much faster than doing
    a for loop and creating masks/indices at each iteration, because
    we utilize a sparse matrix constructor. 
    
    Returns a list of indices grouped together according to the bin.
    Only works for evenly spaced bins.
    
    Parameters
    ----------
    values : ndarray
        Input numpy array. Should be a single dimension.
    bins : int or ndarray
        If bins is an int, it defines the number of equal-width bins 
        in the given range (10, by default). If bins is a sequence, 
        it defines the bin edges, including the rightmost edge.
        In the latter case, the bins must encompass all values.
    return_more : bool
        Option to also return the values organized by bin and 
        the value of the centers (igroups, vgroups, center_vals).
    
    Example
    -------
    Find the standard deviation at each radius of an image
    
        >>> rho = dist_image(image)
        >>> binsize = 1
        >>> bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        >>> igroups, vgroups, center_vals = hist_indices(rho, bins, True)
        >>> # Get the standard deviation of each bin in image
        >>> std = binned_statistic(igroups, image, func=np.std)

    """
    
    from scipy.sparse import csr_matrix
    
    values_flat = values.ravel()

    vmin = values_flat.min()
    vmax = values_flat.max()
    N  = len(values_flat)   
    
    try: # if bins is an integer
        binsize = (vmax - vmin) / bins
        bins = np.arange(vmin, vmax + binsize, binsize)
        bins[0] = vmin
        bins[-1] = vmax
    except: # otherwise assume it's already an array
        binsize = bins[1] - bins[0]
    
    # Central value of each bin
    center_vals = bins[:-1] + binsize / 2.
    nbins = center_vals.size

    # TODO: If input bins is an array that doesn't span the full set of input values,
    # then we need to set a warning.
    if (vmin<bins[0]) or (vmax>bins[-1]):
        raise ValueError("Bins must encompass entire set of input values.")
    digitized = ((nbins-1.0) / (vmax-vmin) * (values_flat-vmin)).astype(np.int)
    csr = csr_matrix((values_flat, [digitized, np.arange(N)]), shape=(nbins, N))

    # Split indices into their bin groups    
    igroups = np.split(csr.indices, csr.indptr[1:-1])
    
    if return_more:
        vgroups = np.split(csr.data, csr.indptr[1:-1])
        return (igroups, vgroups, center_vals)
    else:
        return igroups
    

#######################################
# Miscellaneous
#######################################

def _check_list(value, temp_list, var_name=None):
    """
    Helper function to test if a value exists within a list. 
    If not, then raise ValueError exception.
    This is mainly used for limiting the allowed values of some variable.
    """
    if value not in temp_list:
        # Replace None value with string for printing
        if None in temp_list: 
            temp_list[temp_list.index(None)] = 'None'
        # Make sure all elements are strings
        temp_list2 = [str(val) for val in temp_list]
        var_name = '' if var_name is None else var_name + ' '
        err_str = "Invalid {}setting: {} \n\tValid values are: {}" \
                         .format(var_name, value, ', '.join(temp_list2))
        raise ValueError(err_str)

def tuples_to_dict(pairs, verbose=False):
    """
    Take a list of paired tuples and convert to a dictionary
    where the first element of each tuple is the key and the 
    second element is the value.
    """
    d={}
    for (k, v) in pairs:
        d[k] = v
        if verbose:
            if isinstance(v,float): print("{:<10} {:>10.4f}".format(k, v))
            else: print("{:<10} {:>10}".format(k, v))
    return d