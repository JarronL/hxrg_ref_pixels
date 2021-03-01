# jedi 0.14.1 tab completion fails with IPython/Jupyter 
# Will supposedly be fixed in 0.14.2
# For now, use default IPython tab completion
try:
    import jedi
    if jedi.__version__ == '0.14.1':
        %config Completer.use_jedi = False
except:
    pass

from astropy.io import fits
import ref_pixels
from ref_pixels import robust

# Read in super bias image
fbias = '/Users/jarron/NIRCam/dark_analysis/CV3/SUPER_BIAS/SUPER_BIAS_485.FITS'
hdul = fits.open(fbias)
#superbias = hdul[1].data[::-1,::-1].astype('float') # flip from sci to det coords
superbias = hdul[0].data.astype('float')
hdul.close()

# Read in a CV3 dark data
f = '/Users/jarron/NIRCam/Data/CV3_Darks/485/NRCNRCALONG-DARK-53560804151_1_485_SE_2015-12-22T11h59m47.fits'
hdul = fits.open(f)
data = hdul[0].data.astype('float')
hdul.close()

# Perform super bias subtraction, then reference pixel correction
data -= superbias
kwargs_ref = {
    'nchans': 4, 'in_place': True, 'altcol': True, 'perint': False,
    'fixcol': True, 'avg_type': 'pixel', 'savgol': True
}
data = ref_pixels.reffix_hxrg(data, **kwargs_ref)

# Plot an image
im = data[0]
ind_nan = np.isnan(im)
mn = np.median(im[~ind_nan])
std = robust.medabsdev(im[~ind_nan])
vmin = mn-5*std
vmax = mn+5*std

plt.imshow(im, vmin=vmin, vmax=vmax)