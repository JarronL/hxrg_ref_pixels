# HAWAII-xRG Reference Pixel Corrections

## and Detector Timing Code
--------------------- 

*Authors*: Jarron Leisenring (University of Arizona, Steward Observatory)

This package provides functions for reference pixel correction of HAWAII-1/2/4RG (HxRG) imaging data. The HxRG infrared imaging arrays contain a four-pixel-wide border of embedded reference pixels that surround the bulk of the light-sensitive pixels. These reference pixels are not bonded to the light-sensitive HgCdTe material, but instead hooked to a constant reference capacitance. This setup allows reference pixels to track electronic bias drifts that also occur within the science data. Because they physically sit next to the active pixel, reference pixel data are embedded within the standard pixel clocking procedures as well as the output data streams. Therefore, reference data get subjected to the same amplification and digitizations schemes as the regular science data and get saved as part of the output image.

Reference pixels can be used to track slowly changing offsets in science data measurements, which are caused by internal control voltages that drift over time. This includes frame-to-frame bias shifts, offsets between amplifier readout channels, even/odd column offsets, and 1/f noise imprinted on the imaging data.

While initially developed for JWST NIRCam's detectors and associated readout modes (Slow Mode with 4 or 1 channel outputs), there is built-in flexibility to accommodate a number of HxRG readout schemes. This includes H1RG, H2RG, and H4RGs operating in Slow or Fast Mode with 1, 2, 4, 16, or 32 channel outputs. 

## Installing from source
--------------------- 

To get the most up to date version of ``ref_pixels``, install directly from source. The [development version](https://github.com/JarronL/pynrc>) can be found on GitHub.

In this case, you will need to clone the git repository:
```
$ git clone https://github.com/JarronL/hxrg_ref_pixels
```
Then install the package with:
```
$ cd ref_pixels
$ pip install .
```
For development purposes, you can use editable installations:
```
$ cd ref_pixels
$ pip install -e .
```
This is useful for helping to develop the code, creating bug reports, and pull requests to GitHub. Since the code then lives in your local GitHub directory, this method makes it simple to pull the latest updates straight from the GitHub repo, which are then immediately available in your python installation without needing to reinstall.