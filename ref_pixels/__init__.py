import os
import sys
# from warnings import warn
import astropy
from astropy import config as _config

try:
    from .version import __version__
except ImportError:
    __version__ = ''


class Conf(_config.ConfigNamespace):

    logging_level = _config.ConfigItem(
        ['INFO', 'DEBUG', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'NONE'],
        'Desired logging level for webbpsf_ext.'
    )
    default_logging_level = _config.ConfigItem('INFO', 
        'Logging verbosity: one of {DEBUG, INFO, WARN, ERROR, or CRITICAL}')
    logging_filename = _config.ConfigItem("none", "Desired filename to save log messages to.")
    logging_format_screen = _config.ConfigItem(
        '[%(name)10s:%(levelname)s] %(message)s', 'Format for lines logged to the screen.'
    )
    logging_format_file = _config.ConfigItem(
        '%(asctime)s [%(name)s:%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
        'Format for lines logged to a file.'
    )

conf = Conf()

from .logging_utils import setup_logging#, restart_logging
setup_logging(conf.default_logging_level, verbose=False)

from .ref_pixels import reffix_hxrg, chrem_med, channel_averaging, channel_smooth_savgol