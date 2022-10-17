import logging as root_logging
import os
from pathlib import Path
from typing import Union

from timor.utilities.configurations import TIMOR_CONFIG

"""
Drop in replacement for the basic logging functionality provided by logging.

Replaces the root logger by a custom configured logger.
"""


CRITICAL = root_logging.CRITICAL
ERROR = root_logging.ERROR
WARNING = root_logging.WARNING
INFO = root_logging.INFO
DEBUG = root_logging.DEBUG
NOTSET = root_logging.NOTSET


def getLogger() -> root_logging.Logger:
    """Get the custom logger."""
    return root_logging.getLogger('Timor')


def setLevel(level):
    """Set the logger level."""
    getLogger().setLevel(level)


def basicConfig(filename: str = None,
                filemode: str = 'w',
                output_format: str = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt: str = '%Y-%m-%d %H:%M:%S',
                level: Union[int, str] = INFO):
    """
    Set logging configurations for the runtime environment

    :param filename: Name to store logs to
    :param filemode: Mode (append ('a'), overwrite ('w'), ...)
    :param output_format: Format string for log
    :param datefmt: Specify time in logs
    :param level: Logging level for the logger (use logging.DEBUG, WARN, ...)
    """
    if isinstance(level, str):
        try:
            level = int(level)
        except ValueError:
            level = root_logging.getLevelName(level)
    if filename is not None:
        try:
            Path(filename).parent.mkdir(exist_ok=True)
            handler = root_logging.FileHandler(filename=filename, mode=filemode)
        except PermissionError as e:
            print(f"There is probably a old default log created with docker in {filename}")
            raise e
    else:
        handler = root_logging.StreamHandler()
    handler.setFormatter(root_logging.Formatter(fmt=output_format, datefmt=datefmt))
    getLogger().addHandler(handler)
    setLevel(level)


def getEffectiveLevel():
    """Return the level the logger is set at."""
    return getLogger().getEffectiveLevel()


def flush():
    """Tell logger to empty output buffers."""
    for h in getLogger().handlers:
        h.flush()


def critical(msg, *args, **kwargs):
    """
    Log a message with severity 'CRITICAL'.
    """
    getLogger().critical(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR'.
    """
    getLogger().error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING'.
    """
    getLogger().warning(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    """
    Deprecated name for warning.
    """
    getLogger().warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO'.
    """
    getLogger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG'.
    """
    getLogger().debug(msg, *args, **kwargs)


"""
If there is a settings file (can be empty), basicConfig will be called.
It can be customized by providing arguments as yaml dict.
"""
if TIMOR_CONFIG.has_section('LOGGING') and (os.getuid() != 0):  # never automatically write logs as root
    basicConfig(**TIMOR_CONFIG['LOGGING'])
else:
    basicConfig()
