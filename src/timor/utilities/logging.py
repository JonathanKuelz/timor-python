import logging as root_logging
import os
import yaml

from timor.utilities import file_locations

"""
Drop in replacement for the basic logging functionality provided by logging.

Replaces the root logger by an Timor module logger.
"""


CRITICAL = root_logging.CRITICAL
ERROR = root_logging.ERROR
WARNING = root_logging.WARNING
INFO = root_logging.INFO
DEBUG = root_logging.DEBUG
NOTSET = root_logging.NOTSET
settings = file_locations.head.joinpath('.log_conf')


def setLevel(level):
    """Set Timor logger level."""
    root_logging.getLogger('Timor').setLevel(level)


def basicConfig(filename: str = file_locations.default_log,
                filemode: str = 'w',
                output_format: str = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                datefmt: str = '%Y-%m-%d %H:%M:%S',
                level: int = INFO):
    """
    Set logging configuration for Timor

    :param filename: Name to store logs to
    :param filemode: Mode (append ('a'), overwrite ('w'), ...)
    :param output_format: Format string for log
    :param datefmt: Specify time in logs
    :param level: Logging level for Timor logger (use logging.DEBUG, WARN, ...)
    :return:
    """
    try:
        handler = root_logging.FileHandler(filename=filename, mode=filemode)
        handler.setFormatter(root_logging.Formatter(fmt=output_format, datefmt=datefmt))
        root_logging.getLogger('Timor').addHandler(handler)
        setLevel(level)
    except PermissionError as e:
        print(f"There is probably a old default log created with docker in {filename}")
        raise e


def getEffectiveLevel():
    """Return the level the Timor logger is set at."""
    return root_logging.getLogger('Timor').getEffectiveLevel()


def flush():
    """Tell logger to empty output buffers."""
    for h in root_logging.getLogger('Timor').handlers:
        h.flush()


def critical(msg, *args, **kwargs):
    """
    Log a message with severity 'CRITICAL' on the Timor logger.
    """
    root_logging.getLogger('Timor').critical(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """
    Log a message with severity 'ERROR' on the Timor logger.
    """
    root_logging.getLogger('Timor').error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """
    Log a message with severity 'WARNING' on the Timor logger.
    """
    root_logging.getLogger('Timor').warning(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    """
    Deprecated name for warning.
    """
    root_logging.getLogger('Timor').warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the Timor logger.
    """
    root_logging.getLogger('Timor').info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the Timor logger.
    """
    root_logging.getLogger('Timor').debug(msg, *args, **kwargs)


"""
If there is a settings file (can be empty), basicConfig will be called.
It can be customized by providing arguments as yaml dict.
"""
if settings.exists() and (os.getuid() != 0):  # never automatically write logs as root
    with settings.open('r') as f:
        config = yaml.safe_load(f)
    basicConfig(**config)
