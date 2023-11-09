#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 07.11.23
from enum import Enum
from typing import Callable

import numpy as np


class CallbackReturn(Enum):
    """
    Holds all possible return values of a callback function. None will be interpreted as TRUE.

    This Enum can be used by iterative algorithms to give feedback to the caller about the status of the algorithm.
    It's values can be interpreted as follows:

        * TRUE: The callback was evaluated without any problems and the algorithm should continue.
        * FALSE: The callback detected a problem (e.g., a constraint violation) that can potentially be resolved in
            future iterations. However, the current state is not considered valid/successful.
        * BREAK: The callback detected a problem that cannot be resolved in future iterations. Interrupt the algorithm.
    """

    TRUE = True
    FALSE = False
    BREAK = 'BREAK'

    @classmethod
    def _missing_(cls, value):
        """Allow some aliases"""
        if isinstance(value, str):
            value = value.lower()
            if value == 'break':
                return cls.BREAK
            elif value == 'true':
                return cls.TRUE
            elif value == 'false':
                return cls.FALSE
        if value is None:
            return cls.TRUE
        return super()._missing_(value)


def chain_callbacks(*callbacks: Callable[[any], CallbackReturn]) -> Callable[[any], CallbackReturn]:
    """
    Chains any number of callback functions together.

    The callbacks will be executed in order until either one returns BREAK or all have been evaluated. The chain method
    casts all return values to a CallbackReturn automatically. Note that no return (None) will be considered as TRUE,
    indicating a callback evaluation without problems. If zero callbacks are chained (i.e., the chain is empty), the
    returned callback will always return TRUE.

    :param callbacks: Any number of callbacks to be chained together. They should all return a CallbackReturn value.
    :return: A callback function that chains all input callbacks together. It takes any number of arguments.
    """
    def chained_callback(*args, **kwargs) -> CallbackReturn:
        """The chained callback function."""
        ret = CallbackReturn.TRUE
        for callback in callbacks:
            cb_ret = CallbackReturn(callback(*args, **kwargs))
            if cb_ret is CallbackReturn.FALSE:
                ret = CallbackReturn.FALSE
            elif cb_ret is CallbackReturn.BREAK:
                return CallbackReturn.FALSE
        return ret

    return chained_callback


IKCallback: type = Callable[[np.ndarray], CallbackReturn]
