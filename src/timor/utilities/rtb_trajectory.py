from collections import namedtuple
import math
from typing import Union

import numpy as np

"""
This code is taken from the robotics toolbox v1.1.0
:link: https://github.com/petercorke/robotics-toolbox-python
"""


_scalartypes = (int, np.integer, float, np.floating)
ArrayLike = Union[list, np.ndarray, tuple, set]


class RtbTrajectory:
    """
    A container class for trajectory data. Reduced version of the original, without plotting functionalities.
    """

    def __init__(self, name, t, s, sd=None, sdd=None, istime=False):
        """
        Construct a new trajectory instance

        :param name: name of the function that created the trajectory
        :type name: str
        :param t: independent variable, eg. time or step
        :type t: ndarray(m)
        :param s: position
        :type s: ndarray(m) or ndarray(m,n)
        :param sd: velocity
        :type sd: ndarray(m) or ndarray(m,n)
        :param sdd: acceleration
        :type sdd: ndarray(m) or ndarray(m,n)
        :param istime: ``t`` is time, otherwise step number
        :type istime: bool
        :param tblend: blend duration (``trapezoidal`` only)
        :type istime: float

        The object has attributes:

        - ``t``  the independent variable
        - ``s``  the position
        - ``sd``  the velocity
        - ``sdd``  the acceleration

        If ``t`` is time, ie. ``istime`` is True, then the units of ``sd`` and
        ``sdd`` are :math:`s^{-1}` and :math:`s^{-2}` respectively, otherwise
        with respect to ``t``.

        .. note:: Data is stored with timesteps as rows and axes as columns.
        """
        self.name = name
        self.t = t
        self.s = s
        self.sd = sd
        self.sdd = sdd
        self.istime = istime

    def __str__(self):
        """Print representation of the trajectory"""
        s = f"Trajectory created by {self.name}: {len(self)} time steps x {self.naxes} axes"
        return s

    def __repr__(self):
        """String representation of the trajectory"""
        return str(self)

    def __len__(self):
        """
        Length of trajectory

        :return: number of steps in the trajectory
        :rtype: int
        """
        return self.s.shape[0]

    @property
    def q(self):
        """
        Position trajectory

        :return: trajectory with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.s``, for compatibility with other
            applications.
        """
        return self.s

    @property
    def qd(self):
        """
        Velocity trajectory

        :return: trajectory velocity with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.sd``, for compatibility with other
            applications.
        """
        return self.sd

    @property
    def qdd(self):
        """
        Acceleration trajectory

        :return: trajectory acceleration with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.sdd``, for compatibility with other
            applications.
        """
        return self.sdd

    @property
    def naxes(self):
        """
        Number of axes in the trajectory

        :return: number of axes or dimensions
        :rtype: int
        """
        if self.s.ndim == 1:
            return 1
        else:
            return self.s.shape[1]


def mstraj(
    viapoints,
    dt,
    tacc,
    qdmax=None,
    tsegment=None,
    q0=None,
    qd0=None,
    qdf=None,
    verbose=False):
    """
    Multi-segment multi-axis trajectory

    :param viapoints: A set of viapoints, one per row
    :type viapoints: ndarray(m,n)
    :param dt: time step
    :type dt: float (seconds)
    :param tacc: acceleration time (seconds)
    :type tacc: float
    :param qdmax: maximum speed, defaults to None
    :type qdmax: array_like(n) or float, optional
    :param tsegment: maximum time of each motion segment (seconds), defaults
        to None
    :type tsegment: array_like, optional
    :param q0: initial coordinates, defaults to first row of viapoints
    :type q0: array_like(n), optional
    :param qd0: inital  velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qdf: final  velocity, defaults to zero
    :type qdf: array_like(n), optional
    :param verbose: print debug information, defaults to False
    :type verbose: bool, optional
    :return: trajectory
    :rtype: Trajectory instance

    Computes a trajectory for N axes moving smoothly through a set of
    viapoints.
    The motion comprises M segments:

    - The initial coordinates are the first row of ``viapoints`` or ``q0`` if
      provided.
    - The final coordinates are the last row of ``viapoints``
    - Each segment is linear motion and polynomial blends connect the
      viapoints.
    - All joints arrive at each via point at the same time, ie. the motion is
      coordinated across axes

    The time of the segments can be specified in two different ways:

    #. In terms of segment time where ``tsegment`` is an array of segment times
       which is the number of via points minus one::

            ``traj = mstraj(viapoints, dt, tacc, tsegment=TS)``

    #. Governed by the speed of the slowest axis for the segment.  The axis
       speed is a scalar (all axes have the same speed) or an N-vector of speed
       per axis::

            traj = mstraj(viapoints, dt, tacc, qdmax=SPEED)

    The return value is a namedtuple (named ``mstraj``) with elements:

        - ``t``  the time coordinate as a numpy ndarray, shape=(K,)
        - ``q``  the axis values as a numpy ndarray, shape=(K,N)
        - ``arrive`` a list of arrival times for each segment
        - ``info`` a list of named tuples, one per segment that describe the
          slowest axis, segment time,  and time stamp
        - ``via`` the passed set of via points

    The  trajectory proper is (``traj.t``, ``traj.q``).  The trajectory is a
    matrix has one row per time step, and one column per axis.

    Notes:

        - Only one of ``qdmag`` or ``tsegment`` can be specified
        - If ``tacc`` is greater than zero then the path smoothly accelerates
        between segments using a polynomial blend.  This means that the the via
        point is not actually reached.
        - The path length K is a function of the number of via
        points and the time or velocity limits that apply.
        - Can be used to create joint space trajectories where each axis is a
        joint coordinate.
        - Can be used to create Cartesian trajectories where the "axes"
        correspond to translation and orientation in RPY or Euler angle form.
        - If ``qdmax`` is a scalar then all axes are assumed to have the same
        maximum speed.
        - ``tg`` has extra attributes ``arrive``, ``info`` and ``via``

    :seealso: :func:`trapezoidal`, :func:`ctraj`, :func:`mtraj`
    """
    if q0 is None:
        q0 = viapoints[0, :]
        viapoints = viapoints[1:, :]
    else:
        q0 = getvector(q0)
        if not viapoints.shape[1] == len(q0):
            raise ValueError("WP and Q0 must have same number of columns")

    ns, nj = viapoints.shape
    Tacc = tacc

    if qdmax is not None and tsegment is not None:
        raise ValueError("cannot specify both qdmax and tsegment")

    if qdmax is None:
        if tsegment is None:
            raise ValueError("tsegment must be given if qdmax is not")

        if not len(tsegment) == ns:
            raise ValueError("Length of TSEG does not match number of viapoints")

    if tsegment is None:

        # This is unreachable, left just in case
        if qdmax is None:  # pragma nocover
            raise ValueError("qdmax must be given if tsegment is not")

        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax = np.tile(qdmax, (nj,))
        else:
            qdmax = getvector(qdmax)

            if not len(qdmax) == nj:
                raise ValueError("Length of QDMAX does not match number of axes")

    if isinstance(Tacc, (int, float)):
        Tacc = np.tile(Tacc, (ns,))
    else:
        if not len(Tacc) == ns:
            raise ValueError("Tacc is wrong size")
    if qd0 is None:
        qd0 = np.zeros((nj,))
    else:
        if not len(qd0) == len(q0):
            raise ValueError("qd0 is wrong size")
    if qdf is None:
        qdf = np.zeros((nj,))
    else:
        if not len(qdf) == len(q0):
            raise ValueError("qdf is wrong size")

    # set the initial conditions
    q_prev = q0
    qd_prev = qd0

    clock = 0  # keep track of time
    arrive = np.zeros((ns,))  # record planned time of arrival at via points
    tg = np.zeros((0, nj))
    infolist = []
    info = namedtuple("mstraj_info", "slowest segtime clock")

    def mrange(start, stop, step):
        """
        mrange(start, stop, step) behaves like MATLAB start:step:stop and includes the final value unlike range()
        """
        # ret = []
        istart = round(start / step)
        istop = round(stop / step)
        return np.arange(istart, istop + 1) * step

    for seg in range(0, ns):
        q_next = viapoints[seg, :]  # current target

        if verbose:  # pragma nocover
            print(f"------- segment {seg}: {q_prev} --> {q_next}")

        # set the blend time, just half an interval for the first segment

        tacc = Tacc[seg]

        tacc = math.ceil(tacc / dt) * dt
        tacc2 = math.ceil(tacc / 2 / dt) * dt
        if seg == 0:
            taccx = tacc2
        else:
            taccx = tacc

        # estimate travel time
        #    could better estimate distance travelled during the blend
        dq = q_next - q_prev  # total distance to move this segment

        if qdmax is not None:
            # qdmax is specified, compute slowest axis

            # qb = taccx * qdmax / 2       # distance moved during blend
            tb = taccx

            # convert to time
            tl = abs(dq) / qdmax
            # tl = abs(dq - qb) / qdmax
            tl = np.ceil(tl / dt) * dt

            # find the total time and slowest axis
            tt = tb + tl
            slowest = np.argmax(tt)
            tseg = tt[slowest]

            # best if there is some linear motion component
            if tseg <= 2 * tacc:
                tseg = 2 * tacc

        elif tsegment is not None:
            # segment time specified, use that
            tseg = tsegment[seg]
            slowest = math.nan

        infolist.append(info(slowest, tseg, clock))

        # log the planned arrival time
        arrive[seg] = clock + tseg
        if seg > 0:
            arrive[seg] += tacc2

        if verbose:  # pragma nocover
            print(
                f"seg {seg}, distance {dq}, "
                "slowest axis {slowest}, time required {tseg}"
            )

        # create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        qb = jtraj(q0, q_prev + tacc2 * qd, mrange(0, taccx, dt), qd0=qd_prev, qd1=qd).s
        if verbose:  # pragma nocover
            print(qb)
        tg = np.vstack([tg, qb[1:, :]])

        clock = clock + taccx  # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in mrange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0 = (1 - s) * q_prev + s * q_next  # linear step
            if verbose:  # pragma nocover
                print(t, s, q0)
            tg = np.vstack([tg, q0])
            clock += dt

        q_prev = q_next  # next target becomes previous target
        qd_prev = qd

    # add the final blend
    qb = jtraj(q0, q_next, mrange(0, tacc2, dt), qd0=qd_prev, qd1=qdf).s
    tg = np.vstack([tg, qb[1:, :]])

    infolist.append(info(None, tseg, clock))

    traj = RtbTrajectory("mstraj", dt * np.arange(0, tg.shape[0]), tg)
    traj.arrive = arrive
    traj.info = infolist
    traj.via = viapoints

    return traj


def jtraj(q0, qf, t, qd0=None, qd1=None):
    """
    Compute a joint-space trajectory

    :param q0: initial joint coordinate
    :type q0: array_like(n)
    :param qf: final joint coordinate
    :type qf: array_like(n)
    :param t: time vector or number of steps
    :type t: array_like or int
    :param qd0: initial velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qd1: final velocity, defaults to zero
    :type qd1: array_like(n), optional
    :return: trajectory
    :rtype: Trajectory instance

    - ``tg = jtraj(q0, qf, N)`` is a joint space trajectory where the joint
      coordinates vary from ``q0`` (M) to ``qf`` (M).  A quintic (5th order)
      polynomial is used with default zero boundary conditions for velocity and
      acceleration.  Time is assumed to vary from 0 to 1 in ``N`` steps.

    - ``tg = jtraj(q0, qf, t)`` as above but ``t`` is a uniformly-spaced time
      vector

    The return value is an object that contains position, velocity and
    acceleration data.

    Notes:

    - The time vector, if given, scales the velocity and acceleration outputs
      assuming that the time vector starts at zero and increases
      linearly.

    :seealso: :func:`ctraj`, :func:`qplot`, :func:`~SerialLink.jtraj`
    """
    # print(f"  --- jtraj: {q0} --> {q1} in {tv}")
    if isinstance(t, int):
        tscal = 1.0
        ts = np.linspace(0, 1, t)  # normalized time from 0 -> 1
        tv = ts * t
    else:
        tv = t.flatten()
        tscal = max(t)
        ts = t.flatten() / tscal

    q0 = getvector(q0)
    qf = getvector(qf)

    if not len(q0) == len(qf):
        raise ValueError("q0 and q1 must be same size")

    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        if not len(qd0) == len(q0):
            raise ValueError("qd0 has wrong size")
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd1 = getvector(qd1)
        if not len(qd1) == len(q0):
            raise ValueError("qd1 has wrong size")

    # compute the polynomial coefficients
    A = 6 * (qf - q0) - 3 * (qd1 + qd0) * tscal
    B = -15 * (qf - q0) + (8 * qd0 + 7 * qd1) * tscal
    C = 10 * (qf - q0) - (6 * qd0 + 4 * qd1) * tscal
    E = qd0 * tscal  # as the t vector has been normalized
    F = q0

    # n = len(q0)

    tt = np.array([ts**5, ts**4, ts**3, ts**2, ts, np.ones(ts.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])  # 6xN

    qt = tt @ coeffs

    # compute  velocity
    coeffs = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    qdt = tt @ coeffs / tscal

    # compute  acceleration
    coeffs = np.array(
        [np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)]
    )
    qddt = tt @ coeffs / tscal**2

    return RtbTrajectory("jtraj", tv, qt, qdt, qddt, istime=True)


def getvector(v, dim=None, out="array", dtype=np.float64) -> ArrayLike:
    """
    Return a vector value

    :param v: passed vector
    :param dim: required dimension, or None if any length is ok
    :type dim: int or None
    :param out: output format, default is 'array'
    :type out: str
    :param dtype: datatype for numPy array return (default np.float64)
    :type dtype: numPy type
    :return: vector value in specified format
    :raises TypeError: value is not a list or NumPy array
    :raises ValueError: incorrect number of elements

    - ``getvector(vec)`` is ``vec`` converted to the output format ``out``
      where ``vec`` is any of:

        - a Python native int or float, a 1-vector
        - Python native list or tuple
        - numPy real 1D array, ie. shape=(N,)
        - numPy real 2D array with a singleton dimension, ie. shape=(1,N)
          or (N,1)

    - ``getvector(vec, N)`` as above but must be an ``N``-element vector.

    The returned vector will be in the format specified by ``out``:

    ==========  ===============================================
    format      return type
    ==========  ===============================================
    'sequence'  Python list, or tuple if a tuple was passed in
    'list'      Python list
    'array'     1D numPy array, shape=(N,)  [default]
    'row'       row vector, a 2D numPy array, shape=(1,N)
    'col'       column vector, 2D numPy array, shape=(N,1)
    ==========  ===============================================

    .. runblock:: pycon

        >>> from spatialmath.base import getvector
        >>> import numpy as np
        >>> getvector([1,2])  # list
        >>> getvector([1,2], out='row')  # list
        >>> getvector([1,2], out='col')  # list
        >>> getvector((1,2))  # tuple
        >>> getvector(np.r_[1,2,3], out='sequence')  # numpy array
        >>> getvector(1)  # scalar
        >>> getvector([1])
        >>> getvector([[1]])

    .. note::
        - For 'array', 'row' or 'col' output the NumPy dtype defaults to the
          ``dtype`` of ``v`` if it is a NumPy array, otherwise it is
          set to the value specified by the ``dtype`` keyword which defaults
          to ``np.float64``.
        - If ``v`` is symbolic the ``dtype`` is retained as ``'O'``

    :seealso: :func:`isvector`
    """
    dt = dtype

    if isinstance(v, _scalartypes):  # handle scalar case
        v = [v]

    if isinstance(v, (list, tuple)):
        # list or tuple was passed in

        if dim is not None and v and len(v) != dim:
            raise ValueError("incorrect vector length")
        if out == "sequence":
            return v
        elif out == "list":
            return list(v)
        elif out == "array":
            return np.array(v, dtype=dt)
        elif out == "row":
            return np.array(v, dtype=dt).reshape(1, -1)
        elif out == "col":
            return np.array(v, dtype=dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")

    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim is not None:
            if not (s == (dim,) or s == (1, dim) or s == (dim, 1)):
                raise ValueError(
                    "incorrect vector length: expected {}, got {}".format(dim, s)
                )

        v = v.flatten()

        if v.dtype.kind == "O":
            dt = "O"

        if out in ("sequence", "list"):
            return list(v.flatten())
        elif out == "array":
            return v.astype(dt)
        elif out == "row":
            return v.astype(dt).reshape(1, -1)
        elif out == "col":
            return v.astype(dt).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise TypeError("invalid input type")
