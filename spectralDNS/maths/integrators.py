# pylint: disable=inconsistent-return-statements,fixme,unreachable
__author__ = "Mikael Mortensen <mikaem@math.uio.no> and Nathanael Schilling <nathanael.schilling@in.tum.de>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import numpy as np
from mpi4py import MPI
from ..optimization import optimizer, wraps

comm = MPI.COMM_WORLD

__all__ = ['getintegrator']


def adaptiveRK(A, b, bhat, err_order, fY_hat, u0_new, sc, err, fsal, offset,
               aTOL, rTOL, adaptive, errnorm, rhs, u0, solver, dt, tstep,
               context, additional_callback, params, predictivecontroller=False):
    """
    Take a step using any Runge-Kutta method.
    Parameters
    ----------
    A, b, bhat : arrays
        Runge-Kutta coefficients
    err_order : int
        Order of embedded method
    fY_hat, U_tmp, u0_new, sc, err : work arrays
    fsal : boolean
        Whether method is first-same-as-last
    offset : length-1 array of int
        Where to find the previous RHS evaluation (for FSAL methods).  This can probably be eliminated.
    aTOL, rTOL : float
        Error tolerances
    adaptive : boolean
        If true, adapt the step size
    errnorm : str
        Which norm to use in computing the error estimate.  One of {"2", "inf"}.
    rhs : array
        RHS evaluation
    u0 : array
        solution value (returned)
    solver : calling module
        contains method ComputeRHS for computing RHS of evolution equation
    dt : float
        time step size
    tstep : int
        Number of steps taken so far
    predictivecontroller : boolean
        If True use PI controller
    """
    s = A.shape[0]

    # Some parameters for adaptive time-stepping. See p167, Hairer, Norsett and Wanner. "Solving Ordinary Differential Equations 1"
    # for details.
    facmax_default = 2
    facmax = facmax_default
    fac = 0.8
    facmin = 0.01

    # We may need to repeat the time-step until a small enough value is used.
    while True:
        dt_prev = dt
        if fsal:
            offset[0] = (offset[0] - 1) % s
        for i in range(0, s):
            if not fsal or (tstep == 0 or i != 0):
                fY_hat[(i + offset[0]) % s] = u0
                for j in range(0, i):
                    fY_hat[(i + offset[0]) % s] += dt * A[i, j] * fY_hat[(j + offset[0]) % s]
                # Compute F(Y)
                rhs = solver.ComputeRHS(rhs, fY_hat[(i + offset[0]) % s], solver, **context)
                fY_hat[(i + offset[0]) % s] = rhs

            if i == 0:
                context.fu0 = fY_hat[(0 + offset[0]) % s]
                additional_callback(context)

        # Calculate the new value
        u0_new[:] = u0
        u0_new[:] += dt * b[0] * fY_hat[(0 + offset[0]) % s]
        err[:] = dt * (b[0] - bhat[0]) * fY_hat[(0 + offset[0]) % s]

        for j in range(1, s):
            u0_new[:] += dt * b[j] * fY_hat[(j + offset[0]) % s]
            err[:] += dt * (b[j] - bhat[j]) * fY_hat[(j + offset[0]) % s]

        est = 0.0
        sc[:] = aTOL + np.maximum(np.abs(u0), np.abs(u0_new)) * rTOL
        if errnorm == "2":
            est_to_bcast = None
            nsquared = np.zeros(u0.shape[0])
            for k in range(u0.shape[0]):
                nsquared[k] = comm.reduce(np.sum(np.power(np.abs(err[k] / sc[k]), 2)))
            if comm.Get_rank() == 0:
                est_to_bcast = np.zeros(1)
                est = np.max(np.sqrt(nsquared))
                est /= np.sqrt(np.array(context.T.shape(True)).prod())
                est_to_bcast[0] = est
            est_to_bcast = comm.bcast(est_to_bcast, root=0)
            est = est_to_bcast[0]

        elif errnorm == "inf":
            raise AssertionError("Don't use this, not sure if it works")
            # TODO: Test this error norm
            sc[:] = aTOL + np.maximum(np.abs(u0), np.abs(u0_new)) * rTOL
            err[:] = err[:] / sc[:]
            err = np.abs(err, out=err)
            asdf = np.max(err)
            x = np.zeros(asdf.shape)
            comm.Allreduce(asdf, x, op=MPI.MAX)
            est = np.abs(np.max(x))
            est /= np.sqrt(np.array(context.T.shape(True)).prod())
        else:
            assert False, "Wrong error norm"

        # Check error estimate
        exponent = 1.0 / (err_order + 1)
        if not predictivecontroller:
            factor = min(facmax, max(facmin, fac * pow((1 / est), exponent)))
        else:
            if "last_dt" not in vars(params):
                params.last_dt = dt
            if "last_est" not in vars(params):
                params.last_est = est

            last_dt = params.last_dt
            last_est = params.last_est
            factor = min(facmax, max(facmin, fac * pow((1 / est), exponent) * dt / last_dt * pow(last_est / est, exponent)))
        if adaptive:
            dt = dt * factor
            if est > 1.0:
                facmax = 1
                context.is_step_rejected_callback = True
                context.dt_rejected = dt_prev
                additional_callback(context)
                # The offset gets decreased in the  next step, which is something we do not want.
                if fsal:
                    offset[0] += 1
                continue

        # if predictivecontroller:
            # context.time_integrator["last_dt"] = dt_prev
            # context.time_integrator["last_est"] = est
        break

    # Update u0 and U
    u0[:] = u0_new
    return u0, dt, dt_prev


@optimizer
def RK4(u0, u1, u2, rhs, a, b, dt, solver, context):
    """Runge Kutta fourth order"""
    u2[:] = u1[:] = u0
    for rk in range(4):
        rhs = solver.ComputeRHS(rhs, u0, solver, **context)
        if rk < 3:
            u0[:] = u1 + b[rk] * dt * rhs
        u2 += a[rk] * dt * rhs
    u0[:] = u2
    return u0, dt, dt


@optimizer
def stochasticRK3(u0, u1, rhs, a, b, wi, dt, solver, context):
    """stochastic RK3 following Delong et al. (2013, PRE)"
    Specific to the fluctuating NS equation"""
    u1 = u0.copy()

    N = solver.params.N
    W_A, W_B = solver.generate_noise(N, **context)

    for rk in range(3):
        rhs = solver.ComputeRHS(rhs, u0, solver, W_A, W_B, wi[rk], solver.params.D / dt**0.5, **context)
        u0[:] = a[rk] * u1 + b[rk] * (u0 + dt * rhs)

    return u0, dt, dt


def implicit_predictor_corrector(u0, u1, rhs, dt, solver, context):
    """Implicit predictor corrector method"""
    L_inv = 1 / (1 - 0.5 * dt * context.linear_operator)

    N = solver.params.N
    W_A, W_B = solver.generate_noise(N, **context)

    D = solver.params.D
    # predictor step
    u1[:] = L_inv * (u0
                     + 0.5 * dt * solver.ComputeRHS(rhs, u0, solver, W_A, W_B, 0, D * (0.5 * dt)**(-0.5), **context))

    # corrector step
    u0[:] = L_inv * (u0
                     + 0.5 * dt * context.linear_operator * u0
                     + dt * solver.ComputeRHS(rhs, u1, solver, W_A, W_B, 1, D * (2 * dt)**(-0.5), **context))

    return u0, dt, dt


@optimizer
def ForwardEuler(u0, rhs, dt, solver, context):
    rhs = solver.ComputeRHS(rhs, u0, solver, **context)
    u0 += rhs * dt
    return u0, dt, dt


@optimizer
def AB2(u0, u1, rhs, dt, tstep, solver, context):
    rhs = solver.ComputeRHS(rhs, u0, solver, **context)
    if tstep == 0:
        u0 += rhs * dt
    else:
        u0 += (1.5 * rhs * dt - 0.5 * u1)
    u1[:] = rhs * dt
    return u0, dt, dt


def getintegrator(rhs, u0, solver, context):
    """Return integrator using choice in global parameter integrator."""
    params = solver.params
    u1 = u0.copy()

    if params.integrator == "RK4":
        # RK4 parameters
        a = np.array([1. / 6., 1. / 3., 1. / 3., 1. / 6.], dtype=context.float)
        b = np.array([0.5, 0.5, 1.], dtype=context.float)
        u2 = u0.copy()

        @wraps(RK4)
        def func():
            return RK4(u0, u1, u2, rhs, a, b, params.dt, solver, context)
        return func

    elif params.integrator == 'stochasticRK3':
        assert params.solver == 'FNS2D' or params.solver == 'OFNS', "stochasticRK3 is only implemented for FNS2D and OFNS3D"
        a = np.array([0.0, 3.0 / 4.0, 1.0 / 3.0], dtype=context.float)
        b = np.array([1.0, 1.0 / 4.0, 2.0 / 3.0], dtype=context.float)
        w_i = np.array([(8**0.5 + 3**0.5) / 5,
                        (-32**0.5 + 27**0.5) / 5,
                        (2**0.5 - 12**0.5) / 10], dtype=context.float)

        @wraps(stochasticRK3)
        def func():
            return stochasticRK3(u0, u1, rhs, a, b, w_i, params.dt, solver, context)
        return func

    elif params.integrator == 'implicitPC':
        assert params.solver == 'FNS2D' or params.solver == 'OFNS' or params.solver == 'FS2D', "implicitPC is only implemented for FS2D, FNS2D and OFNS3D"

        @wraps(implicit_predictor_corrector)
        def func():
            return implicit_predictor_corrector(u0, u1, rhs, params.dt, solver, context)
        return func

    elif params.integrator in ("BS5_adaptive", "BS5_fixed"):
        # Remove nodepy dependency since it requires matplotlib and six
        # import nodepy
        # A = nodepy.rk.loadRKM("BS5").A.astype(context.float)
        # b = nodepy.rk.loadRKM("BS5").b.astype(context.float)
        # bhat = nodepy.rk.loadRKM("BS5").bhat.astype(context.float)
        A = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                      [1 / 6, 0, 0, 0, 0, 0, 0, 0],
                      [2 / 27, 4 / 27, 0, 0, 0, 0, 0, 0],
                      [183 / 1372, -162 / 343, 1053 / 1372, 0, 0, 0, 0, 0],
                      [68 / 297, -4 / 11, 42 / 143, 1960 / 3861, 0, 0, 0, 0],
                      [597 / 22528, 81 / 352, 63099 / 585728, 58653 / 366080, 4617 / 20480, 0, 0, 0],
                      [174197 / 959244, -30942 / 79937, 8152137 / 19744439, 666106 / 1039181, -29421 / 29068, 482048 / 414219, 0, 0],
                      [587 / 8064, 0, 4440339 / 15491840, 24353 / 124800, 387 / 44800, 2152 / 5985, 7267 / 94080, 0]], dtype=context.float)
        b = np.array([587 / 8064, 0, 4440339 / 15491840, 24353 / 124800, 387 / 44800, 2152 / 5985, 7267 / 94080, 0], dtype=context.float)
        bhat = np.array([2479 / 34992, 0, 123 / 416, 612941 / 3411720, 43 / 1440, 2272 / 6561, 79937 / 1113912, 3293 / 556956], dtype=context.float)
        err_order = 4
        errnorm = "2"
        fsal = True
        adaptive = True if params.integrator == "BS5_adaptive" else False

        # Offset for fsal stuff. #TODO: infer this from tstep
        offset = [0]
        s = A.shape[0]
        fY_hat = np.zeros((s,) + u0.shape, dtype=u0.dtype)
        sc = np.zeros_like(u0)
        err = np.zeros_like(u0)

        @wraps(adaptiveRK)
        def func():
            return adaptiveRK(A, b, bhat, err_order, fY_hat, u1, sc, err, fsal,
                              offset, params.TOL, params.TOL, adaptive, errnorm,
                              rhs, u0, solver, params.dt, params.tstep, context,
                              solver.additional_callback, params)
        return func

    elif params.integrator == "ForwardEuler":
        @wraps(ForwardEuler)
        def func():
            return ForwardEuler(u0, rhs, params.dt, solver, context)
        return func

    elif params.integrator == "AB2":
        @wraps(AB2)
        def func():
            return AB2(u0, u1, rhs, params.dt, params.tstep, solver, context)
        return func

    else:
        raise NotImplementedError("Temporal integrator, %s, not implemented" % params.integrator)
