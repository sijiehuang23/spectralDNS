__author__ = "Sijie Huang"
__date__ = "2024-06-12"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

# pylint: disable=unused-variable,unused-argument,function-redefined

# Reuses most of NS.py module, but curl in 2D is a scalar
from shenfun import FunctionSpace, CompositeSpace, TensorProductSpace, VectorSpace, Array, Function
from .spectralinit import *
import numpy as np
from .NS import end_of_tstep


def get_context():
    """Set up context for 3D fluctuating Stokes solver (FS)"""
    float, complex, mpitype = datatypes(params.precision)
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    dim = len(params.N)
    def dtype(d): return float if d == dim - 1 else complex
    V = [FunctionSpace(params.N[i], 'F', domain=(0, params.L[i]), dtype=dtype(i)) for i in range(dim)]

    kw0 = {
        'threads': params.threads,
        'planner_effort': params.planner_effort['fft']
    }
    T = TensorProductSpace(
        comm, V, dtype=float,
        slab=(params.decomposition == 'slab'),
        collapse_fourier=collapse_fourier, **kw0
    )
    VT = VectorSpace(T)
    VW = CompositeSpace([T] * (dim**2))

    mask = T.get_mask_nyquist() if params.mask_nyquist else None

    kw = {
        'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
        'dealias_direct': params.dealias == '2/3-rule'
    }
    Tp = T.get_dealiased(**kw)
    VTp = VectorSpace(Tp)

    # Mesh variables
    X = T.local_mesh(True)
    K = T.local_wavenumbers(scaled=True)
    for i in range(dim):
        X[i] = X[i].astype(float)
        K[i] = K[i].astype(float)
    K2 = np.zeros(T.shape(True), dtype=float)
    for i in range(dim):
        K2 += K[i] * K[i]

    K_over_K2 = np.zeros(VT.shape(True), dtype=float)
    for i in range(dim):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    linear_operator = -params.nu * K2

    # Solution variables
    U = Array(VT)
    U_hat = Function(VT)
    P = Array(T)
    P_hat = Function(T)
    curl = Array(T)
    u_dealias = Array(VTp)

    if params.noise_type == 'thermal':
        W = Array(VW)
        W_hat = Function(VW)
    elif params.noise_type == 'correlated':
        W = Array(VT)
        W_hat = Function(VT)

    G = Array(T)
    G_hat = Function(T)
    nu_hat = Function(T)

    # Primary variable
    u = U_hat

    # RHS and work arrays
    dU = Function(VT)
    work = work_arrays()

    hdf5file = FS2DFile(
        config.params.solver,
        checkpoint={'space': VT,
                    'data': {'0': {'u_hat': [U_hat]}}},
        results={
            'space': VT,
            'data': {
                'u': [U[0]],
                'v': [U[1]],
                'w': [U[2]]
            }
        }
    )

    return config.AttributeDict(locals())


class FS2DFile(HDF5File):
    def update_components(self, U, U_hat, W, W_hat, **context):
        U = U_hat.backward(U)


def get_velocity(U, U_hat, VT, **context):
    U = VT.backward(U_hat, U)
    return U


def get_pressure(P, P_hat, T, **context):
    P = T.backward(-1j * P_hat, P)
    return P


def set_velocity(U, U_hat, VT, **context):
    """Compute velocity from context"""
    U_hat = VT.forward(U, U_hat)
    return U_hat


def generate_noise(N, W_hat, K2, **context):
    shape = W_hat.shape
    W_A = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    W_B = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    W_A /= np.prod(N)**0.5
    W_B /= np.prod(N)**0.5

    for i in range(shape[0]):
        W_A[i] *= np.where(K2 == 0, 0, 1)
        W_B[i] *= np.where(K2 == 0, 0, 1)

    return W_A, W_B


def add_thermal_fluctuation(rhs, W_A, W_B, wi, mag, w_hat, K, **context):
    W_i = (W_A + wi * W_B)

    # symmetrize
    w_hat[0] = W_i[0] + W_i[0]
    w_hat[1] = W_i[1] + W_i[3]
    w_hat[2] = W_i[2] + W_i[6]
    w_hat[3] = W_i[3] + W_i[1]
    w_hat[4] = W_i[4] + W_i[4]
    w_hat[5] = W_i[5] + W_i[7]
    w_hat[6] = W_i[6] + W_i[2]
    w_hat[7] = W_i[7] + W_i[5]
    w_hat[8] = W_i[8] + W_i[8]
    w_hat *= 0.5**0.5

    # take divergence and add to rhs
    rhs[0] = 1j * (K[0] * w_hat[0] + K[1] * w_hat[1] + K[2] * w_hat[2]) * mag
    rhs[1] = 1j * (K[0] * w_hat[3] + K[1] * w_hat[4] + K[2] * w_hat[5]) * mag
    rhs[2] = 1j * (K[0] * w_hat[6] + K[1] * w_hat[7] + K[2] * w_hat[8]) * mag

    return rhs


def add_correlated_noise(rhs, W_A, W_B, wi, mag, w_hat, G_hat, **context):
    w_hat[:] = W_A + wi * W_B

    # filter
    w_hat[0] *= G_hat
    w_hat[1] *= G_hat
    w_hat[2] *= G_hat

    # add to rhs
    rhs[0] = w_hat[0] * mag
    rhs[1] = w_hat[1] * mag
    rhs[2] = w_hat[2] * mag

    return rhs


def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, U_hat, work, Tp, K, u_dealias):

            # momentum equation
            curl_dealias = work[(u_dealias, 0, True)]
            curl_hat = work[(rhs, 0, True)]

            curl_hat[0] = 1j * (K[1] * U_hat[2] - K[2] * U_hat[1])
            curl_hat[1] = 1j * (K[2] * U_hat[0] - K[0] * U_hat[2])
            curl_hat[2] = 1j * (K[0] * U_hat[1] - K[1] * U_hat[0])
            
            curl_dealias = Tp.backward(curl_hat, curl_dealias)

            rhs[0] = Tp.forward(u_dealias[1] * curl_dealias[2] - u_dealias[2] * curl_dealias[1], rhs[0])
            rhs[1] = Tp.forward(u_dealias[2] * curl_dealias[0] - u_dealias[0] * curl_dealias[2], rhs[0])
            rhs[2] = Tp.forward(u_dealias[0] * curl_dealias[1] - u_dealias[1] * curl_dealias[0], rhs[0])

            return rhs

    Conv.convection = convection
    return Conv


@optimizer
def projection(rhs, P_hat, K_over_K2, K):
    # Compute Leray projection
    P_hat = np.sum(rhs * K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    for i in range(rhs.shape[0]):
        rhs[i] -= P_hat * K[i]

    return rhs


def add_diffusion(rhs, u_hat, K2, nu, **context):
    rhs[0] -= nu * K2 * u_hat[0]
    rhs[1] -= nu * K2 * u_hat[1]
    rhs[2] -= nu * K2 * u_hat[2]

    return rhs


def ComputeRHS(rhs, u_hat, solver, W_A, W_B, wi, mag, K, K2, K_over_K2, P_hat, mask, W_hat, G_hat, nu_hat, **context):

    # add thermal fluctuation
    if params.noise_type == 'thermal':
        rhs = solver.add_thermal_fluctuation(rhs, W_A, W_B, wi, mag, W_hat, K)
    elif params.noise_type == 'correlated':
        rhs = solver.add_correlated_noise(rhs, W_A, W_B, wi, mag, W_hat, G_hat, **context)

    if mask is not None:
        rhs.mask_nyquist(mask)

    # project to divergence-free space
    rhs = solver.projection(rhs, P_hat, K_over_K2, K)

    # add diffusion
    if params.integrator.casefold() != 'implicitPC'.casefold():
        if params.noise_type == 'thermal':
            rhs = solver.add_diffusion(rhs, u_hat, K2, params.nu)
        elif params.noise_type == 'correlated':
            rhs = solver.add_diffusion(rhs, u_hat, K2, nu_hat)

    return rhs
