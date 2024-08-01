__author__ = "Sijie Huang"
__date__ = "2024-07-10"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from shenfun import FunctionSpace, TensorProductSpace, VectorSpace, CompositeSpace, \
    Array, Function
from .spectralinit import *
import numpy as np
from .NS import end_of_tstep

def get_context():
    """Set up context for odd fluctuationg Navier-Stokes solver"""
    float, complex, mpitype = datatypes(params.precision)
    collapse_fourier = False if params.dealias == '3/2-rule' else True
    dim = len(params.N)
    dtype = lambda d: float if d == dim-1 else complex
    V = [FunctionSpace(params.N[i], 'F', domain=(0, params.L[i]),
                       dtype=dtype(i)) for i in range(dim)]

    kw0 = {'threads': params.threads,
           'planner_effort': params.planner_effort['fft']}
    T = TensorProductSpace(comm, V, dtype=float,
                           slab=(params.decomposition == 'slab'),
                           collapse_fourier=collapse_fourier, **kw0)
    VT = VectorSpace(T)
    UT = CompositeSpace([T]*(dim+1))
    WT = CompositeSpace([T]*(dim**2))

    # Different bases for nonlinear term, either 2/3-rule or 3/2-rule
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    Tp = T.get_dealiased(**kw)
    UTp = CompositeSpace([Tp]*(dim+1))

    mask = T.get_mask_nyquist() if params.mask_nyquist else None

    # Mesh variables
    X = T.local_mesh(True)
    K = T.local_wavenumbers(scaled=True)
    for i in range(dim):
        X[i] = X[i].astype(float)
        K[i] = K[i].astype(float)
    K2 = np.zeros(T.shape(True), dtype=float)
    for i in range(dim):
        K2 += K[i]*K[i]

    K_over_K2 = np.zeros(VT.shape(True), dtype=float)
    for i in range(dim):
        K_over_K2[i] = K[i] / np.where(K2 == 0, 1, K2)

    # Velocity and pressure. Use ndarray view for efficiency
    Uc = Array(UT)
    Uc_hat = Function(UT)
    P = Array(T)
    P_hat = Function(T)
    C = Array(T)
    C_hat = Function(T)
    uc_dealias = Array(UTp)
    W = Array(WT)
    W_hat = Function(WT)
    
    U = Uc[:-1]

    # Primary variable
    u = Uc_hat

    # RHS array
    dU = Function(UT)
    curl = Array(VT)
    Source = Function(WT) 
    work = work_arrays()

    hdf5file = OFNSFile(config.params.solver,
                      checkpoint={'space': VT,
                                  'data': {'0': {'U': [Uc_hat]}}},
                      results={'space': VT,
                               'data': {'U': [Uc[0]], 
                                        'V': [Uc[1]], 
                                        'W': [Uc[2]], 
                                        'C': [Uc[3]],
                                        'P': [P]}})

    return config.AttributeDict(locals())

class OFNSFile(HDF5File):
    """Subclass HDF5File for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, Uc, Uc_hat, P, P_hat, T, **context):
        """Transform to real data before storing the solution"""
        Uc = Uc_hat.backward(Uc)
        P = get_pressure(P, P_hat, T)

def get_velocity(U, Uc_hat, VT, **context):
    """Compute velocity from context"""
    U = VT.backward(Uc_hat[:-1], U)
    return U

def get_scalar(C, Uc_hat, T, **context):
    """Compute scalar from context"""
    C = T.backward(Uc_hat[-1], C)
    return C

def get_pressure(P, P_hat, T, **context):
    """Compute pressure from context"""
    P = T.backward(-1j*P_hat, P)
    return P

def set_velocity(U, Uc_hat, VT, **context):
    """Compute velocity from context"""
    Uc_hat[:-1] = VT.forward(U, Uc_hat[:-1])
    return Uc_hat

def set_scalar(C, C_hat, T, **context):
    """Compute scalar from context"""
    C_hat = T.forward(C, C_hat)
    return C_hat

def get_divergence(T, K, Uc_hat, mask, **context):
    div_u = Array(T)
    div_Uc_hat = 1j*(K[0]*Uc_hat[0]+K[1]*Uc_hat[1]+K[2]*Uc_hat[2])
    div_u = T.backward(div_Uc_hat, div_u)
    return div_u

# def compute_curl(c, a, work, T, K):
#     """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
#     curl_hat = work[(a, 0, False)]
#     curl_hat = cross2(curl_hat, K, a)
#     c = T.backward(curl_hat, c)
#     return c

def standard_convection(rhs, uc_dealias, Uc_hat, work, Tp, K):
    """rhs_i = u_j du_i/dx_j"""
    gradUi = work[(uc_dealias[:-1], 1, False)]
    
    for i in range(uc_dealias.shape[0]):
        for j in range(3):
            gradUi[j] = Tp.backward(1j*K[j]*Uc_hat[i], gradUi[j])
        rhs[i] = Tp.forward(np.sum(uc_dealias[:-1]*gradUi, 0), rhs[i])
        
    return rhs

def divergence_convection(rhs, uc_dealias, work, Tp, K, add=False):
    """rhs_i = div(u_i u_j)"""
    if not add:
        rhs.fill(0)
    UUi_hat = work[(rhs, 0, False)]
    
    # momentum equation 
    for i in range(3):
        UUi_hat[i] = Tp.forward(uc_dealias[0]*uc_dealias[i], UUi_hat[i])
    rhs[0] += 1j*(K[0]*UUi_hat[0] + K[1]*UUi_hat[1] + K[2]*UUi_hat[2])
    rhs[1] += 1j*K[0]*UUi_hat[1]
    rhs[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = Tp.forward(uc_dealias[1]*uc_dealias[1], UUi_hat[0])
    UUi_hat[1] = Tp.forward(uc_dealias[1]*uc_dealias[2], UUi_hat[1])
    UUi_hat[2] = Tp.forward(uc_dealias[2]*uc_dealias[2], UUi_hat[2])
    rhs[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    rhs[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    
    # scalar equation 
    for i in range(3):
        UUi_hat[i] = Tp.forward(uc_dealias[i]*uc_dealias[-1], UUi_hat[i])
        rhs[4] += 1j*K[i]*UUi_hat[i]
    
    return rhs

def getConvection(convection):

    if convection == "Standard":

        def Conv(rhs, Uc_hat, work, Tp, UTp, K, uc_dealias):
            uc_dealias = UTp.backward(Uc_hat, uc_dealias)
            rhs = -standard_convection(rhs, uc_dealias, Uc_hat, work, Tp, K)
            
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, Uc_hat, work, Tp, UTp, K, uc_dealias):
            uc_dealias = UTp.backward(Uc_hat, uc_dealias)
            rhs = divergence_convection(rhs, uc_dealias, work, Tp, K, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skewed":

        def Conv(rhs, Uc_hat, work, Tp, UTp, K, uc_dealias):
            uc_dealias = UTp.backward(Uc_hat, uc_dealias)
            rhs = standard_convection(rhs, uc_dealias, Uc_hat, work, Tp, K)
            rhs = divergence_convection(rhs, uc_dealias, work, Tp, K, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":
        raise NotImplementedError("Rotational form not implemented for OFNS")
    
    else:
        raise ValueError("Unknown convection type")

    Conv.convection = convection
    return Conv

@optimizer
def projection(rhs, K, P_hat, K_over_K2):
    """Add contributions from pressure and diffusion to the rhs"""

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs[:-1]*K_over_K2, 0, out=P_hat)

    # Subtract pressure gradient
    for i in range(3):
        rhs[i] -= P_hat*K[i]

    return rhs

def add_diffusion(rhs, Uc_hat, nu, alpha, K2):
    rhs[:3] -= nu*K2*Uc_hat[:3]
    rhs[3]  -= alpha*K2*Uc_hat[3]
    
    return rhs 

def generate_noise(W, **context):
    W_1 = np.random.randn(*W.shape)
    W_2 = np.random.randn(*W.shape)
    
    return W_1, W_2

def add_thermal_fluctuation(rhs, W_A, W_B, wi, dt, dx, mag, w, w_hat, K, WT, **context):
    W_i = W_A + wi*W_B
    
    # symmetrize
    # the matrix is arranged as follows:
    # w = 
    # --       --
    # | 0  1  2 |
    # | 3  4  5 |
    # | 6  7  8 |
    # --       --
    w[0] = W_i[0]+W_i[0]
    w[1] = W_i[1]+W_i[3]
    w[2] = W_i[2]+W_i[6]
    w[3] = W_i[3]+W_i[1]
    w[4] = W_i[4]+W_i[4]
    w[5] = W_i[5]+W_i[7]
    w[6] = W_i[6]+W_i[2]
    w[7] = W_i[7]+W_i[5]
    w[8] = W_i[8]+W_i[8]
    w *= 0.5**0.5
    
    # forward Fourier transform 
    w_hat = WT.forward(w, w_hat)
    w_hat *= mag/np.sqrt(dt)
    
    rhs[0] += 1j*(K[0]*w_hat[0] + K[1]*w_hat[1] + K[2]*w_hat[2])
    rhs[1] += 1j*(K[0]*w_hat[3] + K[1]*w_hat[4] + K[2]*w_hat[5])
    rhs[2] += 1j*(K[0]*w_hat[6] + K[1]*w_hat[7] + K[2]*w_hat[8])
    
    return rhs

def ComputeRHS(rhs, uc_hat, solver, W_A, W_B, wi, work, K, K2, K_over_K2, P_hat, Tp, UTp, 
               WT, uc_dealias, mask, W, W_hat, **context):
    
    # compute nonlinear term
    rhs = solver.conv(rhs, uc_hat, work, Tp, UTp, K, uc_dealias)
    
    if mask is not None:
        rhs.mask_nyquist(mask)

    # add thermal fluctuations
    # rhs = solver.add_thermal_fluctuation(rhs, W_A, W_B, wi, params.dt, params.dx, params.mag_thermal_fluctuation, W, W_hat, K, WT)

    # projection
    rhs = solver.projection(rhs, K, P_hat, K_over_K2)

    # add diffusion
    rhs = add_diffusion(rhs, uc_hat, params.nu, params.alpha, K2)

    return rhs
