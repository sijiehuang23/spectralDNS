__author__ = "Sijie Huang"
__date__ = "2024-06-12"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

# Reuses most of NS.py module, but curl in 2D is a scalar
from shenfun import FunctionSpace, CompositeSpace, TensorProductSpace, VectorSpace, Array, Function
from .spectralinit import *
import numpy as np
from .NS import end_of_tstep

def get_context():
    """Set up context for 2D odd fluctuating Navier-Stokes solver (OFNS)"""
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
    VM = CompositeSpace([T]*(dim+1))

    mask = T.get_mask_nyquist() if params.mask_nyquist else None

    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    Tp = T.get_dealiased(**kw) 
    VTp = VectorSpace(Tp)
    VMp = CompositeSpace([Tp]*(dim+1))

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

    # Solution variables
    Uc = Array(VM)
    Uc_hat = Function(VM)
    P = Array(T)
    P_hat = Function(T)
    curl = Array(T)
    W_hat = Function(T)
    u_dealias = Array(VTp)
    uc_dealias = Array(VMp)

    # Create views into large data structures
    C = Uc[2]
    C_hat = Uc_hat[2]
    U = Uc[:2]
    U_hat = Uc_hat[:2]

    # Primary variable
    u = Uc_hat

    # RHS and work arrays
    dU = Function(VM)
    work = work_arrays()
    
    hdf5file = OFNS2DFile(config.params.solver,
                      checkpoint={'space': VT,
                                  'data': {'0': {'Uc': [Uc_hat]}}},
                      results={'space': VT,
                               'data': {'Uc': [Uc], 'P': [P]}})
    
    return config.AttributeDict(locals())

class OFNS2DFile(HDF5File):
    def update_components(self, Uc, Uc_hat, **context):
        """Transform to real data before storing the solution"""
        # get_velocity(**context)
        # get_scalar(**context)
        # get_pressure(**context)
        Uc = Uc_hat.backward(Uc)
        
def get_velocity(U, U_hat, VT, **context):
    U = VT.backward(U_hat, U)
    
def get_scalar(C, Uc_hat, T, **context):
    C = T.backward(Uc_hat[2], C)
    
def get_pressure(P, P_hat, T, **context):
    P = T.backward(-1j*P_hat, P)
    
def set_velocity(U, U_hat, VT, **context):
    """Compute velocity from context"""
    U_hat = VT.forward(U, U_hat)
    return U_hat

def set_scalar(C, C_hat, T, **context):
    """Compute velocity from context"""
    C_hat = T.forward(C, C_hat)
    return C_hat

def get_curl(curl, W_hat, U_hat, work, T, K, **context):
    W_hat[:] = 0
    W_hat = cross2(W_hat, K, U_hat)
    curl = W_hat.backward(curl)
    return curl

def get_divergence(T, K, U_hat, mask, **context):
    div_u = Array(T)
    div_u_hat = 1j*(K[0]*U_hat[0]+K[1]*U_hat[1])
    div_u = T.backward(div_u_hat, div_u)
    return div_u

def add_thermal_fluctuation():
    pass 

def add_odd_viscous():
    pass 

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, uc_hat, work, Tp, VMp, K, uc_dealias):
            uc_dealias = VMp.backward(uc_hat, uc_dealias)
            u_dealias = uc_dealias[:2]
            c_dealias = uc_dealias[2]
            
            # momentum equation
            curl_dealias = work[(u_dealias[0], 0, True)]
            curl_hat = work[(rhs[0], 0, True)]
            
            # curl_hat = cross2(curl_hat, K, uc_hat[:2])
            curl_hat = 1j*(K[0]*uc_hat[1]-K[1]*uc_hat[0])
            curl_dealias = Tp.backward(curl_hat, curl_dealias)
            
            rhs[0] = Tp.forward(u_dealias[1]*curl_dealias, rhs[0])
            rhs[1] = Tp.forward(-u_dealias[0]*curl_dealias, rhs[1])
            
            # scalar equation 
            gradUC_hat = work[(uc_hat[:2], 0, True)]
            for i in range(2):
                gradUC_hat[i] = Tp.forward(u_dealias[i]*c_dealias, gradUC_hat[i])
                
            rhs[2] = -1j*(K[0]*gradUC_hat[0]+K[1]*gradUC_hat[1])
            
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_pressure_diffusion(rhs, uc_hat, P_hat, K_over_K2, K, K2, nu, alpha):
    u_hat = uc_hat[:2]
    c_hat = uc_hat[2]

    # Compute Leray projection 
    P_hat = np.sum(rhs[:2]*K_over_K2, 0, out=P_hat)

    # Add pressure gradient
    for i in range(2):
        rhs[i] -= P_hat*K[i]

    # Add contribution from diffusion
    rhs[0] -= nu*K2*u_hat[0]
    rhs[1] -= nu*K2*u_hat[1]
    rhs[2] -= alpha*K2*c_hat
    
    return rhs

def ComputeRHS(rhs, uc_hat, solver, work, K, K2, K_over_K2, P_hat, T, Tp,
               VM, VMp, uc_dealias, mask, **context):
    
    rhs = solver.conv(rhs, uc_hat, work, Tp, VMp, K, uc_dealias)
    
    #TODO add thermal fluctuation and odd viscous terms 
    
    if mask is not None:
        rhs.mask_nyquist(mask)
    rhs = solver.add_pressure_diffusion(rhs, uc_hat, P_hat, K_over_K2, K, K2, params.nu, params.alpha)

    return rhs

