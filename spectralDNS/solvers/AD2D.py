__author__ = "Sijie Huang"
__date__ = "2024-06-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from shenfun import FunctionSpace, TensorProductSpace, Array, Function
from .spectralinit import *
from .NS import end_of_tstep
import matplotlib.pyplot as plt

def get_context():
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

    # Different bases for nonlinear term, either 2/3-rule or 3/2-rule
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    Tp = T.get_dealiased(**kw)

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

    # Velocity and pressure. Use ndarray view for efficiency
    C = Array(T)
    C_hat = Function(T)
    c_dealias = Array(Tp)

    # Primary variable
    u = C_hat

    # RHS array
    dU = Function(T)
    Source = Function(T) # Possible source term initialized to zero
    work = work_arrays()

    hdf5file = ADFile(config.params.solver,
                      checkpoint={'space': T,
                                  'data': {'0': {'C': [C_hat]}}},
                      results={'space': T,
                               'data': {'C': [C]}})

    return config.AttributeDict(locals())


class ADFile(HDF5File):
    """Subclass HDF5File for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, **context):
        """Transform to real data before storing the solution"""
        get_scalar(**context)
        
def get_scalar(C, C_hat, T, **context):
    """Compute velocity from context"""
    C = T.backward(C_hat, C)
    return C

def set_scalar(C, C_hat, T, **context):
    """Compute velocity from context"""
    C_hat = T.forward(C, C_hat)
    return C_hat

def getConvection(convection):
    """Compute convection term"""
    
    def Conv(rhs, c_hat, K):
        adv_u, adv_v = params.advection_velocity
        rhs = -1j*(adv_u*K[0] + adv_v*K[1])*c_hat
        
        return rhs 
    
    Conv.convection = convection
    return Conv

def ComputeRHS(rhs, c_hat, solver, K, K2, Source, mask, **context):
    """Compute right hand side of NS equation"""
    rhs = solver.conv(rhs, c_hat, K)
    if mask is not None:
        rhs.mask_nyquist(mask)
    
    # adding viscous term 
    rhs -= params.nu*K2*c_hat
    
    return rhs