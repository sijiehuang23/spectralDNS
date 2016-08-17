__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *

def setup():
    """Set up context for solver"""
    
    # FFT class performs the 3D parallel transforms
    FFT = get_FFT(params)
    float, complex, mpitype = datatypes(params.precision)
    
    # Mesh variables
    X = FFT.get_local_mesh()
    K = FFT.get_scaled_local_wavenumbermesh()    
    K2 = np.sum(K*K, 0, dtype=float)
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)    
    
    # Velocity and pressure
    U     = empty((3,) + FFT.real_shape(), dtype=float)  
    U_hat = empty((3,) + FFT.complex_shape(), dtype=complex)
    P     = empty(FFT.real_shape(), dtype=float)
    P_hat = empty(FFT.complex_shape(), dtype=complex)

    # Primary variable
    u = U_hat

    # RHS array
    dU     = empty((3,) + FFT.complex_shape(), dtype=complex)
    curl   = empty((3,) + FFT.real_shape(), dtype=float)   
    Source = None
    work = work_arrays()
        
    hdf5file = NSWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P},
                        chkpoint={'current':{'U':U, 'P':P}, 'previous':{}},
                        filename=params.solver+'.h5')

    return config.ParamsBase(locals())

class NSWriter(HDF5Writer):
    """Subclass HDF5Writer for appropriate updating of real components

    The method 'update_components' is used to transform all variables
    that are to be stored. If more variables than U and P are
    wanted, then subclass HDF5Writer in the application.
    """
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)

def get_curl(curl, U_hat, work, FFT, K, **context):
    """Compute curl from context"""
    curl = compute_curl(curl, U_hat, work, FFT, K)
    return curl

def get_velocity(U, U_hat, FFT, **context):
    """Compute velocity from context"""
    for i in range(3):
        U[i] = FFT.ifftn(U_hat[i], U[i])
    return U

def get_pressure(P, P_hat, FFT, **context):
    """Compute pressure from context"""
    P = FFT.ifftn(1j*P_hat, P)

def forward_transform(u, u_hat, FFT):
    """A common method for transforming forward """
    for i in range(3):
        u_hat[i] = FFT.fftn(u[i], u_hat[i])
    return u_hat

def backward_transform(u_hat, u, FFT):
    """A common method for transforming backward"""
    for i in range(3):
        u[i] = FFT.ifftn(u_hat[i], u[i])
    return u

def compute_curl(c, a, work, FFT, K, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2(curl_hat, K, a)
    c[0] = FFT.ifftn(curl_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(curl_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(curl_hat[2], c[2], dealias)
    return c

def _standard_convection(rhs, u_dealias, U_hat, work, FFT, K, dealias=None):
    """rhs_i = u_j du_i/dx_j"""
    gradUi = work[(u_dealias, 2, False)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        rhs[i] = FFT.fftn(np.sum(u_dealias*gradUi, 0), rhs[i], dealias)
    return rhs

def _divergence_convection(rhs, u_dealias, work, FFT, K, dealias=None, add=False):
    """rhs_i = div(u_i u_j)"""
    if not add: rhs.fill(0)
    UUi_hat = work[(rhs, 0, False)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(u_dealias[0]*u_dealias[i], UUi_hat[i], dealias)
    rhs[0] += 1j*np.sum(K*UUi_hat, 0)
    rhs[1] += 1j*K[0]*UUi_hat[1]
    rhs[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(u_dealias[1]*u_dealias[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(u_dealias[1]*u_dealias[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(u_dealias[2]*u_dealias[2], UUi_hat[2], dealias)
    rhs[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    rhs[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return rhs

#@profile
def Cross(rhs, a, b, work, FFT, dealias=None):
    """rhs_k = F_k(a x b)"""
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    rhs[0] = FFT.fftn(Uc[0], rhs[0], dealias)
    rhs[1] = FFT.fftn(Uc[1], rhs[1], dealias)
    rhs[2] = FFT.fftn(Uc[2], rhs[2], dealias)
    return rhs

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = _standard_convection(rhs, u_dealias, u_hat, work, FFT, K, params.dealias)
            rhs[:] *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = _divergence_convection(rhs, u_dealias, work, FFT, K, params.dealias, False)
            rhs[:] *= -1
            return rhs

    elif convection == "Skewed":

        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)
            rhs = _standard_convection(rhs, u_dealias, u_hat, work, FFT, K, params.dealias)
            rhs = _divergence_convection(rhs, u_dealias, work, FFT, K, params.dealias, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":

        #@profile
        def Conv(rhs, u_hat, work, FFT, K):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias),
                              float, 0, False)]
            curl_dealias = work[((3,)+FFT.work_shape(params.dealias),
                                 float, 1, False)]
            for i in range(3):
                u_dealias[i] = FFT.ifftn(u_hat[i], u_dealias[i], params.dealias)

            curl_dealias = compute_curl(curl_dealias, u_hat, work, FFT, K, params.dealias)
            rhs = Cross(rhs, u_dealias, curl_dealias, work, FFT, params.dealias)
            return rhs

    return Conv

@optimizer
def add_pressure_diffusion(rhs, u_hat, K2, K, P_hat, K_over_K2, nu):
    """Add contributions from pressure and diffusion to the rhs"""

    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = np.sum(rhs*K_over_K2, 0, out=P_hat)

    # Subtract pressure gradient
    rhs -= P_hat*K

    # Subtract contribution from diffusion
    rhs -= nu*K2*u_hat

    return rhs

def ComputeRHS(rhs, u_hat, work, FFT, K, K2, P_hat, K_over_K2, **context):
    """Compute rhs of spectral Navier Stokes
    
    args:
        rhs        The right hand side to be returned
        u_hat      The FFT of the velocity at current time. May differ from
                   context.U_hat since it is set by the integrator

    Remaining args extracted from context:
        work       Work arrays
        FFT        Transform class from mpiFFT4py
        K          Scaled wavenumber mesh
        K2         K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        P_hat      Transfomred pressure
    """

    rhs = conv(rhs, u_hat, work, FFT, K)

    rhs = add_pressure_diffusion(rhs, u_hat, K2, K, P_hat, K_over_K2, params.nu)

    return rhs
