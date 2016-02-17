import numpy as np
from numpy.polynomial import chebyshev as n_cheb
from mpiFFT4py import dct
import SFTc
import scipy.sparse.linalg as la
from cbcdns.shen.la import TDMA, PDMA
from cbcdns.shen.Matrices import BBBmat
"""
Fast transforms for pure Chebyshev basis or 
Shen's Chebyshev basis: 

  For homogeneous Dirichlet boundary conditions:
 
    phi_k = T_k - T_{k+2}
    
  For homogeneous Neumann boundary conditions:
    
    phi_k = T_k - (k/k+2)**2 * T_{k+2}
    
  For Biharmonic basis with both homogeneous Dirichlet
  and Neumann:
  
    phi_k = T_k - 2(k+2)/(k+3)*T_{k+2} + (k+1)/(k+3)*T_{k+4}

Use either Chebyshev-Gauss (GC) or Gauss-Lobatto (GL)
points in real space.

The ChebyshevTransform may be used to compute derivatives
through fast Chebyshev transforms.

"""
float, complex = np.float64, np.complex128
pi, zeros, ones = np.pi, np.zeros, np.ones

class ChebyshevTransform(object):
    
    def __init__(self, quad="GL", fast_transform=True): 
        self.quad = quad
        self.fast_transform = fast_transform
        self.points = None
        self.weights = None
        self.return_array = None
        
    def get_return_array(self, v):
        if self.return_array is None:
            self.return_array = np.zeros(v.shape, dtype=v.dtype)
        else:
            if not self.return_array.shape == v.shape:
                self.return_array = np.zeros(v.shape, dtype=v.dtype)
        self.return_array[:] = 0
        return self.return_array
        
    def points_and_weights(self, N):
        self.N = N
        if self.quad == "GL":
            points = (n_cheb.chebpts2(N)[::-1]).astype(float)
            #points = (n_cheb.chebpts2(N)).astype(float)
            weights = zeros(N)+pi/(N-1)
            weights[0] /= 2
            weights[-1] /= 2

        elif self.quad == "GC":
            points, weights = n_cheb.chebgauss(N)
            #points = points[::-1]
            points = points.astype(float)
            weights = weights.astype(float)
            
        return points, weights
    
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-1).T
        
    def chebDerivativeCoefficients(self, fk, fj):
        SFTc.chebDerivativeCoefficients(fk, fj)  
        return fj

    def chebDerivative_3D(self, fj, fd):
        fk = fj.copy()
        fk = self.fct(fj, fk)
        fkd = fk.copy()
        fkd = SFTc.chebDerivativeCoefficients_3D(fk, fkd)
        fd = self.ifct(fkd, fd)
        return fd
    
    def fastChebDerivative(self, fj, fd):
        """Compute derivative of fj at the same points."""
        fk = fj.copy()
        fk = self.fct(fj, fk)
        fkd = fk.copy()
        fkd = self.chebDerivativeCoefficients(fk, fkd)
        fd  = self.ifct(fkd, fd)
        return fd
        
    def fct(self, fj, cj):
        """Fast Chebyshev transform."""
        N = fj.shape[0]
        if self.quad == "GC":
            cj[:] = dct(fj, type=2, axis=0)            
            cj /= N
            cj[0] /= 2
                
        elif self.quad == "GL":
            cj[:] = dct(fj, type=1, axis=0)/(N-1)
            cj[0] /= 2
            cj[-1] /= 2
            
        return cj

    def ifct(self, fk, cj):
        """Inverse fast Chebyshev transform."""
        if self.quad == "GC":
            cj[:] = 0.5*dct(fk, type=3, axis=0)
            cj += 0.5*fk[0]
        
        elif self.quad == "GL":
            cj[:] = 0.5*dct(fk, type=1, axis=0)
            cj += 0.5*fk[0]
            cj[::2] += 0.5*fk[-1]
            cj[1::2] -= 0.5*fk[-1]

        return cj
    
    def fastChebScalar(self, fj, fk):
        """Fast Chebyshev scalar product."""
        if self.fast_transform:
            N = fj.shape[0]
            if self.quad == "GC":
                fk[:] = dct(fj, type=2, axis=0)*pi/(2*N)
            
            elif self.quad == "GL":
                fk[:] = dct(fj, type=1, axis=0)*pi/(2*(N-1))
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:] = np.dot(self.V, fj*self.weights)

        return fk

    def slice(self, N):
        return slice(0, N)

class ShenDirichletBasis(ChebyshevTransform):
    
    def __init__(self, quad="GL", fast_transform=True):
        ChebyshevTransform.__init__(self, quad=quad, fast_transform=fast_transform)
        self.N = -1
        self.w_hat = None
        self.Solver = TDMA(quad, False)
        
    def init(self, N):
        """Vandermonde matrix is used just for verification"""
        self.points, self.weights = self.points_and_weights(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-2 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - n_cheb.chebvander(self.points, N-1)[:, 2:].T

    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-2).astype(float)
        
        else:
            kk = np.mgrid[:N[0]-2, :N[1], :N[2]].astype(float)
            return kk[0]

    #@profile
    def fastShenScalar(self, fj, fk):
                
        if self.fast_transform:
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= fk[2:]
        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-2] = np.dot(self.V, fj*self.weights)
        fk[-2:] = 0     # Last two not used by Shen
        return fk
        
    #@profile
    def ifst(self, fk, fj):
        """Fast inverse Shen transform
        Transform needs to take into account that phi_k = T_k - T_{k+2}
        fk contains Shen coefficients in the first fk.shape[0]-2 positions
        """
        if self.fast_transform:
            N = len(fj)
            if isinstance(self.w_hat, np.ndarray):
                if not self.w_hat.shape == fk.shape:
                    self.w_hat = fk.copy()
                
            if self.w_hat is None:
                self.w_hat = fk.copy()
                
            self.w_hat[-2:] = 0
            self.w_hat[:-2] = fk[:-2] 
            self.w_hat[2:] -= fk[:-2] 
            #self.w_hat[:2] = fk[:2]
            #self.w_hat[2:] = fk[2:]-fk[:-2]  
            
            fj = self.ifct(self.w_hat, fj)
            return fj    
        
        else:
            if self.points is None: self.init(fj.shape[0])
            return np.dot(self.V.T, fk[:-2])

    #@profile
    def fst(self, fj, fk):
        """Fast Shen transform
        """
        fk = self.fastShenScalar(fj, fk)
        fk = self.Solver(fk)
        return fk
    
    def slice(self, N):
        return slice(0, N-2)

class ShenNeumannBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC", fast_transform=True): 
        ShenDirichletBasis.__init__(self, quad, fast_transform)
        self.factor = None        
        self.Solver = TDMA(quad, True)
            
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix. Note! N points in real space gives N-3 bases in spectral space
        self.V = n_cheb.chebvander(self.points, N-3).T - ((k/(k+2))**2)[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 2:].T
        self.V = self.V[1:, :]

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        Chebyshev transform taking into account that phi_k = T_k - (k/(k+2))**2*T_{k+2}
        Note, this is the non-normalized scalar product
        """        
        if self.fast_transform:
            k  = self.wavenumbers(fj.shape)
            fk = self.fastChebScalar(fj, fk)
            fk[:-2] -= ((k/(k+2))**2) * fk[2:]
            fk[0] = 0

        else:
            if self.points is None: self.init(fj.shape[0])
            fk[1:-2] = np.dot(self.V, fj*self.weights)
            
        fk[-2:] = 0
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        if self.w_hat is None:
            self.w_hat = fk.copy()
        elif not self.w_hat.shape == fk.shape:
            self.w_hat = fk.copy()

        recreate = False
        if isinstance(self.factor, np.ndarray):
            if not self.factor.shape == fk.shape:
                recreate = True
            
        if self.factor is None:
            recreate = True
            
        if recreate:
            if len(fk.shape)==3:
                k = self.wavenumbers(fk.shape)                
            elif len(fk.shape)==1:
                k = self.wavenumbers(fk.shape[0])
            self.factor = (k[1:]/(k[1:]+2))**2
            
        self.w_hat[:] = 0
        self.w_hat[1:-2] = fk[1:-2]
        self.w_hat[3:] -= self.factor*fk[1:-2]
        fj = self.ifct(self.w_hat, fj)
        return fj

    def slice(self, N):
        return slice(1, N-2)

class ShenBiharmonicBasis(ShenDirichletBasis):
    
    def __init__(self, quad="GC", fast_transform=True):
        ShenDirichletBasis.__init__(self, quad, fast_transform)
        self.factor1 = None
        self.factor2 = None
        self.Solver = PDMA(quad)
        
    def init(self, N):
        self.points, self.weights = self.points_and_weights(N)
        k = self.wavenumbers(N)
        # Build Vandermonde matrix.
        self.V = n_cheb.chebvander(self.points, N-5).T - (2*(k+2)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-3)[:, 2:].T + ((k+1)/(k+3))[:, np.newaxis]*n_cheb.chebvander(self.points, N-1)[:, 4:].T
        
    def wavenumbers(self, N):
        if isinstance(N, tuple):
            if len(N) == 1:
                N = N[0]
        if isinstance(N, int): 
            return np.arange(N-4).astype(float)
        
        else:
            kk = np.mgrid[:N[0]-4, :N[1], :N[2]].astype(float)
            return kk[0]
        
    def set_factor_arrays(self, v):
        recreate = False
        if isinstance(self.factor1, np.ndarray):
            if not self.factor1.shape == v.shape:
                recreate = True
            
        if self.factor1 is None:
            recreate = True
            
        if recreate:
            if len(v.shape)==3:
                k = self.wavenumbers(v.shape)                
            elif len(v.shape)==1:
                k = self.wavenumbers(v.shape[0])
            self.factor1 = -2*(k+2)/(k+3)
            self.factor2 = (k+1)/(k+3)

    def fastShenScalar(self, fj, fk):
        """Fast Shen scalar product.
        """        
        if self.fast_transform:
            self.set_factor_arrays(fk)
            Tk = self.get_return_array(fk)
            Tk = self.fastChebScalar(fj, Tk)
            fk[:] = Tk[:]
            fk[:-4] += self.factor1 * Tk[2:-2]
            fk[:-4] += self.factor2 * Tk[4:]

        else:
            if self.points is None: self.init(fj.shape[0])
            fk[:-4] = np.dot(self.V, fj*self.weights)
            
        fk[-4:] = 0
        return fk

    def ifst(self, fk, fj):
        """Fast inverse Shen scalar transform
        """
        if self.w_hat is None:
            self.w_hat = fk.copy()
        elif not self.w_hat.shape == fk.shape:
            self.w_hat = fk.copy()
        self.set_factor_arrays(fk)
        self.w_hat[:] = 0
        self.w_hat[:-4] = fk[:-4]
        self.w_hat[2:-2] += self.factor1*fk[:-4]
        self.w_hat[4:]   += self.factor2*fk[:-4]
        fj = self.ifct(self.w_hat, fj)
        return fj
        
    def slice(self, N):
        return slice(0, N-4)
    
if __name__ == "__main__":
    from sympy import Symbol, sin, cos
    N = 16
    #a = np.random.random((N, N, N/2+1))+1j*np.random.random((N, N, N/2+1))
    #af = zeros((N, N, N/2+1), dtype=a.dtype)
    #a[0,:,:] = 0
    #a[-1,:,:] = 0
    #a = np.random.random(N)+np.random.random(N)*1j 
    #af = zeros(N, dtype=np.complex)
    #a[0] = 0
    #a[-1] = 0
    x = Symbol("x")
    f = (1-x**2)*cos(pi*x)    
    CT = ChebyshevTransform(quad="GC")
    points, weights = CT.points_and_weights(N)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = CT.fct(fj, u0)
    u1 = u0.copy()
    u1 = CT.ifct(u0, u1)
    assert np.allclose(u1, fj)    
    
    
    ST = ShenDirichletBasis(quad="GC", fast_transform=True)
    points, weights = ST.points_and_weights(N)
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = ST.fst(fj, u0)
    u1 = u0.copy()
    u1 = ST.ifst(u0, u1)
    assert np.allclose(u1, fj)
    
    SN = ShenNeumannBasis(quad="GL")
    points, weights = SN.points_and_weights(N)
    f = cos(pi*x)  # A function with f(+-1) = 0 and f'(+-1) = 0
    fj = np.array([f.subs(x, j) for j in points], dtype=float)
    fj -= np.dot(fj, weights)/weights.sum()
    u0 = SN.fst(fj, u0)
    u1 = u0.copy()
    u1 = SN.ifst(u0, u1)
    assert np.allclose(u1, fj)

    N = 30
    SB = ShenBiharmonicBasis(quad="GL", fast_transform=False)
    points, weights = SB.points_and_weights(N)
    f = (1-x**2)*sin(2*pi*x)    
    fj = np.array([f.subs(x, j) for j in points], dtype=float)    
    u0 = zeros(N, dtype=float)
    u0 = SB.fst(fj, u0)
    u1 = u0.copy()
    u1 = SB.ifst(u0, u1)
    assert np.allclose(u1, fj)
    

