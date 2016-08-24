"""Orr-Sommerfeld"""
from numpy.polynomial import chebyshev as n_cheb
from spectralDNS import config, get_solver, solve
from numpy import dot, real, pi, exp, sum, zeros, cos, exp, arange, imag, sqrt, array
from mpiFFT4py import dct
import matplotlib.pyplot as plt
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def initialize(U, U_hat, U_hat0, **context):  
    U_hat[:] = 0
    U_hat0[:] = 0
    U[:] = 0    
    if "KMM" in config.params.solver:
        context['g'][:] = 0
        context['H_hat1'][:] = 0
    else:
        context['P'][:] = 0
        context['P_hat'][:] = 0

def set_Source(Source, Sk, FST, ST, **context):
    Source[:] = 0
    Source[1, :] = -2./config.params.Re
    Sk[:] = 0
    Sk[1] = FST.fss(Source[1], Sk[1], ST)

def exact(x, Re, t, num_terms=400):
    beta = 2./Re
    u = zeros(len(x))
    for i in range(1, 2*num_terms, 2):
        lam_k = (2*i-1)*pi/2. 
        lam_kp = (2*(i+1)-1)*pi/2. 
        u[:] -= cos(lam_k*x)*exp(-config.params.nu*lam_k**2*t)/lam_k**3
        u[:] += cos(lam_kp*x)*exp(-config.params.nu*lam_kp**2*t)/lam_kp**3
    u *= (2*beta)/config.params.nu
    u += beta/2./config.params.nu*(1-x**2)
    return u

def reference(Re, t, num_terms=200):
    u = 1.0
    c = 1.0
    for n in range(1, 2*num_terms, 2):
        a = 32. / (pi**3*n**3)
        b = (0.25/Re)*pi**2*n**2
        c = -c
        u += a*exp(-b*t)*c
    return u

im1 = None
def update(context):
    global im1
    params = config.params
    solver = config.solver
    X = context.X
    
    if (params.tstep % params.plot_step == 0 and params.plot_step > 0 or
           params.tstep % params.compute_energy == 0):
        U = solver.get_velocity(**context)
        
    if im1 is None and solver.rank == 0 and params.plot_step > 0:
        plt.figure()
        im1 = plt.contourf(X[1,:,:,0], X[0,:,:,0], context.U[1,:,:,0], 100)
        plt.colorbar(im1)
        plt.draw()
        plt.pause(1e-6)
    
    if params.tstep % params.plot_step == 0 and solver.rank == 0 and params.plot_step > 0:
        im1.ax.clear()
        im1.ax.contourf(X[1, :,:,0], X[0, :,:,0], U[1, :, :, 0], 100) 
        im1.autoscale()
        plt.pause(1e-6)                

    if params.tstep % params.compute_energy == 0: 
        u0 = U[1, :, 0, 0].copy()
        uall = None
        if solver.rank == 0:
            uall = zeros((solver.num_processes, params.N[0]/solver.num_processes))
        solver.comm.Gather(u0, uall, root=0)
        
        if solver.rank == 0:
            uall = uall.reshape((params.N[0],))
            #x = x0
            #pc = zeros(len(x))
            #pc = ST.fct(uall, pc)  # Cheb transform of result
            #solution at x = 0
            #u = n_cheb.chebval(0, pc)
            u_exact = exact(x0, params.Re, params.t)
            #print u_exact-uall
            #u_exact = reference(params.Re, params.t)
            print "Time %2.5f Error %2.12e " %(params.t, sqrt(sum((u_exact-uall)**2)/params.N[0]))

def regression_test(context):
    solver = config.solver
    U = solver.get_velocity(**context)
    u0 = U[1, :, 0, 0].copy()
    uall = None
    if solver.rank == 0:
        uall = zeros((solver.num_processes, params.N[0]/solver.num_processes))

    solver.comm.Gather(u0, uall, root=0)
    if solver.rank == 0:
        uall = uall.reshape((params.N[0],))
        #x = x0
        #pc = zeros(len(x))
        #pc = ST.fct(uall, pc)  # Cheb transform of result
        #solution at x = 0
        #u = n_cheb.chebval(0, pc)
        #u_exact = reference(params.Re, params.t)
        u_exact = exact(x0, params.Re, params.t)
        print "Computed error = %2.8e %2.8e " %(sqrt(sum((uall-u_exact)**2)/params.N[0]), params.dt)

if __name__ == "__main__":
    config.update(
        {
        'Re': 800.,
        'nu': 1./800.,             # Viscosity
        'dt': 0.5,                 # Time step
        'T': 50.,                   # End time
        'L': [2, 2*pi, 4*pi/3.],
        'M': [6, 5, 2]
        },  "channel"
    )
    config.channel.add_argument("--compute_energy", type=int, default=5)
    config.channel.add_argument("--plot_step", type=int, default=10)
    solver = get_solver(update=update, regression_test=regression_test, mesh="channel")    
    context = solver.setup()
    initialize(**context)
    set_Source(**context)
    solve(solver, context)
