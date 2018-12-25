import numpy as np
#from scipy.linalg import solve_banded
import scipy.linalg as linalg

# timer for efficiency analysis
import time

# scipy code for forming sparse matrices & solving banded matrices
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

# script containing 
import rk_base

# plotting functions
import matplotlib.pyplot as plt
from matplotlib import animation

# fourier transform codes
from numpy.fft import rfft
from numpy.fft import rfftfreq

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class chain:
    """
    Model of pendulum chain, including functions for energy analysis
    """
    
    def __init__(self, N):
        """
        Parameters
        ----------
            N - number of masses on pendulum chain
        
        """
        self.N= N
        
        #preparing space
        self.theta, self.a, self.T= (np.zeros(N),)*3
        self.w = np.zeros(N,dtype='double') #increase overflow limit
        
        self.sep, self.s, self.c = (np.zeros(N-1),)*3
        
        #inital L matrix space with middle band filled
        self.L = np.insert(np.zeros((2, N)),1,2*np.ones(N),0)
        self.L[1,0]=1
        

    def __call__(self, l):
        """
        Parameters
        ----------
            l - combinatory vector of pendulum masses' positions and velocities
        """
        
        #unpacking state vector l
        self.theta, self.w = l[:self.N], l[self.N:]
        
        #solve for R (named as T for later replacement)
        self.T = self.w**2
        self.T[0] += np.cos(self.theta[0])
        
        #construct matrix L
        self.sep = np.diff(self.theta) #difference between adjacent angles
        self.s, self.c = np.sin(self.sep), np.cos(self.sep)
        self.L[0, 1:], self.L[2, :-1]= (-self.c,)*2
    
        #solve for T using scipy banded matrix function (overwrites R defined as T)
        linalg.solve_banded((1,1), self.L, self.T, overwrite_b=True, overwrite_ab = False, check_finite = False)
        
        #solve for a= D.T-sin(theta_0)e_0 (alpha) through scalar multiplications
        self.a[:-1], self.a[-1] = self.T[1:]*self.s, 0
        self.a[0] -= np.sin(self.theta[0])
        self.a[1:] -= self.T[:-1]*self.s
            
        return np.concatenate([self.w, self.a])


    def alt_f(self, l):
        """
        Alternative code for pendulum function using scipy sparse matrices
        Much much slower code
        
        also tested compatable with scipy's solve_ivp function
        
        Parameters
        ----------
            l - combinatory vector of pendulum masses' positions and velocities
        """
    
        self.update, self.theta, self.w = update, l[:self.N], l[self.N:]
        
        #solve for R
        self.R = self.w**2
        self.R[0] += np.cos(self.theta[0])
        
        # Construct sparse matrices L and D
        one = np.ones(self.N)
        two = 2*one
        two[0] = 1
        
        delta = sparse.dia_matrix(([-one,-one],[0,+1]),shape=(self.N,self.N))
        delta = sparse.csr_matrix(delta)
        
        self.sep = delta.dot(self.theta)
        self.s, self.c = np.sin(self.sep), np.cos(self.sep)
        
        D = sparse.dia_matrix(([self.s,-np.roll(self.s,1)],[-1,+1]),shape=(self.N,self.N))
        self.D = sparse.csr_matrix(D)
        
        L = sparse.dia_matrix(([-self.c,two,-np.roll(self.c,1)],[-1,0,+1]),shape=(self.N,self.N))
        self.L = sparse.csr_matrix(L)
        
        # Compute R
        R = np.array(self.w, dtype='f')**2
        R[0] += np.cos(self.theta[0])
        
        # Solve L.T = R
        self.T = spsolve(self.L,self.R)
        
        # Compute a
        self.alpha = D.dot(self.T)
        self.alpha[0] -= np.sin(self.theta[0])
        
        return np.concatenate([self.w,self.alpha])


    def coordinates(self):
        """
        calculate coordinates in cartesian plane, including the origin
        """
        return np.insert(np.cumsum(np.sin(self.theta)),0,0), np.insert(-np.cumsum(np.cos(self.theta)),0,0)
        
    def velocities(self):
        """
        calculate velocities in cartesian plane
        """
        return np.cumsum(self.w*np.cos(self.theta)), np.cumsum(self.w*np.sin(self.theta))
    
    def kpe(self, individual=False):
        """
        calculate KE and PE for individual or total of masses on pendulum chain
        """
        x, z = self.coordinates()
        vx, vz = self.velocities()
        
        if individual: 
            return 1/2*(vx**2+vz**2), z[1:]
        else:
            return np.sum(1/2*(vx**2+vz**2)), np.sum(z[1:])
    
    def total_energy(self):
        """
        calculate total energy of pendulum chain
        """
        ke, pe = self.kpe()
        return np.sum(ke + pe)
    
    def available_energy(self):
        """
        calculate available energy of pendulum chain
        """
        vx, vz = self.velocities()
        return np.abs(np.sum(1/2*(vx**2+vz**2-(self.N-np.arange(self.N))*(2*np.cos(self.theta/2))**2)))
    
    def perturb_fstill(self,p):
        """
        construct initial condition of all massess still with only the last mass with angular velocity w
        """
        self.theta = np.zeros(self.N)
        self.w = np.zeros(self.N)
        self.w[-1] = p
        return np.concatenate([self.theta,self.w])
    
    """
    #Code by Yuanming Tao, included for testing purposes
    def position_velocity(self, max_angle = 1, max_speed = 1, velocity = False):
    
        self.theta = np.pi - np.random.uniform(0, max_angle, self.N)
        self.omega = np.zeros(self.N)
        if velocity:
            self.omega =  np.random.uniform(0., max_speed, self.N)

        return np.concatenate([self.theta, self.omega])
    """

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class timestepper():
    """
    Time stepping method to solve for pendulum chain dynamics
    """
    
    def __init__(self, forcing, start, it = 0, time = 0, tol=1e-3, err=1e-16, min_dt=1e-2, max_dt=1.0, coeff=0.84, method="Cash_karp"):
        """
        Parameters
        ----------
            forcing - function in the derivative
            start   - initial state
            it      - iteration count
            time    - record of time as steps were taken
            tor     - estimate of max local truncation error
            min_dt  - minimum stepsize
            max_dt  - maximum stepsize
            coeff   - coefficient of danger
            err     - error
            method  - type of rk method used (CP default)
        """
        
        self.forcing, self.start, self.time, self.it = forcing, start, time, it
        
        self.tol, self.err, self.min_dt, self.max_dt, self.coeff = tol, err, min_dt, max_dt, coeff
        
        #loading tableu from rk_base.py file
        if method=="Cash_karp":
            self.tab_a, self.tab_b = np.array(rk_base.ck_a), np.array(rk_base.ck_b)
            self.s, self.array = 6, np.zeros((6, len(self.start)))
            
        elif method=="Fehlberg":
            self.tab_a, self.tab_b = np.array(rk_base.rf_a), np.array(rk_base.rf_b)
            self.s, self.array = 6, np.zeros((6, len(self.start)))
            
        elif method=="Dormand_prince":
            self.tab_a, self.tab_b = np.array(rk_base.dp_a), np.array(rk_base.dp_b)
            self.s, self.array = 7, np.zeros((7, len(self.start)))

    def __call__(self, dt):
        """
        Parameters
        ----------
            dt - step size
        """
        
        #the pendulum chain function
        f = self.forcing
        
        #update time and iteration count
        self.time += dt
        self.it += 1
        
        #load tableu
        tab_a = self.tab_a*dt
        tab_b = self.tab_b*dt
        
        #implement rk method chosen
        self.array[0] = f(self.start)
        
        for i in np.arange(1,self.s): 
            terms = tab_a[i-1,:i].dot(self.array[:i])
            self.array[i] = f(self.start+terms)
            
        self.start += tab_b[0].dot(self.array)
        
        #Compute optimised stepsize dt
        #method by Yuanming Tao, originating from Jonathan Senning, Gordon College. Website: http://www.math-cs.gordon.edu/courses/mat342/python/diffeq.py
        
        r = self.tol/self.err
        if r<1:
            dt = max([dt*self.coeff*r**(1/4), self.min_dt])
        else:
            dt = min([dt*self.coeff*r**(1/4), self.max_dt])
        
        self.err = np.max(np.abs(tab_b[1].dot(self.array)))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def solve(f, l, T=100, dt=0.01, coll_freq=50, method="Cash_karp"):
    """
    Parameters
    ----------
        l         - combinatory vector of positions & velocities
        T         - time range
        dt        - inital stepsize
        coll_freq - data collection frequency
        method    - other RK45 methods available: "Fehlberg" & "Dormand_prince" (insert these as strings)
    """
    
    #preparing space and inserting inital condition
    P = np.zeros((int(T/dt/coll_freq)+1, 2, f.N+1)) 
    P[0][0], P[0][1] = f.coordinates()
    times = np.zeros(int(T/dt/coll_freq)+1)
    
    start = time.time()
    
    #initialize timestepper
    ts = timestepper(f,l, method=method) 

    while ts.time <= T: 
        
        # Taking one step
        ts(dt) 
        
        if ts.it % coll_freq == 0:
            
            #obtaining cartesain corrdinates of each mass
            P[int(ts.it/coll_freq)][0], P[int(ts.it/coll_freq)][1] = f.coordinates()
            
            #obtaining time of each dataset
            times[int(ts.it/coll_freq)] = ts.time
            
    print("time elapsed -- %.1f seconds" % (time.time()-start))
    
    return P, times

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def dynamics(N, p, T=100, dt=0.01, coll_freq=50, energy=False, method="Cash_karp"):
    """
    Parameters
    ----------
        N         - number of masses on pendulum chain
        p         - perturbation of last mass for perturb_fstill
        T         - time range
        dt        - inital stepsize
        coll_freq - data collection frequency
        energy    - calculate energies for analysis
        method    - other RK45 methods available: "Fehlberg" & "Dormand_prince" (insert these as strings)
    """
    
    f = chain(N)
    l = f.perturb_fstill(p)
    #l = f.position_velocity()
    
    #preparing space and inserting inital condition
    P = np.zeros((int(T/dt/coll_freq)+1, 2, f.N+1)) 

    P[0][0], P[0][1] = f.coordinates()

    times = np.zeros(int(T/dt/coll_freq)+1)
    avail_en = np.zeros(int(T/dt/coll_freq)+1)
    total_en = np.zeros(int(T/dt/coll_freq)+1)
    total_ke = np.zeros(int(T/dt/coll_freq)+1)
    total_pe = np.zeros(int(T/dt/coll_freq)+1)
    
    indi_ke = np.zeros([int(T/dt/coll_freq)+1, N])
    indi_pe = np.zeros([int(T/dt/coll_freq)+1, N])
    
    start = time.time()
    
    #initialize timestepper
    ts = timestepper(f,l, method=method) 

    while ts.time <= T: 
        
        # Taking one step
        ts(dt) 
        
        if ts.it % coll_freq == 0:
            
            #obtaining cartesain corrdinates of each mass
            P[int(ts.it/coll_freq)][0], P[int(ts.it/coll_freq)][1] = f.coordinates()
            
            #obtaining time of each dataset
            times[int(ts.it/coll_freq)] = ts.time
            
            #obtaining total energies
            total_en[int(ts.it/coll_freq)] = f.total_energy()
            
            #obtaining total Kinetic and Potential energies
            total_ke[int(ts.it/coll_freq)], total_pe[int(ts.it/coll_freq)] = f.kpe()
            
            #obtaining available energies at state of data collection 
            avail_en[int(ts.it/coll_freq)] = f.available_energy()
            
            #obtaining individual Kinetic and Potential energies
            indi_ke[int(ts.it/coll_freq),:], indi_pe[int(ts.it/coll_freq),:] = f.kpe(individual=True)
            
            
    print("time elapsed -- %.1f seconds" % (time.time()-start))
    
    return P, times, total_en, total_ke, total_pe, avail_en, indi_ke, indi_pe

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def animate(P, time):
    """
    Basic animation of the pendulum chain dynamics
    
    Parameters
    ----------
        P     - solution that can be found using 'dynamics'
        time  - time corresponding to pendulum positions in P
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-P.shape[-1],P.shape[-1])
    ax.set_ylim(-P.shape[-1],P.shape[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    
    line, = ax.plot(P[0][0], P[0][1],'bo-', lw=1, markersize=3)
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
    
    def animate(i):
        line.set_xdata(P[i][0])
        line.set_ydata(P[i][1])  # update the data
    
        time_text.set_text(time_template % (time[i]))
        return line,
    
    ani = animation.FuncAnimation(fig, animate, np.arange(1, P.shape[0]-1),interval=time[-1], blit=True)
    
    plt.close()
    return ani
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def animate_path(P,time, tracks=[-1]):
    
    fig, ax = plt.subplots()
    ax.set_xlim(-P.shape[-1],P.shape[-1])
    ax.set_ylim(-P.shape[-1],P.shape[-1])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    
    line, = ax.plot([], [], 'bo-', lw=1, markersize = 2)
    track, = ax.plot([],[],'k.', color = "k", markersize = 1)
    
    time_template = 'time = %.1fs'
    time_text = ax.text(0.06, 0.85, '', transform=ax.transAxes)
    trackx, tracky = [], []
    
    def animate(i): 
        trackx.append(P[i,0,-1])
        tracky.append(P[i,1,-1])
        track.set_data(trackx, tracky)
        line.set_data(P[i][0],P[i][1])
        time_text.set_text(time_template % (time[i]))
        return track, line
    
    anip = animation.FuncAnimation(fig, animate, np.arange(1, P.shape[0]-1), interval=time[-1], blit=True)
    
    # ani.save('nlink_pendulum.mp4')
    plt.close()
    return anip

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def path(P, m):
    """
    Shows the path of an individual mass
    
    Parameters
    ----------
        P - solution that can be found using 'dynamics'
        m - index of mass of interest
    """
    fig, ax = plt.subplots()
    ax.plot(P[:,0,m],P[:,1,m],'lightgrey')
    ax.plot(P[:,0,m],P[:,1,m],'k.', markersize=2.5)
    ax.set(xlabel='x', ylabel='z', title= 'Path of {:}-th mass'.format(m))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def phasep(N, p, T=100, dt=0.01, coll_freq=50, method="Cash_karp", pf=[-1,-2]):
    """
    Parameters
    ----------
        N         - number of masses on pendulum chain
        p         - perturbation of last mass for perturb_fstill
        T         - time range
        dt        - inital stepsize
        coll_freq - data collection frequency
        method    - other RK45 methods available: "Fehlberg" & "Dormand_prince" (insert these as strings)
    """
    
    f = chain(N)
    l = f.perturb_fstill(p)

    #preparing space and inserting inital condition
    pp = np.zeros((int(T/dt/coll_freq)+1, 2, f.N)) 

    pp[0][0], pp[0][1] = f.theta, f.w
    
    ts = timestepper(f,l, method=method) 

    while ts.time <= T: 
        
        # Taking one step
        ts(dt) 
        
        if ts.it % coll_freq == 0:
            
            #obtaining angles and angular velocities
            pp[int(ts.it/coll_freq)][0], pp[int(ts.it/coll_freq)][1] = f.theta, f.w
          
    fig, ax = plt.subplots()  
    for i in pf:
        ax.plot(pp[:,0,i], pp[:,1,i], label='mass index {:}'.format(i))
        ax.set(ylabel='$\omega$', xlabel='$\Theta$')
    
    ax.legend()
 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plotter(N, times, e1, e2, estr1='KE', estr2='PE', method="FourierTransform"):
    """
    Helper function to plot Energies and their Fourier Transform/Power spectrum/log-log power spectrum in freq-amp space or log-log space
    
    Parameters
    ----------
        N      - number of masses on pendulum chain
        times  - time corresponding to chain positions recorded
        e1     - first energy
        e2     - second energy
        estr1  - name of first energy
        estr2  - name of second energy
        method - second plot type
    """
    
    if estr1=='Total':
        c1 = 'C0'
        c2 = 'g'
        c1_sub = 'C9'
        c2_sub = 'lightgreen'
    else:
        c1 = 'r'
        c2 = 'b'
        c1_sub = 'pink'
        c2_sub = 'lightblue'
        
    
    fig, ax = plt.subplots(2,2, figsize = (13,4))
    
    ax[0,0].plot(times, e1, c=c1)
    ax[0,0].set(title='Energies of {:}-mass'.format(N), ylabel=estr1)

    ax[1,0].plot(times, e2, c=c2)
    ax[1,0].set(ylabel=estr2, xlabel='Time (s)')


    avg_dt = np.mean(np.diff(times))
    amp_e1 = rfft(e1)
    amp_e2 = rfft(e2)
    freqs = rfftfreq(len(times), avg_dt)
    
    if method=='FourierTransform':
    
        ax[0,1].plot(freqs, amp_e1.real, c=c1, label = 'Real')
        ax[0,1].plot(freqs, amp_e1.imag, c=c1_sub, label = 'Imaginary')
        ax[0,1].set(title='Fourier transform', ylabel=estr1+ ' Amplitude')
        ax[0,1].legend()
        
        ax[1,1].plot(freqs, amp_e2.real, c=c2, label = 'Real')
        ax[1,1].plot(freqs, amp_e2.imag, c=c2_sub, label = 'Imaginary')
        ax[1,1].set(ylabel=estr2+' Amplitude', xlabel='Frequency (Hz)')
        ax[1,1].legend()
    
    if method=='PowerSpectrum':
        
        ax[0,1].plot(freqs, np.abs(amp_e1)**2, c=c1)
        ax[0,1].set(title='Power spectrum', ylabel=estr1+ ' Power')
        
        ax[1,1].plot(freqs, np.abs(amp_e2)**2, c=c2)
        ax[1,1].set(ylabel=estr2+ ' Power', xlabel='Frequency (Hz)')
    
    if method=='loglog':
       
        ax[0,1].loglog(freqs, np.abs(amp_e1)**2, c=c1)
        ax[0,1].set(title='Log-log power spectrum', ylabel=estr1+ ' log(power)')
        
        ax[1,1].loglog(freqs, np.abs(amp_e2)**2, c=c2)
        ax[1,1].set(ylabel=estr2+ ' log(power)', xlabel='log(frequency)')
    
    fig.tight_layout()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plotter2(N, times, e1, e2, estr1='KE', estr2='PE', method="FourierTransform"):
    """
    Helper function to plot Energies and their Fourier Transform/Power spectrum/log-log power spectrum in freq-amp space or log-log space togethor
    
    Parameters
    ----------
        N      - number of masses on pendulum chain
        times  - time corresponding to chain positions recorded
        e1     - first energy
        e2     - second energy
        estr1  - name of first energy
        estr2  - name of second energy
        method - second plot type
    """
    
    if estr1=='Total':
        c1 = 'C0'
        c2 = 'g'
        c1_sub = 'C9'
        c2_sub = 'lightgreen'
    else:
        c1 = 'r'
        c2 = 'b'
        c1_sub = 'pink'
        c2_sub = 'lightblue'
        
    
    fig, ax = plt.subplots(1,2, figsize = (13,4))
    
    ax[0].plot(times, e2, c=c2, label=estr2)
    ax[0].plot(times, e1, c=c1, label=estr1)
    ax[0].set(title='Energies of {:}-mass'.format(N), ylabel='Energy', xlabel='Time (s)')
    ax[0].legend()


    avg_dt = np.mean(np.diff(times))
    amp_e1 = rfft(e1)
    amp_e2 = rfft(e2)
    freqs = rfftfreq(len(times), avg_dt)
    
    if method=='FourierTransform':
        
        ax[1].plot(freqs, amp_e2.real, c=c2, label = 'Real '+estr2)
        ax[1].plot(freqs, amp_e1.real, c=c1, label = 'Real '+estr1)
        ax[1].plot(freqs, amp_e1.imag, c=c1_sub, label = 'Imaginary '+estr1)
        ax[1].plot(freqs, amp_e2.imag, c=c2_sub, label = 'Imaginary '+estr2)

        ax[1].set(title='Fourier transform', xlabel='Frequency (Hz)', ylabel='Amplitude')
        ax[1].legend()
    
    if method=='PowerSpectrum':
        
        ax[1].plot(freqs, np.abs(amp_e2)**2, c=c2, label=estr2)
        ax[1].plot(freqs, np.abs(amp_e1)**2, c=c1, label=estr1)
        ax[1].set(title='Power spectrum', ylabel='Power', xlabel='Frequency (Hz)')
        ax[1].legend()
    
    if method=='loglog':
       
        ax[1].loglog(freqs, np.abs(amp_e2)**2, c=c2, label=estr2)
        ax[1].loglog(freqs, np.abs(amp_e1)**2, c=c1, label=estr1)
        ax[1].set(title='Log-log power spectrum', ylabel='log(power)', xlabel='log(frequency)')
        ax[1].legend()
    
    fig.tight_layout()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def indi_anal(N, times, indi_ke, indi_pe, plter=2, methods=['FourierTransform','PowerSpectrum','loglog'], lis=[]):
    """
    Convenient plotter function helper to plot individual kinetic and potential energies of masses on the pendulum chain
    
    Parameters
    ----------
        N       - number of masses on pendulum chain
        times   - time corresponding to chain positions recorded
        indi_ke - array of individual kinetic energies
        indi_pe - array of individual potential energies
        plter   - choose which plotter function to use
        methods - second plot types as list
                    ('FourierTransform'/'PowerSpectrum'/'loglog')
        lis     - list of indexes for individual masses to plot
    """
    
    for i in lis:
        for m in methods:
            if plter==1:
                plotter(N, times[1:], indi_ke[1:,i], indi_pe[1:,i], estr1='KE{:}'.format(i), estr2='PE{:}'.format(i), method=m)
            else:
                plotter2(N, times[1:], indi_ke[1:,i], indi_pe[1:,i], estr1='KE{:}'.format(i), estr2='PE{:}'.format(i), method=m)
    
    
    