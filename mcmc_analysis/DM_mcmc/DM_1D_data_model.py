import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

#path = '/Users/Oleg/Documents/O_Cen/O_cen_2D/P8R3/'
path = '../../O_cen_2D/P8R3/'
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + 'OC_no_events_15_bins_0.2_degree.txt',usecols=(1,4,5,6),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt(path + 'source2_J1328_0.2_degree.txt',usecols=(2,3),unpack=True)#9 bin data
#dnde = np.loadtxt('/Users/Oleg/Documents/O_Cen/O_cen_2D/data/qq_tables/qq_pythia_spec.txt')
dnde = np.loadtxt(path + 'qq_pythia_spec.txt')
#dnde = np.loadtxt(path + 'spectrum_bb.txt')

masses_x = np.arange(5,50+0.01,0.01)

mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 =Ps_psf2[mask]

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 4726.70 * 1e-3#10204.39 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def no_events_DM(pars,e_min,e_max):
    mass_x, log_alphax = pars
    #J = 21.0
    #sigJ = log_alphax + J
    alpha_x = 10.**log_alphax

    mass_index = np.searchsorted(masses_x,mass_x)
    if mass_index >= len(dnde):
        mass_index -= 1
    
    complete_energy = np.linspace(0.1,mass_x,len(dnde[mass_index-1]))
    dnde_interp = interp1d(complete_energy,dnde[mass_index-1])
    no_events = np.zeros(len(e_min))
    for i in range(len(e_min)):
        if e_min[i]< mass_x <= e_max[i]:
            no_events[i] = integrate.quad(dnde_interp,e_min[i],masses_x[mass_index-1])[0]
        elif e_min[i]<mass_x and e_max[i]<mass_x:
            no_events[i] = integrate.quad(dnde_interp,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp * Ocen_psf)/(8.0 * np.pi * mass_x**2)

def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

def lnhood2(pars,data,e_min,e_max):
    Ocen_1 = no_events_DM(pars[:2],e_min,e_max)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2 + background
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)

def priors(pars,plist):
    mass_x,log_alphax,log_Nn2,alpha2 = pars

    mu2 = 1.9815#1.69943
    sigma2 = 0.077#0.13128

    mu_n2 = 3.8183e-14#1.13141e-14
    sigma_n2 = 3.9030e-15#1.7931e-15

    log_f2 = ((alpha2- mu2)/sigma2)**2 + np.log(2.0*np.pi*sigma2**2)

    Nn2 = 10.**log_Nn2
    log_fn2 = ((Nn2 - mu_n2)/sigma_n2)**2 + np.log(2.0*np.pi*sigma_n2**2)

    if plist[0]<mass_x<plist[1] and plist[2]<log_alphax<plist[3] and\
       plist[4]<log_Nn2<plist[5] and plist[6]<alpha2<plist[7]:
       return -0.5 * (log_f2 + log_fn2)

    return -np.inf

def event_lnpost(pars,data,e_min,e_max,plist):
    pi = priors(pars,plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood2(pars,data,e_min,e_max)
