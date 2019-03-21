import numpy as np
import scipy.integrate as integrate

path = '../../O_cen_2D/P8R3/'
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + 'OC_no_events_15_bins_0.5_degree.txt',usecols=(1,4,5,6),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt(path + 'source2_J1328_0.5_degree.txt',usecols=(2,3),unpack=True)#9 bin data

mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 =Ps_psf2[mask]

"""
Ocen_exp = Ocen_exp[2:]
Ocen_psf = Ocen_psf[2:]
background = background[2:]
Ps_exp1 = Ps_exp1[2:]
Ps_psf1 =Ps_psf1[2:]
Ps_exp2 = Ps_exp2[2:]
Ps_psf2 =Ps_psf2[2:]
#path = '/Users/Oleg/Downloads/O_Cen-2/Primarios/'
#data_events = np.loadtxt('../Primarios/No_events_'+channel_global+'_mass_'+str(mass_global)+'.dat')
"""
def dnde(mass_x,energy):
    alpha1 = 0.95
    alpha2 = 6.5
    frac = energy/mass_x
 
    val = alpha1*pow(frac,-0.5)*np.exp(-alpha2*frac)
    return val

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 4726.70 * 1e-3#10204.39 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def no_events_DM(pars,e_min,e_max):
    mass_x, log_alphax = pars
    alpha_x = 10.**log_alphax

    integrand_dm = lambda x:dnde(mass_x,x)
    no_events = np.zeros(len(e_min))

    for i in range(len(e_min)):
        #if e_min[i]< mass_x <= e_max[i]:
	#    no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        #elif e_min[i]<mass_x and e_max[i]<mass_x:
        no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

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
