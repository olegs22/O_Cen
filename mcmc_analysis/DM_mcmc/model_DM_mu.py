import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from astropy.io import ascii
import sys
sys.path.insert(0,'../Primarios/')
#from Ocentauri import interp

channel_global = 'col10'
mass_global = 5

events, Ocen_exp, Ocen_psf, background = np.loadtxt('data/OC_no_events_9_bins.txt',usecols=(1,4,5,6),unpack=True)#9 bin data
Ps_exp1, Ps_psf1 = np.loadtxt('data/source2_J1326.txt',usecols=(2,3),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt('data/source2_J1328.txt',usecols=(2,3),unpack=True)#9 bin data

mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp1 = Ps_exp1[mask]
Ps_psf1 =Ps_psf1[mask]
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
#the semianalytic model for the mu's channel.
#this has the correct units.
def dnde(mass_x,energy):
    alpha = 1.0 / 137.
    s = 4.0 * mass_x**2
    y = energy / mass_x
    m_e = 0.107#Gev

    val = ((1.-(1.-y)**2)/y) * (np.log((s*(1.-y))/m_e**2) - 1.)
    return (alpha / (np.pi*mass_x)) * val

#model for the first source
def new_source_1(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 530.60 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

#model for the second source.
def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 10204.39 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

#here i calculate the number of events for DM
def no_events_DM(pars,e_min,e_max):
    mass_x, log_alphax = pars
    alpha_x = 10.**log_alphax

    integrand_dm = lambda x:dnde(mass_x,x)
    no_events = np.zeros(len(e_min))

    #this conditionals are to enforce energy conservation.
    for i in range(len(e_min)):
        if e_min[i]< mass_x <= e_max[i]:
	        no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        elif e_min[i]<mass_x and e_max[i]<mass_x:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp)/(8.0 * np.pi * mass_x**2)

def no_events_source1(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_1(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp1 * Ps_psf1 * n_events_ps

def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

def lnhood1(pars,data,e_min,e_max):

    Ocen_1 = no_events_DM(pars[:2],e_min,e_max)
    source1 = no_events_source1(pars[2:4],e_min,e_max)

    H1 = Ocen_1 + source1 + background
    #we use the stirling approximation for the log-factorial term
    p1 = data - H1 + data*np.log(H1/data)
    return np.sum(p1)

def lnhood2(pars,data,e_min,e_max):
    Ocen_1 = no_events_DM(pars[:2],e_min,e_max)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2 + background
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)

def priors(pars,plist):
    mass_x,log_alphax,log_Nn1,alpha1,log_Nn2,alpha2 = pars

    mu1 = 2.69463#1.69943 #
    sigma1 = 0.13373#0.13128 #

    mu2 = 1.69943
    sigma2 = 0.13128

    mu_n1 = 3.8873e-12
    sigma_n1 = 7.4243e-13

    mu_n2 = 1.13141e-14
    sigma_n2 = 1.7931e-15

    log_f1 = ((alpha1 - mu1)/sigma1)**2 + np.log(2.0*np.pi*sigma1**2)
    log_f2 = ((alpha2- mu2)/sigma2)**2 + np.log(2.0*np.pi*sigma2**2)

    Nn1 = 10.**log_Nn1
    Nn2 = 10.**log_Nn2
    log_fn1 = ((Nn1 - mu_n1)/sigma_n1)**2 + np.log(2.0*np.pi*sigma_n1**2)
    log_fn2 = ((Nn2 - mu_n2)/sigma_n2)**2 + np.log(2.0*np.pi*sigma_n2**2)

    if plist[0]<mass_x<plist[1] and plist[2]<log_alphax<plist[3] and\
       plist[4]<log_Nn1<plist[5] and plist[6]<alpha1<plist[7] and\
       plist[8]<log_Nn2<plist[9] and plist[10]<alpha2<plist[11]:
       return -0.5 * (log_f1 + log_f2 + log_fn1 + log_fn2)

    return -np.inf

def event_lnpost(pars,data,e_min,e_max,plist):
    pi = priors(pars,plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood1(pars,data,e_min,e_max) + lnhood2(pars,data,e_min,e_max)
