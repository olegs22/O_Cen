import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.insert(0,'../Primarios')
import scipy.special as spc
from Ocentauri import interp
import scipy.integrate as integrate
from numpy.polynomial.polynomial import polyval

path = '/Users/Oleg/Documents/O_Cen/O_cen_2D/Data_2018/'
#events, Ocen_exp, Ocen_psf, background = np.loadtxt('data/OC_no_events_9_bins.txt',usecols=(1,4,5,6),unpack=True)#9 bin data
#events, Ocen_exp, Ocen_psf, background = np.loadtxt('data/OC_no_events_15_bins_05_v2.txt',usecols=(1,4,5,6),unpack=True)#15 bin data
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + 'OC_no_events_15_bins_0.5_degree.txt',usecols=(1,4,5,6),unpack=True)#15 bin data

#Ps_exp1, Ps_psf1 = np.loadtxt('data/source2_J1326.txt',usecols=(2,3),unpack=True)#9 bin data
#Ps_exp2, Ps_psf2 = np.loadtxt('data/source2_J1328.txt',usecols=(2,3),unpack=True)#9 bin data
Ps_exp1, Ps_psf1 = np.loadtxt(path + 'source2_J1326_15_bins_05_v2.txt',usecols=(2,3),unpack=True)#15 bin data
Ps_exp2, Ps_psf2 = np.loadtxt(path + 'source2_J1328_0.5_degree.txt',usecols=(2,3),unpack=True)#15 bin data
#Ps_exp2, Ps_psf2 = np.loadtxt('data/source2_J1328_15_bins_05_v2.txt',usecols=(2,3),unpack=True)#15 bin data

mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp1 = Ps_exp1[mask]
Ps_psf1 =Ps_psf1[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 =Ps_psf2[mask]

def E2dNdE_pulsar(pars,energy):
    Gamma,log_E,log_N = pars

    energy_cut = 10.**log_E
    No = 10.**log_N

    #for the flux we have to multipy No by energy**2
    val = No*(energy**(-1.0*Gamma))*np.exp((-1.0)*energy/energy_cut)
    return val

def pulsar_p_source(pars,energy):
    log_Nn,alpha = pars

    #alpha = 1.69943
    #mu,sigma = [1.69943,0.13128]
    Nn = 10.**log_Nn

    #for the flux we have to multipy No by energy**2
    val_2 = Nn*(energy**(-1.0*alpha))

    return val_2

def new_source_1(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 530.60
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 4726.70#10204.39
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val


def no_events_ocen(pars,e_min,e_max):

    integrand_oc = lambda x:E2dNdE_pulsar(pars,x)

    n_events_oc = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_oc[i] = integrate.quad(integrand_oc,e_min[i],e_max[i])[0]

    return (Ocen_exp * Ocen_psf * n_events_oc)  + background


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

    Ocen_1 = no_events_ocen(pars[:3],e_min,e_max)
    source1 = no_events_source1(pars[3:5],e_min,e_max)

    H1 = Ocen_1 + source1
    #we use the stirling approximation for the log-factorial term
    p1 = data - H1 + data*np.log(H1/data)
    return np.sum(p1)

def lnhood2(pars,data,e_min,e_max):
    Ocen_1 = no_events_ocen(pars[:3],e_min,e_max)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)

def priors1(pars,plist):
    Gamma,log_E,log_N,log_Nn1,alpha1,log_Nn2,alpha2  = pars

    mu1 = 2.69463#1.69943 #
    sigma1 = 0.13373#0.13128 #

    mu2 = 1.9815#1.69943
    sigma2 = 0.077#0.13128

    mu_n1 = 3.8873e-12
    sigma_n1 = 7.4243e-13

    mu_n2 = 3.8183e-14#1.13141e-14
    sigma_n2 = 3.9030e-15#1.7931e-15

    log_f1 = ((alpha1 - mu1)/sigma1)**2 + np.log(2.0*np.pi*sigma1**2)
    log_f2 = ((alpha2- mu2)/sigma2)**2 + np.log(2.0*np.pi*sigma2**2)

    Nn1 = 10.**log_Nn1
    Nn2 = 10.**log_Nn2
    log_fn1 = ((Nn1 - mu_n1)/sigma_n1)**2 + np.log(2.0*np.pi*sigma_n1**2)
    log_fn2 = ((Nn2 - mu_n2)/sigma_n2)**2 + np.log(2.0*np.pi*sigma_n2**2)

    if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and plist[4]<log_N<plist[5] and\
       plist[6]<log_Nn1<plist[7] and plist[8]<alpha1<plist[9] and plist[10]<log_Nn2<plist[11] and\
        plist[12]<alpha2<plist[13]:
       return -0.5 * (log_f1 + log_f2 + log_fn1 + log_fn2)
    return -np.inf


def event_lnpost(pars,data,e_min,e_max,plist):
    pi = priors1(pars,plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood1(pars,data,e_min,e_max) + lnhood2(pars,data,e_min,e_max)
