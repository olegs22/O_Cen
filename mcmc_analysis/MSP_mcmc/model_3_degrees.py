import numpy as np
from scipy.interpolate import interp1d
import sys
import scipy.special as spc
import scipy.integrate as integrate
from numpy.polynomial.polynomial import polyval

path = '/Users/Oleg/Documents/O_Cen/O_cen_2D/P8R3/'
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + 'OC_no_events_15_bins_3_degree.txt',usecols=(1,4,5,6),unpack=True)#15 bin data

Ps_exp2, Ps_psf2 = np.loadtxt(path + 'source2_J1328_3_degree.txt',usecols=(2,3),unpack=True)#15 bin data
#new sources 
Ps_exp_new1, Ps_psf_new1 = np.loadtxt(path+'new_source_J136_3_degree.txt',usecols=(2,3),unpack=True)
Ps_exp_new2, Ps_psf_new2 = np.loadtxt(path+'new_source_J1318_3_degree.txt',usecols=(2,3),unpack=True)
Ps_exp_new3, Ps_psf_new3 = np.loadtxt(path+'new_source_J1335_3_degree.txt',usecols=(2,3),unpack=True)




mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]

Ps_exp2 = Ps_exp2[mask]
Ps_exp_new1 = Ps_exp_new1[mask]
Ps_exp_new2 = Ps_exp_new2[mask]
Ps_exp_new3 = Ps_exp_new3[mask]

Ps_psf2 =Ps_psf2[mask]
Ps_psf_new1 = Ps_psf_new1[mask]
Ps_psf_new2 = Ps_psf_new2[mask]
Ps_psf_new3 = Ps_psf_new3[mask]

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

def source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 4726.70#10204.39
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def new_source_1(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 1318.53
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 1293.54
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def new_source_3(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 1330.36
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def no_events_ocen(pars,e_min,e_max):

    integrand_oc = lambda x:E2dNdE_pulsar(pars,x)

    n_events_oc = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_oc[i] = integrate.quad(integrand_oc,e_min[i],e_max[i])[0]

    return (Ocen_exp * Ocen_psf * n_events_oc)  + background


def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

def no_events_new_sources(pars,e_min,e_max):
    log_ns_N1,alpha_ns_1,log_ns_N2,alpha_ns_2,log_ns_N3,alpha_ns_3 = pars
    
    integrand_ns_1 = lambda x:new_source_1([log_ns_N1,alpha_ns_1],x)
    integrand_ns_2 = lambda x:new_source_2([log_ns_N2,alpha_ns_2],x)
    integrand_ns_3 = lambda x:new_source_3([log_ns_N3,alpha_ns_3],x)

    n_events_1 = np.zeros(len(e_min))
    n_events_2 = np.zeros(len(e_min))
    n_events_3 = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_1[i] = integrate.quad(integrand_ns_1,e_min[i],e_max[i])[0]
        n_events_2[i] = integrate.quad(integrand_ns_2,e_min[i],e_max[i])[0]
        n_events_3[i] = integrate.quad(integrand_ns_3,e_min[i],e_max[i])[0]

    val_1 = Ps_exp_new1 * Ps_psf_new1 * n_events_1
    val_2 = Ps_exp_new2 * Ps_psf_new2 * n_events_2
    val_3 = Ps_exp_new3 * Ps_psf_new3 * n_events_3

    return val_1 + val_2 + val_3

def lnhood2(pars,data,e_min,e_max):
    Ocen_1 = no_events_ocen(pars[:3],e_min,e_max)
    source2 = no_events_source2(pars[3:5],e_min,e_max)
    new_sources = no_events_new_sources(pars[5:],e_min,e_max)

    H2 = Ocen_1 + source2 + new_sources
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)

def priors1(pars,plist):
    Gamma,log_E,log_N,log_Ns2,alphas2,log_ns_N1,alpha_ns1,log_ns_N2,alpha_ns2,log_ns_N3,alpha_ns3 = pars

    mu_alpha = 1.9815#1.69943
    sigma_alpha = 0.077#0.13128
    G_prior_alpha = ((alphas2- mu_alpha)/sigma_alpha)**2 + np.log(2.0*np.pi*sigma_alpha**2)

    mu_Ns2 = 3.8183e-14#1.13141e-14
    sigma_Ns2 = 3.9030e-15#1.7931e-15
    Ns2 = 10.**log_Ns2
    G_prior_Ns2 = ((Ns2 - mu_Ns2)/sigma_Ns2)**2 + np.log(2.0*np.pi*sigma_Ns2**2)

    mu_alpha_ns1 = 2.3826
    mu_alpha_ns2 = 2.5564
    mu_alpha_ns3 = 2.3881
    sigma_alpha_ns1 = 0.1439
    sigma_alpha_ns2 = 0.1668
    sigma_alpha_ns3 = 0.1269
    
    G_prior_alpha_ns1 = ((alpha_ns1- mu_alpha_ns1)/sigma_alpha_ns1)**2 + np.log(2.0*np.pi*sigma_alpha_ns1**2)
    G_prior_alpha_ns2 = ((alpha_ns2- mu_alpha_ns2)/sigma_alpha_ns2)**2 + np.log(2.0*np.pi*sigma_alpha_ns2**2)
    G_prior_alpha_ns3 = ((alpha_ns3- mu_alpha_ns3)/sigma_alpha_ns3)**2 + np.log(2.0*np.pi*sigma_alpha_ns3**2)

    mu_N_ns1 = 1.7783e-13
    mu_N_ns2 = 1.9094e-13
    mu_N_ns3 = 1.9818e-13
    sigma_N_ns1 = 3.4861e-14
    sigma_N_ns2 = 3.8053e-14
    sigma_N_ns3 = 3.6012e-14
    N_ns1 = 10.**log_ns_N1
    N_ns2 = 10.**log_ns_N2
    N_ns3 = 10.**log_ns_N3

    G_prior_N_ns1 = ((N_ns1 - mu_N_ns1)/sigma_N_ns1)**2 + np.log(2.0*np.pi*sigma_N_ns1**2)
    G_prior_N_ns2 = ((N_ns2 - mu_N_ns2)/sigma_N_ns2)**2 + np.log(2.0*np.pi*sigma_N_ns2**2)
    G_prior_N_ns3 = ((N_ns3 - mu_N_ns3)/sigma_N_ns3)**2 + np.log(2.0*np.pi*sigma_N_ns3**2)

    if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and plist[4]<log_N<plist[5] and\
       plist[6]<log_Ns2<plist[7] and plist[8]<alphas2<plist[9] and plist[10]<log_ns_N1<plist[11] and\
       plist[12]<alpha_ns1<plist[13] and plist[14]<log_ns_N2<plist[15] and plist[16]<alpha_ns2<plist[17] and\
       plist[18]<log_ns_N3<plist[19] and plist[20]<alpha_ns3<plist[21]:
       
        prior_vals = G_prior_alpha + G_prior_Ns2 + G_prior_alpha_ns1 + G_prior_alpha_ns2 + G_prior_alpha_ns3 +\
                     G_prior_N_ns1 + G_prior_N_ns2 + G_prior_N_ns3

        return -0.5 * prior_vals
    return -np.inf


def event_lnpost(pars,data,e_min,e_max,plist):
    pi = priors1(pars,plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood2(pars,data,e_min,e_max)
