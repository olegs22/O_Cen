import numpy as np
import scipy.integrate as integrate


path = '/Users/Oleg/Documents/O_Cen/O_cen_2D/P8R3/'
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + 'OC_no_events_15_bins_0.2_degree.txt',usecols=(1,4,5,6),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt(path + 'source2_J1328_0.2_degree.txt',usecols=(2,3),unpack=True)#9 bin data
J_table = np.loadtxt(path + 'J_factor_convolution/J_factor_conv_table.dat')

mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 = Ps_psf2[mask]
#J_factor = J_factor[mask]
#J_factor = 10.**J_factor

def dnde(mx,energy):
    alpha=1./137.
    y=energy/mx
    s=4.0*mx**2
    m_mu = 107 * 1.0e-3
    term_1=(1.0+(1.0-y)**2)/y
    term_2=np.log(s*(1-y)/m_mu/m_mu)
    if term_1 <0 or term_2<0:
        return 0.
    return alpha*term_1*term_2/np.pi/mx

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 4726.70 * 1e-3#10204.39 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def no_events_DM(pars,e_min,e_max,J_index):
    mass_x, log_alphax = pars
  
    alpha_x = 10.**log_alphax

    integrand_dm = lambda x:dnde(mass_x,x)
    no_events = np.zeros(len(e_min))
    J_factor = J_table[J_index]
    J_factor = J_factor[mask]
    J_factor = 10.**J_factor

    for i in range(len(e_min)):
        if e_min[i]< mass_x <= e_max[i]:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        elif e_min[i]<mass_x and e_max[i]<mass_x:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp * J_factor)/(8.0 * np.pi * mass_x**2)

def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

def lnhood2(pars,data,e_min,e_max,index):
    Ocen_1 = no_events_DM(pars[:2],e_min,e_max,index)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2 + background
    #print H2/data
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

def event_lnpost(pars,data,e_min,e_max,plist,index):
    pi = priors(pars,plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood2(pars,data,e_min,e_max,index)
