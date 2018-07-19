#this is the analysis for the flux
import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.insert(0,'../Primarios')
import scipy.special as spc
from Ocentauri import interp
import scipy.integrate as integrate
from numpy.polynomial.polynomial import polyval

background = np.loadtxt('back_events.txt',usecols=(1),unpack=True) #this is the background for the 5 bins
#background = np.loadtxt('data/O_cen_data_Bck_no_events.txt',usecols=(1),unpack=True)[:7]

#background = np.loadtxt('data/O_cen_data_Bck_flux_true.txt',usecols=(1),unpack=True)[:7]
#background_flux = np.loadtxt('data/O_cen_data_Bck_flux.txt',usecols=(1),unpack=True)#non-isotropic
#background_flux = np.loadtxt('data/O_cen_data_Bck_flux2.txt',usecols=(1),unpack=True)[:7]#isotropic
#background_flux = np.loadtxt('data/O_cen_data_Bck_flux3.txt',usecols=(1),unpack=True)[:7]#los

Ocen_exp, Ocen_psf = np.loadtxt('O_cen_data.txt',usecols=(4,5),unpack=True)
#Ocen_exp, Ocen_psf = np.loadtxt('data/O_cen_data_no_events.txt',usecols=(4,5),unpack=True)
#Ocen_exp, Ocen_psf = Ocen_exp[:7], Ocen_psf[:7]

Ps_exp, Ps_psf = np.loadtxt('Source_data.txt',usecols=(4,5),unpack=True)
#Ps_exp, Ps_psf = np.loadtxt('data/O_cen_data_no_events_source.txt',usecols=(4,5),unpack=True)
#Ps_exp, Ps_psf = Ps_exp[:7], Ps_psf[:7]

def EdNdE_primary(mass,energy,col):
    #val1,val2 = interp(mass,col)
    a=interp1d(interp(mass,col)[0],interp(mass,col)[1],fill_value='extrapolate')
    return a(energy)

def phi_primary(pars,energy,col):
    mass,log_sig,log_J= pars

    sigmav = 10.**log_sig
    J = 10.**log_J
    probe = energy/(1000.)

    var = J*(1./(4.*np.pi))*sigmav*EdNdE_primary(mass,probe,col)/(2.*(mass**2))

    return energy*(1000.)*(1.602e-6)*var

def E2dNdE_pulsar(pars,energy):
    Gamma,log_E,log_N = pars
    #Gamma=0.7
    #energy_cut=1.2*1000.
    #No = 1e-11
    #Gamma = 10.**log_g
    energy_cut = 10.**log_E
    No = 10.**log_N

    #for the flux we have to multipy No by energy**2
    val = No*(energy**(-1.0*Gamma))*np.exp((-1.0)*energy/energy_cut)
    return val

def E2dNdE_flux(pars,energy):
    Gamma,log_E,log_N = pars
    #Gamma=0.7
    #energy_cut=1.2*1000.
    #No = 1e-11
    #Gamma = 10.**log_g
    energy_cut = 10.**log_E
    No = 10.**log_N

    #for the flux we have to multipy No by energy**2
    return No*(energy**(-1.0*Gamma))*np.exp((-1.0)*energy/energy_cut)


def pulsar_p_source(pars,energy):
    log_Nn,alpha = pars

    #alpha = 1.69943
    #mu,sigma = [1.69943,0.13128]
    Nn = 10.**log_Nn

    #for the flux we have to multipy No by energy**2
    val_2 = Nn*(energy**(-1.0*alpha))

    return val_2

def no_events_model_primary(pars,e_min,e_max,species,exposure,psf_energy):

    integrand= lambda x: phi_primary(pars, x, species)/x

    n_events=np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events[i]=integrate.quad(integrand,e_min[i]/1000.,e_max[i]/1000.)[0]*exposure[i]*psf_energy[i]
    return n_events


#No_events pulsar
def no_events_model_pulsar(pars,e_min,e_max):


    integrand = lambda x: E2dNdE_pulsar(pars,x)
    n_events=np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events[i]=integrate.quad(integrand,e_min[i],e_max[i])[0]
    return n_events * Ocen_exp * Ocen_psf

def no_events_pulsar_complete(pars,e_min,e_max):


    integrand = lambda x:E2dNdE_pulsar(pars[:3],x)
    n_events=np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events[i]=integrate.quad(integrand,e_min[i],e_max[i])[0]

    inte_n = lambda x:pulsar_p_source(pars[3:],x)
    events_n=np.zeros(len(e_min))

    for i in range(len(e_min)):
        events_n[i]=integrate.quad(inte_n,e_min[i],e_max[i])[0]

    #return n_events + events_n + background
    #return (n_events * Ocen_exp * Ocen_psf) + (events_n * Ps_exp * Ps_psf) + background
    return Ocen_exp * Ocen_psf * (n_events + events_n) + background

def flux_complete(pars,energy):
    Omega_cen_flux = E2dNdE_flux(pars[:3],energy) * energy**2
    Source_flux = pulsar_p_source(pars[3:],energy) * energy**2
    return 1.602e-6*(Omega_cen_flux + Source_flux) + background

class Events(object):
    def __init__(self, e_min,e_max, model):
        self.model = model
        self.E_min = e_min
        self.E_max = e_max


    def __call__(self,pars,*params):
        if self.model == 'primary':
            val = no_events_model_primary(pars,self.E_min,self.E_max,species,exposure,psf_energy)
            return val
        if self.model == 'pulsar':
            val = no_events_model_pulsar(pars,self.E_min,self.E_max)
            return val
        if self.model == 'p+b':
            val = no_events_pulsar_complete(pars,self.E_min,self.E_max)
            return val

class Flux(object):
    def __init__(self,energy, model_i='primary'):
        self.energy = energy
        self.model = model_i

    def __call__(self,pars,*parms):
        if self.model == 'primary':
            col = parms
            val = phi_primary(pars,self.energy,col)
            return val

        elif self.model == 'pulsar':
            val = E2E2dNdE_flux(pars,self.energy)
            return val
        elif self.model == 'p+b':
            val = flux_complete(pars,self.energy)
            return val

def flux_lnhood(pars,data,err,energy,model,col):
    ini_func = Flux(energy,model)
    H = ini_func(pars,col)

    p = (((data - H)**2) / err) + np.log(2. * np.pi * err)

    return -0.5*np.sum(p)

def event_lnhood(pars,data,e_min,e_max,model,col,expo,psf):

    #alpha = np.random.normal(1.69943,0.13128,1)
    ini_fun = Events(e_min,e_max,model)
    H = ini_fun(pars,col,expo,psf)

    #we use the stirling approximation for the log-factorial term
    p = data - H + data*np.log(H/data)
    #p = data*np.log(H) - H
    return np.sum(p)

def priors(pars,plist,model):
    if model == 'primary':
        mass,log_sig,log_J = pars

        if plist[0]<mass<plist[1] and plist[2]<log_sig<plist[3] and\
        plist[4]<log_J<plist[5]:
            return 0.0
        return -np.inf
    if model == 'pulsar':
        Gamma,log_E,log_N = pars

        if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and\
         plist[4]<log_N<plist[5]:
            return 0.0
        return -np.inf

    elif model == 'p+b':
        Gamma,log_E,log_N,log_Nn,alpha = pars

        mu = 1.69943
        sigma = 0.13128

        log_f = ((alpha - mu)/sigma)**2 + np.log(2.0*np.pi*sigma**2)

        if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and plist[4]<log_N<plist[5] and\
           plist[6]<log_Nn<plist[7] and plist[8]<alpha<plist[9]:
           return -0.5*log_f
        return -np.inf

        """
        mu = 1.69943
        sigma = 0.13128

        log_f = ((alpha - mu)/sigma)**2 + np.log(2.0*np.pi*sigma**2)


        return np.log(np.random.normal(mu,sigma))
        """
def flux_lnpost(pars,data,err,energy,plist,model,col):
    pi = priors(pars,plist,model)
    if not np.isfinite(pi):
        return -np.inf
    return pi+flux_lnhood(pars,data,err,energy,model,col)

def event_lnpost(pars,data,e_min,e_max,model,plist,col,expo,psf):
    pi = priors(pars,plist,model)
    if not np.isfinite(pi):
        return -np.inf
    return pi+event_lnhood(pars,data,e_min,e_max,model,col,expo,psf)
