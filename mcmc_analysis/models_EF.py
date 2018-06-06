#this is the analysis for the flux
import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.insert(0,'../Primarios')
import scipy.special as spc
from Ocentauri import interp
import scipy.integrate as integrate

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
    return No*(energy**(-1.0*Gamma))*np.exp((-1.0)*energy/energy_cut)

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
    return n_events

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
            val = E2dNdE_pulsar(pars,self.energy)
            return val

def flux_lnhood(pars,data,err,energy,model,col):
    ini_func = Flux(energy,model)
    H = ini_func(pars,col)

    p = (((data - H)**2) / err) + np.log(2. * np.pi * err)

    return -0.5*np.sum(p)

def event_lnhood(pars,data,e_min,e_max,model,col,expo,psf):
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
    elif model == 'pulsar':
        Gamma, log_E,log_N = pars

        if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and\
         plist[4]<log_N<plist[5]:
            return 0.0
        return -np.inf

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
