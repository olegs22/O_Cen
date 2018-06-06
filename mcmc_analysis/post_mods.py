import numpy as np
from scipy.interpolate import interp1d
import sys
sys.path.insert(0, '../Primarios')
from Ocentauri import interp

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

def lnhood(pars,data,err,energy,col):
    model = phi_primary(pars,energy,col)

    p = (((data - model)**2) / err) + np.log(2. * np.pi * err)

    return -0.5*np.sum(p)

def priors(pars,plist):
    mass,log_sig,log_J = pars

    if plist[0]<mass<plist[1] and plist[2]<log_sig<plist[3] and\
    plist[4]<log_J<plist[5]:
        return 0.0
    return -np.inf

def lnpost(pars,data,err,energy,plist,col):
    pi = priors(pars, plist)
    if not np.isfinite(pi):
        return -np.inf
    return pi+lnhood(pars,data,err,energy,col)
