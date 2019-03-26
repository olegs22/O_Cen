import numpy as np
from scipy.interpolate import interp1d
import sys,os
import scipy.special as spc
import scipy.integrate as integrate
from numpy.polynomial.polynomial import polyval

from scipy.stats import norm
from pymultinest.solve import solve

path = '/Users/almagonzalez/Documents/projects/O_Cen/data/'
data_file='OC_no_events_15_bins_0.5_degree.txt'
source_file='source2_J1328_0.5_degree.txt'
events, Ocen_exp, Ocen_psf, background = np.loadtxt(path + data_file,usecols=(1,4,5,6),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt(path + source_file,usecols=(2,3),unpack=True)#9 bin data

Ener,no_events,e_min,e_max = np.loadtxt(path + data_file,usecols=(0,1,2,3),unpack=True)


mask = events != 0.0
Ocen_exp = Ocen_exp[mask]
Ocen_psf = Ocen_psf[mask]
background = background[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 =Ps_psf2[mask]
Ener=Ener[mask]
no_events=no_events[mask]
e_min=e_min[mask]
e_max=e_max[mask]


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
"""
def new_source_1(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 530.60
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val
"""
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

"""
def no_events_source1(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_1(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp1 * Ps_psf1 * n_events_ps
"""
def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps
"""
def lnhood1(pars,data,e_min,e_max):

    Ocen_1 = no_events_ocen(pars[:3],e_min,e_max)
    source1 = no_events_source1(pars[3:5],e_min,e_max)

    H1 = Ocen_1 + source1
    #we use the stirling approximation for the log-factorial term
    p1 = data - H1 + data*np.log(H1/data)
    return np.sum(p1)
"""
def lnhood2(pars,data=no_events,e_min=e_min,e_max=e_max):
    Ocen_1 = no_events_ocen(pars[:3],e_min,e_max)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)


def myprior(cube):
    mu_alpha2 = 1.9815
    sigma_alpha2 = 0.077
    mu_n2 = 3.8183e-14
    sigma_n2 = 3.9030e-15    #Priors from recent fermi catalog
    cube[0]=cube[0]*10    #Gamma
    cube[1]=cube[1]*10  #Flat prior only around the Ecut found in the 1D analysis.
    cube[2]=cube[2]*(-10)  #N0
    cube[3]=np.log10(norm.ppf(cube[3], mu_n2, sigma_n2))
    cube[4]=norm.ppf(cube[4],mu_alpha2,sigma_alpha2)
    return cube


if __name__ == "__main__":
    #print(np.shape(no_events_MSP([1.55812824,5.97957804,-9.3947961])))
    import json
    
    if not len(sys.argv) == 2:
        print("Sintaxis is python script path_output_directory")
        exit()
    else:
        outdir=sys.argv[1]
        if not os.path.exists(outdir): os.mkdir(outdir)
    
    # number of dimensions our problem has
    parameters = ["Gamma", "Ecut","logN","logNn2","alpha2"]
    n_params = len(parameters)
    # name of the output files
    prefix = outdir+"/1-"
    
    with open('%sparams.json' % prefix, 'w') as f:
        json.dump(parameters, f, indent=2)
    # run MultiNest
    result = solve(LogLikelihood=lnhood2, Prior=myprior,
                   n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
    print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/3-
# For that, we need to store the parameter names:
