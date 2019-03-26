import numpy as np
import scipy.integrate as integrate
from scipy.stats import norm
from pymultinest.solve import solve
import sys,os

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
e_min=e_min[mask]/1000.
e_max=e_max[mask]/1000.


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
    alpha = 1.0 / 137.
    s = 4.0 * mass_x**2
    y = energy / mass_x
    m_e = 0.107#Gev
    T_1 = ((1.-(1.-y)**2)/y)
    T_2 = (np.log(s*(1.-y)/m_e**2) - 1.)
    if T_1 < 0.0 or T_2 < 0.0:
        return 0.0
    elif np.isnan(T_1*T_2) == True:
        return 0.0
        #val = ((1.-(1.-y)**2)/y) * (np.log(s*(1.-y)/m_e**2) - 1.)
    else:
        return (alpha / (np.pi*mass_x)) * T_1 * T_2

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
        if e_min[i]< mass_x <= e_max[i]:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        elif e_min[i]<mass_x and e_max[i]<mass_x:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp * Ocen_psf)/(8.0 * np.pi * mass_x**2)

def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

def lnhood2(pars,data=no_events,e_min=e_min,e_max=e_max):
    Ocen_1 = no_events_DM(pars[:2],e_min,e_max)
    source2 = no_events_source2(pars[-2:],e_min,e_max)

    H2 = Ocen_1 + source2 + background
    #print H2/data
    p2 = data - H2 + data*np.log(H2/data)
    return np.sum(p2)



def myprior(cube):
    mu_alpha2 = 1.9815
    sigma_alpha2 = 0.077
    mu_n2 = 3.8183e-14
    sigma_n2 = 3.9030e-15    #Priors from recent fermi catalog
    cube[0]=cube[0]*25    #m_xi
    cube[1]=cube[1]*(-20)  #log10(sigmav*J)
    cube[2]=np.log10(norm.ppf(cube[2], mu_n2, sigma_n2))   #Gaussian N2
    cube[3]=norm.ppf(cube[3],mu_alpha2,sigma_alpha2)   #Gaussian alpha_2
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
    parameters = ["m", "sigmav*J","logNn2","alpha2"]
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
