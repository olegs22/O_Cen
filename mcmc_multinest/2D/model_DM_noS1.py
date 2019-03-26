import numpy as np
import h5py
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pymultinest.solve import solve
import sys
import os

if not len(sys.argv) == 2:
    print("Sintaxis is python model_2D_gaussianpr.py path_datafile path_output_directory")
    exit()
else:
    outdir=sys.argv[1]
    if not os.path.exists(outdir): os.mkdir(outdir)

def plot_maps(MSP,S1,S2,back,data):
    plt.subplot(2, 2, 1)
    plt.imshow((S1 + S2+ back).reshape((20,20)),cmap='gist_gray',aspect='equal')
    plt.title("Sources+Background")
    plt.subplot(2, 2, 2)
    plt.imshow(MSP.reshape((20,20)),cmap='gist_gray',aspect='equal')
    plt.title("Omega_cent")
    plt.subplot(2, 2, 3)
    tmp=(MSP+ S1 + S2+ back).reshape((20,20))
    plt.title("Model")
    plt.imshow(tmp,cmap='gist_gray',aspect='equal')
    plt.subplot(2, 2, 4)
    plt.imshow(data.reshape((20,20)),cmap='gist_gray',aspect='equal')
    plt.title("Data")
    plt.show()
    plt.clf()
    return


path = '/Users/almagonzalez/Documents/projects/O_Cen/O_cen_2D/'
File_data_parameters = h5py.File(path+'gtbin_data.h5py','r')

#in File_data can also be fake data
File_data = h5py.File(path+'measure_data_real.h5py','r')

File_PSF =  h5py.File(path+'PSF_all_sources_inte.h5py','r')

psf_oc = File_PSF['psf_oc/psf'][()]
psf_s1 = File_PSF['psf_s1/psf'][()]
psf_s2 = File_PSF['psf_s2/psf'][()]

exposure_oc = File_data_parameters['exposure_oc/exposure'][()]
exposure_s1 = File_data_parameters['exposure_s1/exposure'][()]
exposure_s2 = File_data_parameters['exposure_s2/exposure'][()]

ebounds = File_data_parameters['ebounds/maps'][()]
naxis1 = File_data_parameters['naxis1/maps'][()]
naxis2 = File_data_parameters['naxis2/maps'][()]
naxis3 = File_data_parameters['naxis3/maps'][()]
binsz = File_data_parameters['binsz/maps'][()]*np.pi/180.

data = File_data['data/photons'][()]

Backgroundfile = path+'general_files/omegacen_fermibgmodel.txt'
energy_back,Isotropic,Diffuse=np.loadtxt(Backgroundfile,usecols=(0,1,2),unpack=True)
energy_back,Isotropic,Diffuse=energy_back*1000.,Isotropic/1000.,Diffuse/1000.

def dnde_qq(mass,energy):
    alpha1 = .95
    alpha2 = 6.5
    frac = energy/mass

    val = alpha1 * pow(frac,-0.5) * np.exp(-alpha2*frac)
    return val

def dnde_mu(mass,energy):
    alpha = 1/137
    print(alpha)
    s = 4.0 * mass**2
    frac = energy/mass
    m_e = 107.0 * 1e-3

    T_1 = (1.0 - (1.0-frac)**2) / frac
    T_2 = np.log(s*(1.0-frac)/m_e**2) - 1.0

    if T_1 < 0.0 or T_2 < 0.0:
        return 0.0
    elif np.isnan(T_1*T_2) == True:
        return 0.0
    else:
	    return (alpha / (np.pi*mass)) * T_1 * T_2

def no_events_qq(pars):
    mass_x, log_beta = pars
    beta = 10**log_beta

    no_events = np.zeros((naxis3,naxis1,naxis2))
    integrand = lambda x: dnde_qq(mass_x,x)

    for i in range(naxis3):
        no_events[i] = integrate.quad(integrand,ebounds[i][0]*1e-6,ebounds[i][1]*1e-6)[0] * exposure_oc[i] * np.ones((naxis1,naxis2))
    
    no_events = (no_events * beta * psf_oc) / (8.0*np.pi*mass_x**2)
    events_vec = np.zeros((15,int(20*20)))
    for i in range(15):
        events_vec[i] = no_events[i].flatten()
    
    return events_vec

def no_events_mu(pars):
    mass_x, log_beta = pars
    beta = 10**log_beta

    no_events = np.zeros((naxis3,naxis1,naxis2))
    integrand = lambda x: dnde_mu(mass_x,x)

    for i in range(naxis3):
        e_max = ebounds[i][1] * 1e-6
        e_min = ebounds[i][0] * 1e-6
        if e_min < mass_x <= e_max:
            no_events[i] = integrate.quad(integrand,e_min,mass_x)[0] * exposure_oc[i] * np.ones((naxis1,naxis2))
        
        elif (e_min < mass_x) and (e_max<mass_x):
            no_events[i] = integrate.quad(integrand,e_min,e_max)[0] * exposure_oc[i] * np.ones((naxis1,naxis2))
            
    no_events = (no_events * beta * psf_oc) / (8.0*np.pi*mass_x**2)
    events_vec = np.zeros((15,int(20*20)))
    
    for i in range(15):
        events_vec[i] = no_events[i].flatten()
    
    return events_vec

def no_events_source(pars,Ep,psf_source,psf_source_data):
    def Flux_PowerLaw(pars,energy,Ep):
        log_N,alpha = pars
        No = 10**log_N
        e=energy/Ep
        return No*(e**(-alpha))    
    
    #exposure=psf_s1_data[1].data.field(1)
    #ebounds = maps_data[1].data
    
    no_events=np.zeros((naxis3,naxis1,naxis2))
    integrand=lambda x: Flux_PowerLaw(pars,x,Ep)
    for k in range(naxis3):
        no_events[k]=integrate.quad(integrand,ebounds[k][0]/1000.,ebounds[k][1]/1000.)[0]\
        *psf_source_data[k]*np.ones((naxis1,naxis2))
    
    events_matrix = no_events*psf_source
    events_vec = np.zeros((15,int(20*20))) #change the binning
    for i in range(15):
        events_vec[i] = events_matrix[i].flatten()
    return events_vec

def no_events_Background():
    #ebounds=maps_data[1].data    #KeV
    #exposure=psf_s2_data[1].data.field(1)
    #binsz=maps_data[0].header['CDELT2']*np.pi/180.

    Back_int=interp1d(energy_back,Isotropic+Diffuse)
    Background_map=np.zeros((naxis3,naxis1,naxis2))
    integrand_Back=lambda x: Back_int(x)
    for i in range(naxis3):
        Background_map[i]=integrate.quad(integrand_Back,ebounds[i][0]/1000.\
                                      ,ebounds[i][1]/1000.)[0]\
        *exposure_oc[i]*binsz*binsz*np.ones((naxis1,naxis2))
    return Background_map.reshape((15,int(20*20))) #change the bining

def lnhood_mumu(pars):

    DM = no_events_mu(pars[:2])

    S1 = no_events_source(pars[2:4],530.6,psf_s2,exposure_s2) 
    S2 = no_events_source(pars[-2:],10204.39,psf_s1,exposure_s1)
    back = no_events_Background()

    p1 = np.zeros(15)
    
    for i in range(15):
        data_flat = data[i].flatten()
        mask = data_flat != 0.0

        data_flat = data_flat[mask]
        DM_mask = DM[i][mask]
        S1_mask = S1[i][mask] 
        S2_mask = S2[i][mask]
        back_mask = back[i][mask]
  
        H1_matrix = DM_mask + S2_mask + back_mask + S1_mask

        p1[i] = np.sum(data_flat - H1_matrix +\
             data_flat*np.log(H1_matrix/data_flat))
    
    return np.sum(p1)


def lnhood_qq(pars):
    DM = no_events_qq(pars[:2])
    S2 = no_events_source(pars[-2:],10204.39,psf_s1,exposure_s1)

    back = no_events_Background()

    p1 = np.zeros(15)

    for i in range(15):
        data_flat = data[i].flatten()
        mask = data_flat < 0
        DM_mask,S2_mask,back_mask =np.copy(DM[i]),np.copy(S2[i]),np.copy(back[i])
        data_flat[mask]=0.
        DM_mask[mask]=0.
        S2_mask[mask]=0.
        back_mask[mask]=0.
        """
        data_flat = data_flat[mask]
        DM_mask = DM[i][mask]
        S1_mask = S1[i][mask]
        S2_mask = S2[i][mask]
        back_mask = back[i][mask] """
        H1_matrix = DM_mask  + S2_mask + back_mask
        tmp= data_flat - H1_matrix +\
                       data_flat*np.log(H1_matrix/data_flat)
        s=np.isnan(tmp)
        p1[i] =tmp[~s].sum()
                       
    lnlike=np.sum(p1)
    return lnlike



def myprior(cube):
    mu2 = 1.9815
    sigma2 = 0.077
    mu_n2 = 3.8183e-14
    sigma_n2 = 3.9030e-15    #Priors from recent fermi catalog
    cube[0]=cube[0]*25    #m_xi
    cube[1]=cube[1]*(-10)  #log10(sigmav*J)
    cube[2]=np.log10(norm.ppf(cube[2], mu_n2, sigma_n2))   #Gaussian N1
    cube[3]=norm.ppf(cube[3],mu2,sigma2)   #Gaussian alpha_1

    return cube

if __name__ == "__main__":
    #print(np.shape(no_events_MSP([1.55812824,5.97957804,-9.3947961])))
    import json
    
    
    
    # number of dimensions our problem has
    parameters = ["m", "sigmav*J","logNn1","alpha"]
    n_params = len(parameters)
    # name of the output files
    prefix = outdir+"/1-"
    
    with open('%sparams.json' % prefix, 'w') as f:
        json.dump(parameters, f, indent=2)
    # run MultiNest
    result = solve(LogLikelihood=lnhood_qq, Prior=myprior,
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

