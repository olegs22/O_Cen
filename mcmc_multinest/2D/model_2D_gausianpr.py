from __future__ import absolute_import, unicode_literals, print_function


import numpy as np
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import units as u
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import norm

import numpy
from numpy import pi, cos
from pymultinest.solve import solve
import sys
import os



if not len(sys.argv) == 2:
    print("Sintaxis is python model_2D_gaussianpr.py path_datafile path_output_directory")
    exit()
else:
    outdir=sys.argv[1]
    if not os.path.exists(outdir): os.mkdir(outdir)





#import emcee

path='/Users/almagonzalez/Documents/projects/O_Cen/O_cen_2D/'

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


def plot_maps(MSP,S1,S2,back,data):
    plt.subplot(2, 2, 1)
    plt.imshow((S1 + S2+ back).reshape((20,20)),cmap='rainbow_r',norm=colors.Normalize(vmin=0, vmax=1.0),aspect='equal')
    plt.title("Sources+Background")
    plt.subplot(2, 2, 2)
    plt.imshow(MSP.reshape((20,20)),cmap='rainbow_r',norm=colors.Normalize(vmin=0, vmax=1.0),aspect='equal')
    plt.title("Omega_cent")
    plt.subplot(2, 2, 3)
    tmp=(MSP+ S1 + S2+ back).reshape((20,20))
    plt.title("Model")
    plt.imshow(tmp,cmap='rainbow_r',norm=colors.Normalize(vmin=0, vmax=1.0),aspect='equal')
    plt.subplot(2, 2, 4)
    plt.imshow(data.reshape((20,20)),cmap='rainbow_r',norm=colors.Normalize(vmin=0, vmax=1.0),aspect='equal')
    plt.title("Data")
    plt.show()
    plt.clf()
    return

def Flux_MSP(pars,energy):
    Gamma,log_E,log_No = pars

    No = 10**log_No
    E_cut = 10**log_E

    return No*(energy**(-Gamma))*np.exp(-energy/E_cut)

def no_events_MSP(pars):
    
    #exposure=psf_oc_d[1].data.field(1)
    #ebounds=gtbin_data[1].data    #KeV
    #ebounds = maps_data[1].data
    
    no_events=np.zeros((naxis3,naxis1,naxis2))
    integrand=lambda x: Flux_MSP(pars,x)
    for k in range(naxis3):
        no_events[k]=integrate.quad(integrand,ebounds[k][0]/1000.,ebounds[k][1]/1000.)[0]\
        *exposure_oc[k]*np.ones((naxis1,naxis2))

    event_matrix = no_events*psf_oc
    events_vec = np.zeros((15,int(20*20)))
    for i in range(15):
        events_vec[i] = event_matrix[i].flatten()
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
    events_vec = np.zeros((15,int(20*20)))
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
        *exposure_s2[i]*binsz*binsz*np.ones((naxis1,naxis2))
    return Background_map.reshape((15,int(20*20)))

def lnhood(pars):
    MSP = no_events_MSP(pars[:3])
    S1 = no_events_source(pars[3:5],530.6,psf_s2,exposure_s2) 
    S2 = no_events_source(pars[-2:],10204.39,psf_s1,exposure_s1)
    back = no_events_Background()
    
    p1 = np.zeros(15)
    p2 = np.zeros(15)
    for i in range(15):
        data_flat = data[i].flatten()
        mask = data_flat <0
        MSP_mask,S1_mask,S2_mask,back_mask =np.copy(MSP[i]),np.copy(S1[i]),np.copy(S2[i]),np.copy(back[i])
        data_flat[mask]=0.
        MSP_mask[mask]=0.
        S1_mask[mask]=0.
        S2_mask[mask]=0.
        back_mask[mask]=0.
        #        plot_maps(MSP_mask,S1_mask,S2_mask,back_mask,data_flat)
        H1_matrix = MSP_mask + S1_mask + S2_mask+ back_mask
        #H1_matrix=H1_matrix.reshape((20,20))
        
        #H2_matrix = MSP_mask + S2_mask + back[mask]
        tmp=data_flat - H1_matrix +\
            data_flat*np.log(H1_matrix/data_flat)
        s=np.isnan(tmp)
        p1[i] =tmp[~s].sum()
        # plt.imshow(tmp.reshape((20,20)))
        #plt.colorbar()
   #     plt.show()
        #p2 = np.sum(data_flat - H2_matrix +\
        #     data_flat*np.log(H2_matrix/data_flat))
    
    #we use the stirling approximation for the log-factorial term
    return np.sum(p1)

def myprior(cube):
    mu1 = 2.69463
    sigma1 = 0.13373
    #mu2 = 1.69943
    #sigma2 = 0.13128
    mu2 = 1.9815
    sigma2 = 0.077
    mu_n1 = 3.8873e-12
    sigma_n1 = 7.4243e-13
    #mu_n2 = 1.13141e-14
    #sigma_n2 = 1.7931e-15
    mu_n2 = 3.8183e-14
    sigma_n2 = 3.9030e-15    #Priors from recent fermi catalog
    cube[0]=cube[0]*10    #Gamma
    cube[1]=cube[1]*10  #Flat prior only around the Ecut found in the 1D analysis.
    cube[2]=cube[2]*(-10)  #N0
    cube[3]=np.log10(norm.ppf(cube[3], mu_n1, sigma_n1))   #Gaussian N1
    cube[4]=norm.ppf(cube[4],mu1,sigma1)   #Gaussian alpha_1
    cube[5]=np.log10(norm.ppf(cube[5], mu_n2, sigma_n2))  #Gaussian N2
    cube[6]=norm.ppf(cube[6], mu2, sigma2)    #Gaussian alpha_2
    return cube
    
    #def myloglike(cube):
    #   tmp=lnhood(cube)
#   return tmp


if __name__ == "__main__":
    #print(np.shape(no_events_MSP([1.55812824,5.97957804,-9.3947961])))

    from scipy import optimize as op
    import json
    
   

    # number of dimensions our problem has
    parameters = ["Gamma", "Ecut","logN","logNn1","alpha","logNn2","alpha2"]
    n_params = len(parameters)
    # name of the output files
    prefix = outdir+"/1-"

    with open('%sparams.json' % prefix, 'w') as f:
            json.dump(parameters, f, indent=2)
    # run MultiNest
    result = solve(LogLikelihood=lnhood, Prior=myprior,
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

