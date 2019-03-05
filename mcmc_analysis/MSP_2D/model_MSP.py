import numpy as np
import h5py
from scipy.interpolate import interp1d
import scipy.integrate as integrate

path = '../../O_cen_2D/'
File_data_parameters = h5py.File(path+'gtbin_data.h5py','r')

#in File_data can also be fake data
File_data = h5py.File(path+'measure_data_real.h5py','r')

File_PSF =  h5py.File(path+'PSF_all_sources.h5py','r')

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

def Flux_MSP(pars,energy):
    Gamma,log_E,log_No = pars

    No = 10**log_No
    E_cut = 10**log_E

    return No*(energy**(-Gamma))*np.exp(-energy/E_cut)

def no_events_MSP(pars):
    
    no_events=np.zeros((naxis3,naxis1,naxis2))
    integrand=lambda x: Flux_MSP(pars,x)
    for k in range(naxis3):
        no_events[k]=integrate.quad(integrand,ebounds[k][0]/1000.,ebounds[k][1]/1000.)[0]\
        *exposure_oc[k]*np.ones((naxis1,naxis2))

    event_matrix = no_events*psf_oc
    events_vec = np.zeros((15,int(20*20))) #change the bining
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

def lnhood(pars):

    MSP = no_events_MSP(pars[:3])
    S1 = no_events_source(pars[3:5],530.6,psf_s2,exposure_s2) 
    S2 = no_events_source(pars[-2:],10204.39,psf_s1,exposure_s1)
    back = no_events_Background()

    p1 = np.zeros(15)

    for i in range(15):
        data_flat = data[i].flatten()
        mask = data_flat != 0.0

        data_flat = data_flat[mask]
        MSP_mask = MSP[i][mask]
        S1_mask = S1[i][mask] 
        S2_mask = S2[i][mask]
        back_mask = back[i][mask]
  
        H1_matrix = MSP_mask + S1_mask + S2_mask + back_mask

        p1[i] = np.sum(data_flat - H1_matrix +\
             data_flat*np.log(H1_matrix/data_flat))

    return np.sum(p1) 

def priors1(pars):
    Gamma,log_E,log_No,log_Nn1,alpha1,log_Nn2,alpha2  = pars
    plist = np.array([0.,3., 2.,8.,-18.,0.,-15.,0.,0.,5.,-15.,0.,0.,5.])
    mu1 = 2.69463#1.69943 #
    sigma1 = 0.13373#0.13128 #

    mu2 = 1.69943
    sigma2 = 0.13128

    mu_n1 = 3.8873e-12
    sigma_n1 = 7.4243e-13

    mu_n2 = 1.13141e-14
    sigma_n2 = 1.7931e-15

    log_f1 = ((alpha1 - mu1)/sigma1)**2 + np.log(2.0*np.pi*sigma1**2)
    log_f2 = ((alpha2- mu2)/sigma2)**2 + np.log(2.0*np.pi*sigma2**2)

    Nn1 = 10.**log_Nn1
    Nn2 = 10.**log_Nn2
    log_fn1 = ((Nn1 - mu_n1)/sigma_n1)**2 + np.log(2.0*np.pi*sigma_n1**2)
    log_fn2 = ((Nn2 - mu_n2)/sigma_n2)**2 + np.log(2.0*np.pi*sigma_n2**2)

    if plist[0]<Gamma<plist[1] and plist[2]<log_E<plist[3] and plist[4]<log_No<plist[5] and\
       plist[6]<log_Nn1<plist[7] and plist[8]<alpha1<plist[9] and plist[10]<log_Nn2<plist[11] and\
        plist[12]<alpha2<plist[13]:
       return -0.5 * (log_f1 + log_f2 + log_fn1 + log_fn2)
    return -np.inf

def event_lnpost(pars):
    pi = priors1(pars)
    if not np.isfinite(pi):
        return -np.inf
    return pi + lnhood(pars)


if __name__ == "__main__":
    #print(np.shape(no_events_MSP([1.55812824,5.97957804,-9.3947961])))
    from scipy import optimize as op
    mid = [1.,6.,-6.,-6.,1.5,-6.,1.5]
    bnds = ((0.,3.), (2.,8.), (-18.,0.),(-15.,0.),(0.,5.),(-15.,0.),(0.,5.))
    fun = lambda *args: -lnhood(*args)
    #result = op.minimize(fun, mid,method = 'TNC',bounds=bnds)
    #print(result.x)
    print(np.shape(no_events_Background()))

    print(lnhood([1.83,3.94,-6.09,-11.41,2.70,-13.96,1.70]))