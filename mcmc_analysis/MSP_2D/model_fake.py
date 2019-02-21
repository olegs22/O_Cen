import numpy as np
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy import units as u
import scipy.integrate as integrate


File_psf_oc_data = h5py.File('../../O_cen_2D/psf_data_oc.h5py','r')
File_psf_s1_data = h5py.File('../../O_cen_2D/psf_data_source1.h5py','r')
File_psf_s2_data = h5py.File('../../O_cen_2D/psf_s2_data.h5py','r')

File_data = h5py.File('../../O_cen_2D/measured_data_20p.h5py','r')
File_fake_data = h5py.File('../../O_cen_2D/fake_data_perfect.h5py','r')

File_psf_oc =  h5py.File('../../O_cen_2D/PSF_OC_2D_20p_v2.h5py','r')
File_psf_s1 = h5py.File('../../O_cen_2D/PSF_S1_2D_20p_v2.h5py','r')
File_psf_s2 = h5py.File('../../O_cen_2D/PSF_S2_2D_20p_v2.h5py','r')

psf_oc = File_psf_oc['psf/psf_oc'][()]
psf_s1 = File_psf_s1['psf/psf_s1'][()]
psf_s2 = File_psf_s2['psf/psf_s2'][()]

exposure_oc = File_psf_oc_data['exposure/psf_data'][()]
exposure_s1 = File_psf_s1_data['exposure/psf_data'][()]
exposure_s2 = File_psf_s2_data['exposure/psf_data'][()]

ebounds = File_data['ebounds/maps'][()]
naxis1 = File_data['naxis1/maps'][()]
naxis2 = File_data['naxis2/maps'][()]
naxis3 = File_data['naxis3/maps'][()]
binsz = File_data['binsz/maps'][()]*np.pi/180.
data_fake = File_fake_data['data/photons'][()]

Backgroundfile = '../../O_cen_2D/omegacen_fermibgmodel.txt'
energy_back,Isotropic,Diffuse=np.loadtxt(Backgroundfile,usecols=(0,1,2),unpack=True)
energy_back,Isotropic,Diffuse=energy_back*1000.,Isotropic/1000.,Diffuse/1000.

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
    #p2 = np.zeros(15)
    for i in range(15):
        data_flat = data_fake[i].flatten()
        mask = data_flat != 0.0

        data_flat = data_flat[mask]
        MSP_mask = MSP[i][mask]
        S1_mask = S1[i][mask] 
        S2_mask = S2[i][mask]
        back_mask = back[i][mask]
  
        H1_matrix = MSP_mask + S1_mask + S2_mask + back_mask
        #H2_matrix = MSP_mask + S2_mask + back[mask]
        p1 = np.sum(data_flat - H1_matrix +\
             data_flat*np.log(H1_matrix/data_flat))

        #p2 = np.sum(data_flat - H2_matrix +\
        #     data_flat*np.log(H2_matrix/data_flat))
    """
    H1 = np.zeros(15)
    H2 = np.zeros(15)
    data = np.zeros(15)
    for i in range(15):
        H1[i] = np.sum(MSP[i]) + np.sum(S1[i]) + np.sum(back)
        H2[i] = np.sum(MSP[i]) + np.sum(S2[i]) + np.sum(back)
        data[i] = np.sum(maps_data[0].data[i])
    """
  
    #we use the stirling approximation for the log-factorial term

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