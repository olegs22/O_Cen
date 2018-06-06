
"""

Created on Sat Dic 23 2017  14:49
@author: Javier Reynoso-Cordova


"""

##############################################
#####          Import libraries           ####
##############################################



from pylab import*
from matplotlib import*
from math import*
from astropy.io import fits
from matplotlib.colors import*
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import sys


"""
Capture the name of the file XXXX "gtpsf_XXXX.fits", "gtbin_XXXX.fits"
"""

object=np.str(sys.argv[1])


##############################################
####     Import .fits Fermi_tools        #####
##############################################

#"Import the results of the fermi_tools, in this case you need to have the output of gtbin for PHAI and gtpsf"

gtpsf=fits.open('gtpsf_'+str(object)+'.fits')
gtbin=fits.open('gtbin_PHA1_'+str(object)+'.fits')


#############################################
####          Extract the data          #####
#############################################

gtpsf_data=gtpsf[1].data
gtbin_data=gtbin[1].data

gtpsf_theta=gtpsf[2].data
gtbin_energy=gtbin[2].data

no_events=gtbin_data.field(1)           #number of events per energy

stat_error=gtbin_data.field(2)          #sys error
exposure=gtpsf_data.field(1)            #exposure as a function of energy
psf=gtpsf_data.field(2)                 #point spread function as a function of energy and solid angle


error_exposure=0.05*exposure            #%5 error in the Aeff

#############################################
####        Compute diff flux           #####
#############################################

"""
The number of events between energy E_i-1 and E_i and within R degrees of the source equals
the differential flux dphi/dE times Delta E (difference between E_i-1 and E_i) times the exposure
and PSF(E)
"""

#Compute the Delta E
e_min=gtbin_energy.field(1)/1000.   #units are in keV
e_max=gtbin_energy.field(2)/1000.



delta_energy=zeros(len(e_min))
log_energy=zeros(len(e_min))
delta_log_energy=(log10(max(e_max))-log10(min(e_min)))/len(e_max)

energy=zeros(len(e_min))
#the energy for gpsf must be in the center of the bin
for i in range(len(e_max)):
    delta_energy[i]=e_max[i]-e_min[i]
    log_energy[i]=log10(e_min[i])+ 0.5*delta_log_energy
    energy[i]=10**log_energy[i]
print(energy)

#Compute the integrated PSF(E)

theta=gtpsf_theta.field(0)*pi/180.
thetamax=1.
n_theta=len(psf)


psf_energy=zeros(len(psf))
psf_energy_1=zeros(len(psf))
psf_energy_2=zeros(len(psf))


for i in range(len(psf)):
    psf_solid_energy=interp1d(theta,psf[i])
    integrand=lambda thta: 2.*pi*thta*psf_solid_energy(thta)
    psf_energy[i]=integrate.quad(integrand,min(theta),thetamax*pi/180.)[0]
    psf_energy_1[i]=integrate.quad(integrand,min(theta),0.5*pi/180.)[0]
    psf_energy_1[i]=integrate.quad(integrand,min(theta),0.25*pi/180.)[0]

data_OC_fermi=loadtxt('OC_data_fermi.txt').T
energy_fermi=data_OC_fermi[0]
spectrum_fermi_e2=data_OC_fermi[1]

#compute the errors

#energy error
energy_error_up=empty(len(e_max))
energy_error_down=empty(len(e_max))
energy_error_up[0]=10**(log10((e_max[0]-e_min[0])/2.))
for j in range(1,len(e_max)):
    energy_error_up[j]=10**(log10((e_max[j]-e_min[j])/2.))
    energy_error_down[j]=10**(log10((e_min[j]-e_min[j-1])))

sys_error=-(energy**2)*(no_events)*(1.602e-6)/(delta_energy*(exposure+error_exposure)*psf_energy)+(energy**2)*(no_events)*(1.602e-6)/(delta_energy*exposure*psf_energy)

error=(energy**2)*(stat_error)*(1.602e-6)/(delta_energy*exposure*psf_energy)

#Compute the differential flux
E2dphidE_arr=(energy**2)*no_events/(delta_energy*exposure*psf_energy)*(1.602e-6)


######################################################
##           Import Flux function DM  secondary     ##
######################################################

import sys
sys.path.insert(0, '../Spectrum_Tables')

from Flux_DM import dEdN
"""
    dEdN returns E dN/dE
    phi returns the differential flux times E in units of cm^-2 s^-1
    takes an input of mass,energy,sigmav,J-factor,background,'species'
"""

"""
    sigmav 3.*10^-24 - 3.*10^-28
    mass 0.1-100
    J 10^18 - 10^22
"""

def phi_secondary(mass,energy,sigmav,J,N,species):
    return N+J*(1./(4.*pi))*sigmav*dEdN(mass,energy/mass,species)/(2.*(mass**2))

energy_probe=logspace(2.,4.,20.)
mass=100.
sigmav=2.e-26
J=10**18.05
N=0.

"""
species can be 'Electron', 'Muon', 'Tau', 'Q', 'C', 'B', 'T','W','Z','g','Photon','h'
"""
species='T'

#plot(energy_probe,(energy_probe)*(1000)*(1.602e-6)*phi_secondary(mass,energy_probe/1000.,sigmav,J,N,species))
#errorbar(energy,E2dphidE_arr,[error+sys_error,error+sys_error],[energy_error_up,energy_error_up],'o',ms=3.5)
#xlabel('E [MeV]')
#ylabel(r'$E^2 \frac{d\phi}{dE}$')
#xscale('log')
#yscale("log")
#xlim([1.e2,2.e4])
#ylim([1.e-13,3.e-11])
#savefig('OC_top.pdf')
#show()

#######################################################
##           Import Flux function DM Primary         ##
#######################################################

sys.path.insert(0, '../Primarios')

"""
#The columns are  [ mDM, Log10x , dN/d Log10x for 28 primary channels]
interp returns E dN/dE, input: mass,'col#'
"""
from Ocentauri import interp

"""
energy goes from 0 to mass
"""
def EdNdE_primary(mass,energy,col):
    a=interp1d(interp(mass,col)[0],interp(mass,col)[1])
    return a(energy)


def phi_primary(mass,energy,sigmav,J,N,col):
    return N+J*(1./(4.*pi))*sigmav*EdNdE_primary(mass,energy,col)/(2.*(mass**2))

savetxt('../mcmc_analysis/energy.dat',energy)
savetxt('../mcmc_analysis/flux.dat',E2dphidE_arr)
savetxt('../mcmc_analysis/data_errors.dat',error+sys_error)

plot(energy_probe,(energy_probe)*(1000)*(1.602e-6)*phi_primary(mass,energy_probe/1000.,sigmav,J,N,'col11'))
errorbar(energy,E2dphidE_arr,[error+sys_error,error+sys_error],[energy_error_up,energy_error_up],'o',ms=3.5)
xlabel('E [MeV]')
ylabel(r'$E^2 \frac{d\phi}{dE}$')
xscale('log')
yscale("log")
xlim([1.e2,2.e4])
ylim([1.e-13,3.e-11])
savefig('OC_top_primary.pdf')
show()
