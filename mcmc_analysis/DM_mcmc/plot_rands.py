import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sb
import corner as corner
sb.set_style('white')

def dNdE_muon(energy,mx):
   alpha=1./137.
   y=energy/mx
   s=4.*mx*mx
   m_mu=107./1000.
   term_1=(1-(1-y)*(1-y))/y
   term_2=np.log(s*(1-y)/m_mu/m_mu)-1
   return alpha*term_1*term_2/np.pi/mx

def dNdE_qq(energy,mx):
   alpha_1=.95
   alpha_2=6.5
   term_1=energy/mx
   term_2=(energy/mx)**(-1.5)
   term_3=np.exp(-alpha_2*energy/mx)
   return alpha_1*term_1*term_2*term_3

def dnde_uu(mass_x,energy):
    alpha1 = 0.95
    alpha2 = 6.5
    frac = energy/mass_x

    val = alpha1*pow(frac,-0.5)*np.exp(-alpha2*frac)
    return val

def dnde_mu(mass_x,energy):
    alpha = 1.0 / 137.
    s = 4.0 * mass_x**2
    y = energy / mass_x
    m_e = 0.107#Gev

    val = ((1.-(1.-y)**2)/y) * (np.log((s*(1.-y))/m_e**2) - 1.)
    return (alpha / (np.pi*mass_x)) * val

def no_events_DM_uu(pars,e_min,e_max):
    mass_x, log_alphax = pars
    alpha_x = 10.**log_alphax

    integrand_dm = lambda x:dnde_uu(mass_x,x)
    no_events = np.zeros(len(e_min))

    for i in range(len(e_min)):
        #if e_min[i]< mass_x <= e_max[i]:
        #    no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        #elif e_min[i]<mass_x and e_max[i]<mass_x:
        no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp)/(8.0 * np.pi * mass_x**2)

def no_events_DM_mu(pars,e_min,e_max):
    mass_x, log_alphax = pars
    alpha_x = 10.**log_alphax

    integrand_dm = lambda x:dnde_mu(mass_x,x)
    no_events = np.zeros(len(e_min))

    for i in range(len(e_min)):
        if e_min[i]< mass_x <= e_max[i]:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],mass_x)[0]
        elif e_min[i]<mass_x and e_max[i]<mass_x:
            no_events[i] = integrate.quad(integrand_dm,e_min[i],e_max[i])[0]

    return (no_events * alpha_x * Ocen_exp)/(8.0 * np.pi * mass_x**2)

def new_source_1(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 530.60 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def new_source_2(pars,energy):
    log_Nn,alpha = pars

    Nn = 10.**log_Nn
    E_p = 10204.39 * 1e-3
    val = Nn * (energy/E_p)**(-1.0*alpha)
    return val

def no_events_source1(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_1(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp1 * Ps_psf1 * n_events_ps

def no_events_source2(pars,e_min,e_max):

    #integrand_ps = lambda x:pulsar_p_source(pars,x)
    integrand_ps = lambda x:new_source_2(pars,x)
    n_events_ps = np.zeros(len(e_min))

    for i in range(len(e_min)):
        n_events_ps[i] = integrate.quad(integrand_ps,e_min[i],e_max[i])[0]

    return Ps_exp2 * Ps_psf2 * n_events_ps

Ener,no_events,e_min,e_max,Ocen_exp,background = np.loadtxt('data/OC_no_events_9_bins.txt',usecols=(0,1,2,3,4,6),unpack=True)
Ps_exp1, Ps_psf1 = np.loadtxt('data/source2_J1326.txt',usecols=(2,3),unpack=True)#9 bin data
Ps_exp2, Ps_psf2 = np.loadtxt('data/source2_J1328.txt',usecols=(2,3),unpack=True)#9 bin data
mask = no_events != 0.0
Ener = Ener[mask]
no_events = no_events[mask]
e_min = e_min[mask]
e_max = e_max[mask]
background = background[mask]
Ocen_exp = Ocen_exp[mask]
Ps_exp1 = Ps_exp1[mask]
Ps_psf1 =Ps_psf1[mask]
Ps_exp2 = Ps_exp2[mask]
Ps_psf2 =Ps_psf2[mask]

Ener_2 = Ener
events_2 = no_events
e_min = e_min/1000.
e_max = e_max/1000.

min2 = e_min
max2 = e_max
"""
Ener = Ener[2:]
no_events = no_events[2:]
e_min = e_min[2:]
e_max = e_max[2:]
Ocen_exp = Ocen_exp[2:]
"""

yerr = np.zeros(len(e_min))
chains_mu = np.loadtxt('chains/DM_mu_v2.dat')
#print no_events_DM_mu([4.75,-4.63],e_min,e_max)
fig1 = plt.figure()
xi_square_mu = []
for c1,c2,c3,c4,c5,c6 in chains_mu[np.random.randint(len(chains_mu), size=100)]:
    val_1 = no_events_DM_mu([c1,c2],e_min,e_max)
    val_2 = no_events_source1([c3,c4],e_min,e_max)
    val_3 = no_events_source2([c5,c6],e_min,e_max)
    fig_1,=plt.loglog(Ener,val_1,'g',alpha=0.4);
    #fig_2,=plt.loglog(Ener,val_2,'b',alpha=0.4);
    #fig_3,=plt.loglog(Ener,val_3,'r',alpha=0.4);

    #fig_2,=plt.plot(Ener,no_events_model_pulsar([np.mean(sampler.flatchain[:,0]),np.mean(sampler.flatchain[:,1]),np.mean(sampler.flatchain[:,2])],e_min,e_max)+ background);
    #plt.plot(Ener,no_events,'*b');
    total_model = val_1+val_2+val_3+background
    xi_square_mu.append(np.sum(pow(no_events - total_model,2)))
    fig_5, = plt.loglog(Ener,val_1+val_2+val_3+background,'c',alpha=0.5);
    #fig_6,=plt.loglog(Ener,background,'orangered');
    #plt.legend([fig_1,fig_2,fig_3,fig_4],[r'$\mu^+\mu^-$','S_1','S_2','data'],loc='best');
plt.plot(Ener,no_events,'*k');
#plt.errorbar(Ener,no_events,xerr=[e_min*1e3,e_max*1e3],fmt='o',color='k')
plt.title(r'\mu^+\mu^-')
plt.xlabel('Energy',fontsize=16);
plt.ylabel('No_Events',fontsize=16);
np.savetxt('xi_square_mu.dat',xi_square_mu)
chains_uu = np.loadtxt('chains/DM_uu_v2.dat')
#print no_events_DM_uu([14.75,-5.97],min2,max2)
fig2 = plt.figure()
xi_square_uu = []
for c1,c2,c3,c4,c5,c6 in chains_uu[np.random.randint(len(chains_uu), size=100)]:
    val_1 = no_events_DM_uu([c1,c2],min2,max2)
    val_2 = no_events_source1([c3,c4],e_min,e_max)
    val_3 = no_events_source2([c5,c6],e_min,e_max)
    fig_1,=plt.loglog(Ener_2,val_1,'g',alpha=0.4);
    #fig_2,=plt.loglog(Ener,val_2,'b',alpha=0.4);
    #fig_3,=plt.loglog(Ener,val_3,'r',alpha=0.4);
    total_model = val_1+val_2+val_3+background
    xi_square_uu.append(np.sum(pow(no_events - total_model,2)))

    #fig_2,=plt.plot(Ener,no_events_model_pulsar([np.mean(sampler.flatchain[:,0]),np.mean(sampler.flatchain[:,1]),np.mean(sampler.flatchain[:,2])],e_min,e_max)+ background);
    #plt.plot(Ener,no_events,'*b');
    fig_4,=plt.loglog(Ener_2,events_2,'*k');
    fig_5, = plt.loglog(Ener,val_1+val_2+val_3+background,'c',alpha=0.5);
    #fig_6,=plt.loglog(Ener,background,'orangered');
    #plt.legend([fig_1,fig_2,fig_3,fig_4],[r'$\mu^+\mu^-$','S_1','S_2','data'],loc='best');
    plt.title('uu')
    plt.xlabel('Energy',fontsize=16);
    plt.ylabel('No_Events',fontsize=16);
np.savetxt('xi_square_uu.dat',xi_square_uu)
E_flux,flux,e_l,e_r,e_up,e_down = np.loadtxt('data/data_flux.txt',usecols=(0,1,2,3,4,5),unpack=True)
E_flux = E_flux[:-2]
flux = flux[:-2]
e_l = e_l[:-2]
e_r = e_r[:-2]
e_up = e_up[:-2]
e_down = e_down[:-2]
E_gev = E_flux*1e-3
#flux *= 1.602e-6
J_mean_uu = 10**np.mean(chains_uu[:,1])
dnde_mean_uu = (dnde_uu(np.mean(chains_uu[:,0]),E_gev) * J_mean_uu * E_gev**2) / (8.0 * np.pi * np.mean(chains_uu[:,0])**2)
dnde_mean_uu *= 1e3
fig4 = plt.figure()
for c1,c2,c3,c4,c5,c6 in chains_uu[np.random.randint(len(chains_uu), size=100)]:
    J = 10**c2
    dnde = (dnde_uu(c1,E_gev) * J * E_gev**2) / (8.0 * np.pi * c1**2)
    dnde *= 1e3
    fig_1=plt.loglog(E_flux,dnde,'g',alpha=0.4)
    #plt.xscale('log')
    #plt.yscale('log')
fig_2=plt.errorbar(E_flux,flux,[e_down,e_up],[e_l,e_r],fmt='o',color='orangered')
fig_3=plt.loglog(E_flux,dnde_mean_uu,'r')
plt.xlabel('E[MeV]')
plt.ylabel('Flux')
#plt.legend([fig_1,fig_2,fig_3],['samples','data','mean'],loc='best')
plt.title(r'$uu$')

J_mean_mu = 10**np.mean(chains_mu[:,1])
dnde_mean_mu = (dnde_mu(np.mean(chains_mu[:,0]),E_gev) * J_mean_mu * E_gev**2) / (8.0 * np.pi * np.mean(chains_mu[:,0])**2)
dnde_mean_mu *= 1e3
fig5 = plt.figure()
for c1,c2,c3,c4,c5,c6 in chains_mu[np.random.randint(len(chains_mu), size=100)]:
    J = 10**c2
    dnde = (dnde_mu(c1,E_gev) * J * E_gev**2) / (8.0 * np.pi * c1**2)
    dnde *= 1e3
    fig_1=plt.loglog(E_flux,dnde,'g',alpha=0.5)
    #plt.xscale('log')
    #plt.yscale('log')
fig_2=plt.errorbar(E_flux,flux,[e_down,e_up],[e_l,e_r],fmt='o',color='orangered')
fig_3=plt.loglog(E_flux,dnde_mean_mu,'r')
plt.xlabel('E[MeV]')
plt.ylabel('Flux')
#plt.legend([fig_1,fig_2,fig_3],['samples','data','mean'],loc='best')
plt.title(r'$\mu^+\mu^-$')


plt.show()
