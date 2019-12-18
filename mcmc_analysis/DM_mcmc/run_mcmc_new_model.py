import numpy as np
import emcee
from model_DM_qq_P8R3_new_model import *
#from model_DM_mu_P8R3_new_model import *
import scipy.optimize as op

path = #'/Users/Oleg/Documents/O_Cen/O_cen_2D/P8R3/'
Ener,no_events,e_min,e_max = np.loadtxt(path + 'OC_no_events_15_bins_0.2_degree.txt',usecols=(0,1,2,3),unpack=True)

mask = no_events != 0.0
Ener = Ener[mask]
no_events = no_events[mask]
e_min = e_min[mask]
e_max = e_max[mask]
e_min = e_min/1000.
e_max = e_max/1000.

fun = lambda *args: -lnhood2(*args)
mid = [10.,-20.0,-6.,1.5]
bnds = ((0.5,50.),(-40.,0.),(-20.,0.),(0.,5.))

result = op.minimize(fun,mid,args=(no_events,e_min,e_max),
                    method='TNC',bounds=bnds)
print(result.x)

p_list = [ 0.5,50.,-40.,0.,-20.,0.,0.,5.]
nwalkers = 160
ndim = int((len(p_list)/2))
z = np.zeros((ndim,nwalkers))

h = 1e-2
pos_i=[]
for i in range(ndim):
    z[i,:] = result.x[i] + h*np.random.randn(nwalkers)

for i in range(nwalkers):
    pos_i.append(np.array([z[0,i],z[1,i],z[2,i],z[3,i]]))

b_steps, steps = 500,1500
sampler = emcee.EnsembleSampler(nwalkers,ndim,event_lnpost,
                                args=(no_events,e_min,e_max,p_list),threads=2)

pos,prob,state = sampler.run_mcmc(pos_i, b_steps)
sampler.reset()
_,_,_=sampler.run_mcmc(pos, steps, rstate0=state)
print(sampler.acceptance_fraction.mean())
np.savetxt('qq_new_model_chains.dat',sampler.flatchain)