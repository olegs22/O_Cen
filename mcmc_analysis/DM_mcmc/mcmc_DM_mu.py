import numpy as np
import emcee
from scipy import optimize as op
from model_DM_mu import *

Ener,no_events,e_min,e_max = np.loadtxt('data/OC_no_events_9_bins.txt',usecols=(0,1,2,3),unpack=True)
mask = no_events != 0.0
Ener = Ener[mask] #this one is not used in this analysis
no_events = no_events[mask]
e_min = e_min[mask]
e_max = e_max[mask]
e_min = e_min/1000. #to correct for the units.
e_max = e_max/1000.
"""
Ener = Ener[2:]
no_events = no_events[2:]
e_min = e_min[2:]
e_max = e_max[2:]
"""
#here a call for the total likelihood
fun = lambda *args: -(lnhood1(*args) + lnhood2(*args))
mid = [10.,-10.0,-6.,1.5,-6.,1.5]
bnds = ((0.5,20.),(-20.,0.),(-20.,0.),(0.,5.),(-15.,0.),(0.,5.))
#i search for the maximun likelihood to initialize my chains in a high prob zone.
result = op.minimize(fun,mid,args=(no_events,e_min,e_max),
                    method='TNC',bounds=bnds)
print result.x

p_list = [ 0.5,20.,-20.,0.,-20.,0.,0.,5.,-15.,0.,0.,5.]
nwalkers = 240
ndim = (len(p_list)/2)
z = np.zeros((ndim,nwalkers))

h = 1e-3
pos_i=[]
for i in range(ndim):
    z[i,:] = result.x[i] + h*np.random.randn(nwalkers)

for i in range(nwalkers):
    pos_i.append(np.array([z[0,i],z[1,i],z[2,i],z[3,i],z[4,i],z[5,i]]))

b_steps, steps = 500,1500

sampler = emcee.EnsembleSampler(nwalkers,ndim,event_lnpost,
                                args=(no_events,e_min,e_max,p_list),threads=2)

pos,prob,state=sampler.run_mcmc(pos_i, b_steps)
print sampler.acceptance_fraction.mean()
sampler.reset()
_,_,_=sampler.run_mcmc(pos, steps, rstate0=state)
print sampler.acceptance_fraction.mean()
np.savetxt('chains/DM_mu_nocondition.dat',sampler.flatchain)
