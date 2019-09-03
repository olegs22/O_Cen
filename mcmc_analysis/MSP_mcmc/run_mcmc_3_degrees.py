import numpy as np
import emcee
from model_3_degrees import *
from scipy import optimize as op

path_data = '/Users/Oleg/Documents/O_Cen/O_cen_2D/P8R3/'
Ener,no_events,e_min,e_max = np.loadtxt(path_data+'OC_no_events_15_bins_3_degree.txt',usecols=(0,1,2,3),unpack=True)

mask = no_events != 0.0
Ener = Ener[mask]
no_events = no_events[mask]
e_min = e_min[mask]
e_max = e_max[mask]

fun = lambda *args: -lnhood2(*args)
mid = [1.,6.,-6.,-10.,1.5,-10.,1.5,-10.,1.5,-10.,1.5]
bnds = ((0.,3.), (2.,8.), (-18.,0.),(-20.,0.),(0.,5.),(-20.,0.),(0.,5.),(-20.,0.),(0.,5.),(-20.,0.),(0.,5.))
result = op.minimize(fun, mid, args=(no_events,e_min,e_max),method = 'TNC',bounds=bnds)
print(result.x)

p_list = [0.,3., 2.,8.,-18.,0.,-20.,0.,0.,5.,-20.,0.,0.,5.,-20.,0.,0.,5.,-20.,0.,0.,5.]
nwalkers = 440
ndim = int((len(p_list)/2))
print('the dimension of the parameter space is: ',ndim)
z = np.zeros((ndim,nwalkers))

h = 1e-2
pos_i=[]
for i in range(ndim):
    z[i,:] = result.x[i] + h*np.random.randn(nwalkers)

for i in range(nwalkers):
    pos_i.append(np.array([z[0,i],z[1,i],z[2,i],z[3,i],z[4,i],
                 z[5,i],z[6,i],z[7,i],z[8,i],z[9,i],z[10,i]]))

b_steps, steps = 1000, 3000
sampler = emcee.EnsembleSampler(nwalkers,ndim,event_lnpost, 
                                args=(no_events,e_min,e_max,p_list),
                                threads = 8)

pos,prob,state=sampler.run_mcmc(pos_i, b_steps)
print(sampler.acceptance_fraction.mean())
sampler.reset()
_,_,_=sampler.run_mcmc(pos, steps, rstate0=state)
print(sampler.acceptance_fraction.mean())

np.savetxt('../chains/MSP_3_degree.dat',sampler.flatchain)
