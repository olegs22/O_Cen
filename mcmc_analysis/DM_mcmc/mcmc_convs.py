import numpy as np
import emcee
import argparse
import scipy.optimize as op


parser = argparse.ArgumentParser()
parser.add_argument('--model',type=str)
args = parser.parse_args()

if args.model == 'qq':
    from model_DM_qq_P8R3_new_model_conv import *
elif args.model == 'mu':
    from model_DM_mu_P8R3_new_model_conv import *



path = '../../O_cen_2D/P8R3/'
Ener,no_events,e_min,e_max = np.loadtxt(path + 'OC_no_events_15_bins_0.2_degree.txt',usecols=(0,1,2,3),unpack=True)

mask = no_events != 0.0
Ener = Ener[mask]
no_events = no_events[mask]
e_min = e_min[mask]
e_max = e_max[mask]
e_min = e_min/1000.
e_max = e_max/1000.

for i in range(15):
    fun = lambda *args: -lnhood2(*args)
    mid = [20.,-10.0,-6.,1.5]
    bnds = ((5.,51.),(-20.,0.),(-20.,0.),(0.,5.))

    result = op.minimize(fun,mid,args=(no_events,e_min,e_max,i),
                          method='TNC',bounds=bnds)
    print(result.x)

    p_list = [5.0,51.,-20.,0.,-20.,0.,0.,5.]
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
                                    args=(no_events,e_min,e_max,p_list,i),threads=8)

    pos,prob,state=sampler.run_mcmc(pos_i, b_steps)
    print sampler.acceptance_fraction.mean()
    sampler.reset()
    _,_,_=sampler.run_mcmc(pos, steps, rstate0=state)
    print sampler.acceptance_fraction.mean()

    if args.model == 'qq':
        np.savetxt('chains/cov_qq_J_'+str(i)+'.dat',sampler.flatchain)
    elif args.model == 'mu':
        np.savetxt('chains/cov_mu_J_'+str(i)+'.dat',sampler.flatchain)