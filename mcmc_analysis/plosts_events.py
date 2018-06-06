import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from models_EF import *
sb.set_style('white')

chains_w_back = np.loadtxt('chains_events_wb.dat')
chains_wo_back = np.loadtxt('chains_events_wob.dat')

Ener_w,no_events_w,el,er = np.loadtxt('data_events_back.txt',unpack=True)
Ener_wo,no_events_wo,elwo,erwo = np.loadtxt('data_events.txt',unpack=True)
ew_min = Ener_w - el
ew_max = Ener_w + er

ewo_min = Ener_wo - elwo
ewo_max = Ener_wo + erwo

params_wback = []
params_woback = []

print len(no_events_wo), len(erwo)

for c1,c2,c3 in chains_w_back[np.random.randint(len(chains_w_back), size=100)]:
    params_wback.append(np.array([c1,c2,c3]))

for c1,c2,c3 in chains_wo_back[np.random.randint(len(chains_wo_back), size=100)]:
    params_woback.append(np.array([c1,c2,c3]))

for i in range(100):
    fig1,=plt.plot(Ener_w,no_events_model_pulsar(params_wback[i],ew_min,ew_max),'g',alpha=0.4)
    fig2,=plt.plot(Ener_wo,no_events_model_pulsar(params_woback[i],ewo_min,ewo_max),'r',alpha=0.4)


fig3,=plt.plot(Ener_w,no_events_w,'*k')
fig4,=plt.plot(Ener_wo,no_events_wo,'.b')
plt.legend([fig1,fig2,fig3,fig4],['with_Background','without_Background','Data_WB','Data_WoB'],loc='best')
plt.xlabel('Energy',fontsize=16)
plt.ylabel('No_Events',fontsize=16)
plt.show()
