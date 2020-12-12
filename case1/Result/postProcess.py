import numpy as np
import matplotlib.pyplot as plt 
import tikzplotlib
import pdb
res_c=np.loadtxt('MRes.txt')
res_x=np.loadtxt('XRes.txt')
res_y=np.loadtxt('YRes.txt')

err_u=np.loadtxt('EU.txt')
err_v=np.loadtxt('EV.txt')
err_p=np.loadtxt('EP.txt')


epoch=15000
interval=100
iteration=np.asarray([i for i in range(epoch)])
idx=[i for i in range(epoch) if i%interval==0]

plt.figure()
plt.plot(iteration[idx],res_c[idx],'o',label='continuity')
plt.plot(iteration[idx],res_x[idx],'x',label='x-momentum')
plt.plot(iteration[idx],res_y[idx],'*',label='y-momentum')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.yscale('log')
plt.savefig('Res'+str(epoch)+'.pdf',
            bbox_inches='tight')
tikzplotlib.save('Res'+str(epoch)+'.tikz')

plt.figure()
plt.plot(iteration[idx],err_u[idx],'o',label=r'$u$')
plt.plot(iteration[idx],err_v[idx],'x',label=r'$v$')
plt.plot(iteration[idx],err_p[idx],'*',label=r'$p$')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.yscale('log')
plt.savefig('Err'+str(epoch)+'.pdf',
            bbox_inches='tight')
tikzplotlib.save('Err'+str(epoch)+'.tikz')
