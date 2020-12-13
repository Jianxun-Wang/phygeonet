import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tikzplotlib
MRes=np.loadtxt('87252_MRes.txt')
I=[i for i in range(MRes.shape[0])]
ID=[i for i in I if i%100==0]
I=np.asarray(I)
plt.figure()
plt.plot(I[ID],MRes[ID],'-*',label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('high_dimension_case_convergence.pdf',bbox_inches='tight')
tikzplotlib.save('high_dimension_case_convergence.tikz')