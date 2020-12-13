##########################################################################
##########################################################################
##########################################################################
import numpy as np
import numpy.matlib
import sys # Add extra path/directory
import pdb # Python debugger
import os  # Operating system
import shutil   # File copy and remove
import subprocess # Call the command line
from subprocess import call
import matplotlib.pyplot as plt # For plotting
#from PIL import Image
## Import local modules (Prof.JX-W's python code)
#RWOF_dir = os.path.expanduser("/home/hangao/Desktop/HanGao_Research/MyCodeTrival/utility")
# add the RWOF_dir to sys.path
#sys.path.append(RWOF_dir)
# import the modules you need
#import foamFileOperation as foamOp
#from utilities import readInputData
#import HanGaoRomUtility as HGRU
#import foamFileAddNoise as AN
#from sklearn.neural_network import MLPRegressor
#import Ofpp
#import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


##########################################################################
##########################################################################
##########################################################################

centerLineU = np.loadtxt('./center_line_U.xy')
centerLineP = np.loadtxt('./center_line_p.xy')
#stenosisLineU = np.loadtxt('./stenosis_line_U.xy')
#stenosisLineP = np.loadtxt('./stenosis_line_p.xy')

plt.figure()
plt.scatter(centerLineU[:,0], np.sqrt(centerLineU[:,1]**2 + centerLineU[:,2]**2 + centerLineU[:,3]**2), label = 'CFD steady state')  #Normalized by 8*mu^2*Re/(D^2*rho) 
plt.legend()
plt.xlabel('x coordinate')
plt.ylabel('Velocity Mag')
plt.title('CFD_CenterLineVelocity')
plt.savefig('CenterVelocity.pdf')

plt.figure()
plt.scatter(centerLineP[:,0], centerLineP[:,1], label = 'CFD steady state')  #Normalized by 8*mu^2*Re/(D^2*rho) 
plt.legend()
plt.xlabel('x coordinate')
plt.ylabel('P')
plt.title('CFD_CenterLinePressure_SeeTheDrop')
plt.savefig('CenterPressure.pdf')
'''
plt.figure()
plt.scatter(stenosisLineU[:,0], np.sqrt(stenosisLineU[:,1]**2 + stenosisLineU[:,2]**2 + stenosisLineU[:,3]**2), label = 'CFD steady state')  #Normalized by 8*mu^2*Re/(D^2*rho) 
plt.legend()
plt.xlabel('y coordinate')
plt.ylabel('Velocity Mag')
plt.title('CFD_StenosisLineVelocity')
plt.savefig('StenosisVelocity.pdf')

plt.figure()
plt.scatter(stenosisLineP[:,0], stenosisLineP[:,1], label = 'CFD steady state')  #Normalized by 8*mu^2*Re/(D^2*rho) 
plt.legend()
plt.xlabel('y coordinate')
plt.ylabel('P')
plt.title('CFD_StenosisLinePressure_SeeTheDrop')
plt.savefig('CFD_StenosisPressure.pdf')
'''
plt.show()
