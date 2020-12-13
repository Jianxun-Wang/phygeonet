# Python function to manipulate OpenFOAM files
# Developer: Jian-Xun Wang (jwang33@nd.edu)

###############################################################################

# system import
import numpy as np
import numpy.matlib
import sys # Add extra path/directory
import os
import os.path as ospt
import shutil
import subprocess # Call the command line
from subprocess import call
import matplotlib.pyplot as plt # For plotting
import re
import tempfile
import pdb
from vtk.util import numpy_support as VN
from matplotlib import pyplot as plt
import vtk.numpy_interface.dataset_adapter as dsa
import vtk
# local import
import bfinverse.util.utility.utilities as Tool
#import utilities as Tool
from PIL import Image
from vtk.util import numpy_support as VN
import vtk
import vtk.numpy_interface.dataset_adapter as dsa
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPRegressor
import multiprocessing
from functools import partial
import time
import multiprocessing
from functools import partial

import scipy.sparse as sp

global unitTest 
unitTest = False;

def getHMatrix(coarseMesh, fineMesh, NVOBS, NV, NP):
	'''
	coarseMesh: NgC * 3
	fineMesh:   NgF * 3
	NV: number of variable
	NP: number of paramter
	Method: Nearest point
	'''
	NgC=coarseMesh.shape[0]
	NgF=fineMesh.shape[0]
	H=sp.coo_matrix((NVOBS*NgC,NV*NgF+NP))
	H = sp.csr_matrix(H)
	Hobs=sp.csr_matrix((NVOBS*NgC,NV*NgF))
	for i in range(NgC):
		print('Building H Matrix '+str(i)+'/'+str(NgC))
		tempIndex = closest_node(coarseMesh[i,:], fineMesh)
		for j in range(NVOBS):
			H[i+j*NgC, tempIndex+j*NgF]=1
			Hobs[i+j*NgC, tempIndex+j*NgF]=1
	H=sp.coo_matrix(H) 
	Hobs=sp.coo_matrix(Hobs)
	return H.T, Hobs.T


def constructHMatrix(rasFile, Ncell, Nstate, NstateSample):
    """ Function is to construct H matrix
    To be modifyed
    Arg:
    
    Returns:
    
    """
# row, column, value tuple of a sparse matrix
    Nvector = 3     #number of elements in a vector

    idx = np.loadtxt(rasFile+"/constant/indexH.txt") #index of H Matrix
    #weight (element of H Matrix)
    weight = np.zeros((idx.shape[0], 1)) 
    weight[:, 0] = np.loadtxt(rasFile+"/constant/weightH.txt") 

    m, n = idx.shape
    idx3 = np.zeros((m*Nvector, n))
    weight3 = np.zeros((m*Nvector, 1))

    currentI = 0
    for i in range(int(idx[:, 0].max())+1): # for each block
        rg = np.where(idx[:,0]==i)[0]
        
        start, duration = rg[0], len(rg)

        
        idxBlock = np.copy(idx[start:start+duration, :])
        
        for ii in range(duration):

            idxBlock[ii,1] = idxBlock[ii,1]*Nvector

        wgtBlock = np.copy(weight[start:start+duration, :])
        
        idxBlock[:, 0] = currentI
        
        idxBlock1 =  np.copy(idxBlock)
        idxBlock1[:, 0] += 1
        idxBlock1[:, 1] += 1

        idxBlock2  = np.copy(idxBlock)
        idxBlock2[:, 0] += 2
        idxBlock2[:, 1] += 2

        idx3[Nvector*start:Nvector*(start+duration), :] = np.vstack((idxBlock, idxBlock1, idxBlock2))
        weight3[Nvector*start:Nvector*(start+duration), :] = np.vstack((wgtBlock, wgtBlock, wgtBlock))
        
        currentI += Nvector

    idxK = np.copy(idx)
    idxK[:, 0] += currentI
    idxK[:, 1] += Nvector*Ncell

    idx3 = np.append(idx3, idxK, axis=0)
    weight3 = np.append(weight3, weight, axis=0)

    H = sp.coo_matrix((weight3.flatten(1),(idx3[:,1],idx3[:,0])), shape=(Nstate,NstateSample))
    return H

def readTurbStressFromFile(tauFile):
	""" 
	Arg: 
	tauFile: The directory path of file of tau (OpenFOAM symmetric tensor files like Reynolds stress)

	Regurn: 
	tau: Matrix of Reynolds stress (sysmetric tensor)    
	"""
	resMid = extractSymmTensor(tauFile) 
	fout = open('tau.txt', 'w')
	glob_pattern = resMid.group()
	glob_pattern = re.sub(r'\(', '', glob_pattern)
	glob_pattern = re.sub(r'\)', '', glob_pattern)
	
	tau = glob_pattern
	fout.write(tau)
	fout.close();

	tau = np.loadtxt('tau.txt')
	#pdb.set_trace();
	return tau


def readTensorFromFile(tauFile):
	""" 
	Arg: 
	tauFile: The directory path of file of tau (OpenFOAM tensor files)

	Regurn: 
	tau: Matrix of tensor    
	"""
	resMid = extractTensor(tauFile)

	# write it in Tautemp 
	fout = open('tau.txt', 'w')
	glob_pattern = resMid.group()
	glob_pattern = re.sub(r'\(', '', glob_pattern)
	glob_pattern = re.sub(r'\)', '', glob_pattern)
	
	tau = glob_pattern
	fout.write(tau)
	fout.close();

	tau = np.loadtxt('tau.txt')
	#pdb.set_trace();
	return tau

def readVectorFromFile(UFile):
	""" 
	Arg: 
	tauFile: The directory path of OpenFOAM vector file (e.g., velocity)

	Regurn: 
	vector: Matrix of vector    
	"""
	resMid = extractVector(UFile)
	fout = open('Utemp', 'w');
	glob_pattern = resMid.group()
	glob_pattern = re.sub(r'\(', '', glob_pattern)
	glob_pattern = re.sub(r'\)', '', glob_pattern)
	fout.write(glob_pattern)
	fout.close();
	vector = np.loadtxt('Utemp')
	return vector

def readVelocityFromFile(UFile):
	""" Function is to get value of U from the openFoam U files
	
	Args:
	UFile: directory of U file in OpenFoam

	Returns:
	U: as flat vector (u1,v1,w1,u2,v2,w2,....uNcell,vNcell,wNcell)
	"""
	resMid = extractVector(UFile)    
	# write it in Utemp 
	fout = open('Utemp', 'w');
	fout.write(resMid.group())
	fout.close();
	
	# write it in UM with the pattern that numpy.load txt could read
	fin = open('Utemp', 'r')
	fout = open('UM.txt', 'w');
	
	while 1:
		line = fin.readline()
		line = line[1:-2]
		fout.write(line)
		fout.write(" ")
		if not line:
			break
	fin.close()
	fout.close();
	# to convert UM as U vector: (u1,v1,w1,u2,v2,w2,....uNcell,vNcell,wNcell)
	U = np.loadtxt('UM.txt')
	return U

def readTurbCoordinateFromFile(fileDir):
	""" 

	Arg: 
	fileDir: The directory path of file of Cx, Cy, and Cz

	Regurn: 
	coordinate: matrix of (x, y, z)    
	"""
	coorX = fileDir + "Cx"
	coorY = fileDir + "Cy"
	coorZ = fileDir + "Cz"

	resMidx = extractScalar(coorX)
	resMidy = extractScalar(coorY)
	resMidz = extractScalar(coorZ)

	# write it in Tautemp 
	fout = open('xcoor.txt', 'w')
	glob_patternx = resMidx.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	xVec = np.loadtxt('xcoor.txt')
	
	fout = open('ycoor.txt', 'w')
	glob_patterny = resMidy.group()
	glob_patterny = re.sub(r'\(', '', glob_patterny)
	glob_patterny = re.sub(r'\)', '', glob_patterny)
	fout.write(glob_patterny)
	fout.close();
	yVec = np.loadtxt('ycoor.txt')    
	
	fout = open('zcoor.txt', 'w')
	glob_patternz = resMidz.group()
	glob_patternz = re.sub(r'\(', '', glob_patternz)
	glob_patternz = re.sub(r'\)', '', glob_patternz)
	fout.write(glob_patternz)
	fout.close();
	zVec = np.loadtxt('zcoor.txt')

	coordinate = np.vstack((xVec, yVec, zVec))
	coordinate = coordinate.T
	return coordinate

def readTurbCellAreaFromFile(fileDir):    
	""" 

	Arg: 
	fileDir: The directory path of file of cv, dz.dat

	Regurn: 
	coordinate: vector of cell area    
	"""
	cellVolumn = fileDir + "cv"
	dvfile = fileDir + "dz.dat"
	resMid = extractScalar(cellVolumn)
	
	# write it in Tautemp 
	fout = open('cellVolumn.txt', 'w')
	glob_patternx = resMid.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	cvVec = np.loadtxt('cellVolumn.txt')
	cvVec = np.array([cvVec])
	dz = np.loadtxt(dvfile)
	cellArea = cvVec / dz
	cellArea = cellArea.T
	return cellArea

def readTurbCellVolumeFromFile(fileDir):    
	""" 

	Arg: 
	fileDir: The directory path of file of cv

	Regurn: 
	coordinate: vector of cell area    
	"""
	cellVolumn = fileDir + "cv"
	resMid = extractScalar(cellVolumn)
	
	# write it in Tautemp 
	fout = open('cellVolumn.txt', 'w')
	glob_patternx = resMid.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	cvVec = np.loadtxt('cellVolumn.txt')
	cvVec = np.array([cvVec])
	cellVolume = cvVec.T
	return cellVolume

	
def readScalarFromFile(fileName):    
	""" 

	Arg: 
	fileName: The file name of OpenFOAM scalar field

	Regurn: 
	a vector of scalar field    
	"""
	resMid = extractScalar(fileName)
	
	# write it in Tautemp 
	fout = open('temp.txt', 'w')
	glob_patternx = resMid.group()
	glob_patternx = re.sub(r'\(', '', glob_patternx)
	glob_patternx = re.sub(r'\)', '', glob_patternx)
	fout.write(glob_patternx)
	fout.close();
	scalarVec = np.loadtxt('temp.txt')
	return scalarVec

def writeLocToFile(coords, locFile):
	"""Write the coordinate matrix to the file: "obsLocations"
	
	Args:
	coords: coordinates (x, y, z) of locations
	locFile: path of the obsLocations file in OpenFOAM format
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'loctemp'
	np.savetxt(tempFile, coords)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	# read patterns
	(resStart, resEnd) = _extractLocPattern(locFile)
	#pdb.set_trace()     
	fout=open(locFile, 'w');
	fout.write(resStart.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEnd.group())
	fout.close();
	
	
def writeTurbStressToFile(tau, tauFile):
	"""Write the modified tau to the tauFile
	
	Args:
	tau: tau matrix (Reynolds stress matrix)
	tauFile: path of the tau
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'tauUpdate'
	np.savetxt(tempFile,tau)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	
	# read tensor out
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	# read patterns
	(resStartTau, resEndTau) = extractTensorPattern(tauFile)
	
	if(unitTest == True):
		tauFile = "./Tau"            
	fout=open(tauFile, 'w');
	fout.write(resStartTau.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEndTau.group())
	fout.close();
	
def writeVelocityToFile(U, UFile):
	"""Write the modified tau to the tauFile
	
	Args:
	U: modified velocity
	UFile: path of the U file in OpenFOAM
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'Utemp'
	np.savetxt(tempFile,U)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	
	# read tensor out
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	# read patterns
	(resStartU, resEndU) = extractTensorPattern(UFile)
			   
	fout=open(UFile, 'w');
	fout.write(resStartU.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEndU.group())
	fout.close();

def writeScalarToFile(Scalar, ScalarFile):
	"""Write the modified scalar to the scalar the OpenFOAM file
	
	Args:
	Scalar: E.g. DeltaXi or DeltaEta
	ScalarFile: path of the Scalar file in OpenFOAM
	
	Returns:
	None
		
	"""    

	# Find openFoam scalar file's pattern'
	(resStartk, resEndk) = extractTensorPattern(ScalarFile)

	tempFile = 'scalarTemp'
	np.savetxt('scalarTemp',Scalar)

	# read scalar field
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	#revise k

	fout=open(ScalarFile, 'w');
	fout.write(resStartk.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEndk.group())
	fout.close();


################################################ Regular Expression ##################################################### 
def extractSymmTensor(tensorFile):
	""" subFunction of readTurbStressFromFile
		Using regular expression to select tau value out (sysmetric tensor)
		Requiring the tensor to be 6-components tensor, and output is with 
		Parentheses.
	
	Args:
	tensorFile: The directory path of file of tensor

	Returns:
	resMid: the tau as (tau11, tau12, tau13, tau22, tau23, tau33);
			you need use resMid.group() to see the content.
	"""
	
	fin = open(tensorFile, 'r')  # need consider directory
	line = fin.read() # line is U file to read
	fin.close()

	### select U as (X X X)pattern (Using regular expression)
	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)                                                   # match )
	(\n|\ )                                              # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)

	resMid = patternMid.search(line)

	return resMid


def extractTensor(tensorFile):
	""" subFunction of readTurbStressFromFile
		Using regular expression to select tau value out (general tensor)
		Requiring the tensor to be 9-components tensor, and output is with 
		Parentheses.
	
	Args:
	tensorFile: The directory path of file of tensor

	Returns:
	resMid: the tau as (tau11, tau12, tau13, tau22, tau23, tau33);
			you need use resMid.group() to see the content.
	"""
	
	fin = open(tensorFile, 'r')  # need consider directory
	line = fin.read() # line is U file to read
	fin.close()

	### select U as (X X X)pattern (Using regular expression)
	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)            
	(\n|\ )                                              # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)

	resMid = patternMid.search(line)

	return resMid

def extractVector(vectorFile):
	""" Function is using regular expression select Vector value out
	
	Args:
	UFile: The directory path of file: U

	Returns:
	resMid: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
	"""

	fin = open(vectorFile, 'r')  # need consider directory
	line = fin.read() # line is U file to read
	fin.close()
	### select U as (X X X)pattern (Using regular expression)
	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)                                                   # match )
	\n                                                   # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)
	return resMid    
	
def extractScalar(scalarFile):
	""" subFunction of readTurbStressFromFile
		Using regular expression to select scalar value out 
	
	Args:
	scalarFile: The directory path of file of scalar

	Returns:
	resMid: scalar selected;
			you need use resMid.group() to see the content.
	"""
	fin = open(scalarFile, 'r')  # need consider directory
	line = fin.read() # line is k file to read
	fin.close()
	### select k as ()pattern (Using regular expression)
	patternMid = re.compile(r"""
		\(                                                   # match"("
		\n                                                   # match next line
		(
		[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
		\n                                                   # match next line
		)+                                                   # search greedly
		\)                                                   # match")"
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)

	return resMid

def extractTensorPattern(tensorFile):
	""" Function is using regular expression select OpenFOAM tensor files pattern
	
	Args:
	tensorFile: directory of file U in OpenFoam, which you want to change

	Returns:
	resStart: Upper Pattern 
	resEnd:  Lower Pattern
	"""
	fin = open(tensorFile, 'r')
	line=fin.read()
	fin.close()
	patternStart = re.compile(r"""
		.                        # Whatever except next line
		+?                       # Match 1 or more of preceding-Non-greedy
		internalField            # match interanlField
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		<((vector)|(symmTensor)|(scalar))>    # match '<vector>' or '<scalar>'
		((\ )|(\n))+?            # space or next line--non greedy
		[0-9]+                   # match 0-9
		((\ )|(\n))+?            # match space or next line
		\(                       # match (   
	""",re.DOTALL | re.VERBOSE)
	resStart = patternStart.search(line)

	patternEnd = re.compile(r"""
		\)                       # match )
		((\ )|;|(\n))+?          # match space or nextline or ;
		boundaryField            # match boundaryField
		((\ )|(\n))+?            # match space or nextline
		\{                       # match {
		.+                       # match whatever in {}
		\}                       # match }
	""",re.DOTALL | re.VERBOSE)
	resEnd = patternEnd.search(line)
	return resStart, resEnd

def _extractLocPattern(locFile):
	""" Function is using regular expression select OpenFOAM Location files pattern
	
	Args:
	tensorFile: directory of Locations in OpenFoam, which you want to change

	Returns:
	resStart: Upper Pattern 
	resEnd:  Lower Pattern
	"""
	fin = open(locFile, 'r')
	line=fin.read()
	fin.close()
	patternStart = re.compile(r"""
		.                        # Whatever except next line
		+?                       # Match 1 or more of preceding-Non-greedy
		\(                       # match (                   
	""",re.DOTALL | re.VERBOSE)
	resStart = patternStart.search(line)

	patternEnd = re.compile(r"""
		\)                       # match )
		((\ )|;|(\n))+          # match space or nextline or ;
		((\ )|;|(\n))+          # match space or nextline or ;
	""",re.DOTALL | re.VERBOSE)
	resEnd = patternEnd.search(line)
	#pdb.set_trace()
	return resStart, resEnd


  
def genFolders(Npara, Ns, caseName, caseNameObservation, DAInterval, Tau):
	""" Function:to generate case folders
	
	Args:
	Npara: number of parameters
	Ns: number of parameters
	caseName: templecase name(string)
	caseNameObservation: Observationcase name
	DAInterval: data assimilation interval

	Returns:
	None
	"""
	# remove previous ensemble case files
	os.system('rm -fr '+caseName+'-tmp_*');
	os.system('rm -fr '+caseName+'_benchMark');
	#pdb.set_trace()
	writeInterval = "%.6f"%DAInterval
	ii = 0
	caseCount = np.linspace(1, Ns, Ns)
	for case in caseCount:
		
		print("#", case, "/", Ns, " Creating folder for Case = ", case)
				
		tmpCaseName = caseName + "-tmp_" + str(case)
		
		if(ospt.isdir(tmpCaseName)): #see if tmpCaseName's'directory is existed
			shutil.rmtree(tmpCaseName)
		shutil.copytree(caseName, tmpCaseName) # copy
		
		# Replace Tau ensemble for cases ensemble
		tauTemp = Tau[ii, :, :]
		tauFile = './' + tmpCaseName + '/0/Tau'
		writeTurbStressToFile(tauTemp, tauFile)
		
		# Replace case writeInterval
		rasFile = ospt.join(os.getcwd(), tmpCaseName, "system", "controlDict")
		Tool.replace(rasFile, "<writeInterval>", writeInterval);          
		
		ii += 1
		
	#generate observation folder
	if(ospt.isdir(caseNameObservation)):
		shutil.rmtree(caseNameObservation)
	#pdb.set_trace()
	shutil.copytree(caseName, caseNameObservation) # copy   
	# prepare case directory
	rasFile = ospt.join(os.getcwd(), caseNameObservation, "system", "controlDict")
	Tool.replace(rasFile, "<writeInterval>", writeInterval);
	
def callFoam(ensembleCaseName, caseSolver, pseudoObs, parallel=False):

	""" Function is to call myPisoFoam and sampling (solved to next DAInterval)
	
	Args:
	ensembleCaseName: name of openFoam ensemble case folder

	Returns:
	None
	"""
	if(parallel):
		#run pisoFoam (or other OpenFOAM solver as appropriate)
		os.system('mpirun -np 4 pisoFoam -case ' + ensembleCaseName + 
				  ' -parallel > '+ ensembleCaseName + '/log')
		# extract value at observation location by "sample"_pseudo matrix H)
		os.system('mpirun -np 4 sample -case ' + ensembleCaseName + 
				  ' -latestTime -parallel > ' + ensembleCaseName + '/log')
	else: # same as above, but for single processor runs
		os.system(caseSolver + ' -case ' + ensembleCaseName + ' &>>' + 
				  ensembleCaseName + '/log')
		os.system('sample -case ' + ensembleCaseName + ' -latestTime > ' +
				  ensembleCaseName + '/sample.log')

		# os.system('myPisoFoam -case ' + ensembleCaseName)

		# extract value at observation location
		if pseudoObs == 1:
			os.system('sample -case ' + ensembleCaseName + ' -time 0 >> '+ 
					ensembleCaseName + '/log')
			os.system('sample -case ' + ensembleCaseName + ' -latestTime >> '+ 
					ensembleCaseName + '/log')
		else:
			pass

	
#########Han Gao add for OpenFOAM#########


def extractTensorPatternEdges(tensorFile):
	""" Function is using regular expression select OpenFOAM tensor files pattern    
	Args:
	tensorFile: directory of file blockMeshDict in OpenFoam, which you want to change
	Returns:
	resStart: Upper Pattern 
	resEnd:  Lower Pattern
	"""
	fin = open(tensorFile, 'r')
	line=fin.read()
	fin.close()
	patternStart = re.compile(r"""
		.                        # Whatever except next line
		+?                       # Match 1 or more of preceding-Non-greedy
		edges
		[a-zA-Z\ \n]+
		\(   
	""",re.DOTALL | re.VERBOSE)
	resStart = patternStart.search(line)

	patternEnd = re.compile(r"""
		\);
		[a-zA-Z\ \n]+
		boundary
		[a-zA-Z\ \n]+
		\(                      
		.+                       # match whatever in {}
		\);                      # match );
	""",re.DOTALL | re.VERBOSE)
	resEnd = patternEnd.search(line)
	return resStart, resEnd

def writeLowerFrontToFile(U1,U2,U3,U4,UFile):
	"""Write the modified tau to the tauFile
	
	Args:
	U: modified velocity
	UFile: path of the U file in OpenFOAM
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'Utemp'
	np.savetxt(tempFile,U1)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field1=fin.read()
	fin.close()

	np.savetxt(tempFile,U2)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field2=fin.read()
	fin.close()

	np.savetxt(tempFile,U3)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field3=fin.read()
	fin.close()

	np.savetxt(tempFile,U4)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field4=fin.read()
	fin.close()


	# read patterns
	(resStartU, resEndU) = extractTensorPatternEdges(UFile)		   
	fout=open(UFile, 'w');
	fout.write(resStartU.group())
	fout.write("\n")
	fout.write("spline 0 1\n")
	fout.write("(\n")
	fout.write(field1)
	fout.write(")\n")
	fout.write("spline 4 5\n")
	fout.write("(\n")
	fout.write(field2)
	fout.write(")\n")
	fout.write("spline 3 2\n")
	fout.write("(\n")
	fout.write(field3)
	fout.write(")\n")
	fout.write("spline 7 6\n")
	fout.write("(\n")
	fout.write(field4)
	fout.write(")\n")
	fout.write(resEndU.group())
	fout.close();

def writeLeftAndRightToFile(U1,U2,U3,U4,UFile):
	"""Write the modified tau to the tauFile
	
	Args:
	U: modified velocity
	UFile: path of the U file in OpenFOAM
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'Utemp'
	np.savetxt(tempFile,U1)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field1=fin.read()
	fin.close()

	np.savetxt(tempFile,U2)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field2=fin.read()
	fin.close()

	np.savetxt(tempFile,U3)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field3=fin.read()
	fin.close()

	np.savetxt(tempFile,U4)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	# read tensor out
	fin = open(tempFile, 'r')
	field4=fin.read()
	fin.close()


	# read patterns
	(resStartU, resEndU) = extractTensorPatternEdges(UFile)		   
	fout=open(UFile, 'w');
	fout.write(resStartU.group())
	fout.write("\n")
	fout.write("spline 0 3\n")
	fout.write("(\n")
	fout.write(field1)
	fout.write(")\n")
	fout.write("spline 4 7\n")
	fout.write("(\n")
	fout.write(field2)
	fout.write(")\n")
	fout.write("spline 1 2\n")
	fout.write("(\n")
	fout.write(field3)
	fout.write(")\n")
	fout.write("spline 5 6\n")
	fout.write("(\n")
	fout.write(field4)
	fout.write(")\n")
	fout.write(resEndU.group())
	fout.close();

def extractTensorPattern_Inlet(tensorFile):
	""" Function is using regular expression select OpenFOAM tensor files pattern
	
	Args:
	tensorFile: directory of file U in OpenFoam, which you want to change

	Returns:
	resStart: Upper Pattern 
	resEnd:  Lower Pattern
	"""
	fin = open(tensorFile, 'r')
	line=fin.read()
	fin.close()
	patternStart = re.compile(r"""
		.                        # Whatever except next line
		+?                       # Match 1 or more of preceding-Non-greedy
		inlet                    # match interanlField
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		{                        # match {
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		type                     # match type
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		fixedValue;              # match calculated;
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		value                    # match value
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		<((vector)|(symmTensor)|(scalar))>    # match '<vector>' or '<scalar>'
		((\ )|(\n))+?            # space or next line--non greedy
		[0-9]+                   # match 0-9
		((\ )|(\n))+?            # match space or next line
		\(                       # match (   
	""",re.DOTALL | re.VERBOSE)
	resStart = patternStart.search(line)

	patternEnd = re.compile(r"""
		\)                       # match )
		((\ )|;|(\n))+?          # match space or nextline or ;
		\}                       # match {
		.+                       # match whatever in {}
		\}                       # match }
	""",re.DOTALL | re.VERBOSE)
	resEnd = patternEnd.search(line)
	return resStart, resEnd

def writeInletVelocityToFile(U, UFile):
	"""Write the modified tau to the tauFile
	
	Args:
	U: modified velocity
	UFile: path of the U file in OpenFOAM
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'Utemp'
	np.savetxt(tempFile,U)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	
	# read tensor out
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	# read patterns
	(resStartU, resEndU) = extractTensorPattern_Inlet(UFile)
			   
	fout=open(UFile, 'w');
	fout.write(resStartU.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEndU.group())
	fout.close();




#Han Gao test write universal based on boundary name and value type

def extractTensorPattern_UDF(tensorFile, boundaryName, valueType):
	""" Function is using regular expression select OpenFOAM tensor files pattern
	
	Args:
	tensorFile: directory of file U in OpenFoam, which you want to change

	Returns:
	resStart: Upper Pattern 
	resEnd:  Lower Pattern
	"""
	fin = open(tensorFile, 'r')
	line=fin.read()
	fin.close()
	patternStart = re.compile(r"""
		.                        # Whatever except next line
		+?""" +                        # Match 1 or more of preceding-Non-greedy
		'.*(%s).*'%boundaryName#inlet                    # match interanlField
		+ r"""
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		{                        # match {
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		type                     # match type
		[a-zA-Z\ \n]""" +            # match class contained a-z A-Z space and \n
		
		'.*(%s).*'%valueType 
		+r""";              # match calculated;
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		value                    # match value
		[a-zA-Z\ \n]+            # match class contained a-z A-Z space and \n
		<((vector)|(symmTensor)|(scalar))>    # match '<vector>' or '<scalar>'
		((\ )|(\n))+?            # space or next line--non greedy
		[0-9]+                   # match 0-9
		((\ )|(\n))+?            # match space or next line
		\(                       # match (   
	""",re.DOTALL | re.VERBOSE)
	resStart = patternStart.search(line)

	patternEnd = re.compile(r"""
		\)                       # match )
		((\ )|;|(\n))+?          # match space or nextline or ;
		\}                       # match {
		.+                       # match whatever in {}
		\}                       # match }
	""",re.DOTALL | re.VERBOSE)
	resEnd = patternEnd.search(line)
	return resStart, resEnd


def writeUDFVectorToFile(U, UFile, boundaryName, valueType):
	"""Write the modified tau to the tauFile
	
	Args:
	U: modified velocity
	UFile: path of the U file in OpenFOAM
	
	Returns:
	None        
	"""
	# add parentheses to tensor
	tempFile = 'Utemp'
	np.savetxt(tempFile,U)
	os.system("sed -i -e 's/^/(/g' "+tempFile) 
	os.system("sed -i -e 's/\($\)/)/g' "+tempFile)
	
	# read tensor out
	fin = open(tempFile, 'r')
	field=fin.read()
	fin.close()
	# read patterns
	(resStartU, resEndU) = extractTensorPattern_UDF(UFile, boundaryName, valueType)
			   
	fout=open(UFile, 'w');
	fout.write(resStartU.group())
	fout.write("\n")
	fout.write(field)
	fout.write(resEndU.group())
	fout.close();

















































#########Han Gao add for VTK-postprocessing OpenFOAM#########
def readVTK(filename):
	#read the vtk file with an unstructured grid
	reader = vtk.vtkUnstructuredGridReader()
	reader.SetFileName(filename)
	reader.ReadAllFieldsOn()
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	return reader

def createLine(p1,p2,numPoints):
	# Create the line along which you want to sample
	line = vtk.vtkLineSource()
	line.SetResolution(numPoints)
	line.SetPoint1(p1)
	line.SetPoint2(p2)
	line.Update()
	return line

def createPlaneByPointNorm(p,n):
	# Create the plnae by p and n
	# for example p = (0, 0, 0) n = (1, 1, 1)
	plane = vtk.vtkPlane()
	plane.SetOrigin(p)
	plane.SetNormal(n)
	return plane

def createCutPlaneEdges(plane, reader):
	data = reader.GetOutput()
	planeCut = vtk.vtkCutter()
	planeCut.SetInputData(data)
	planeCut.SetCutFunction(plane)
	FeatureEdges = vtk.vtkFeatureEdges()
	FeatureEdges.SetInputConnection(planeCut.GetOutputPort())
	FeatureEdges.BoundaryEdgesOn()
	FeatureEdges.FeatureEdgesOff()
	FeatureEdges.NonManifoldEdgesOff()
	FeatureEdges.ManifoldEdgesOff()
	FeatureEdges.Update()
	return FeatureEdges

def probeOverLine(line,reader,fieldName):
	#Interpolate the data from the VTK-file on the created line
	data = reader.GetOutput()
	probe = vtk.vtkProbeFilter()
	probe.SetInputConnection(line.GetOutputPort())
	probe.SetSourceData(data)#probe.SetSource(data)
	probe.Update()
	numPoints = probe.GetOutput().GetNumberOfPoints()
	x = np.zeros(numPoints)
	y = np.zeros(numPoints)
	z = np.zeros(numPoints)
	points = np.zeros((numPoints , 3))
	#get the coordinates of the points on the line
	for i in range(numPoints):
		x[i],y[i],z[i] = probe.GetOutput().GetPoint(i)
		points[i,0]=x[i]
		points[i,1]=y[i]
		points[i,2]=z[i]
	field = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(fieldName))
	return points, field

def probeOverPlane(plane, reader, fieldName):
	data = reader.GetOutput()
	planeCut = vtk.vtkCutter()
	planeCut.SetInputData(data)
	planeCut.SetCutFunction(plane)
	probe = vtk.vtkProbeFilter()
	probe.SetInputConnection(planeCut.GetOutputPort())
	probe.SetSourceData(data)
	probe.Update()
	numPoints = probe.GetOutput().GetNumberOfPoints()
	x = np.zeros(numPoints)
	y = np.zeros(numPoints)
	z = np.zeros(numPoints)
	points = np.zeros((numPoints , 3))
	#get the coordinates of the points on the line
	for i in range(numPoints):
		x[i],y[i],z[i] = probe.GetOutput().GetPoint(i)
		points[i,0]=x[i]
		points[i,1]=y[i]
		points[i,2]=z[i]
	field = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(fieldName)) 
	return points, field

def probeOverEdges(FeatureEdges, reader, fieldName):
	data = reader.GetOutput()
	probe = vtk.vtkProbeFilter()
	probe.SetInputConnection(FeatureEdges.GetOutputPort())
	probe.SetSourceData(data)
	probe.Update()
	numPoints = probe.GetOutput().GetNumberOfPoints()
	x = np.zeros(numPoints)
	y = np.zeros(numPoints)
	z = np.zeros(numPoints)
	points = np.zeros((numPoints , 3))
	#get the coordinates of the points on the line
	for i in range(numPoints):
		x[i],y[i],z[i] = probe.GetOutput().GetPoint(i)
		points[i,0]=x[i]
		points[i,1]=y[i]
		points[i,2]=z[i]
	field = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(fieldName)) 
	return points, field

def probeOverEdges_ForYLessThanZero(FeatureEdges, reader, fieldName):
	'''
	Han Gao temp generated
	'''
	data = reader.GetOutput()
	probe = vtk.vtkProbeFilter()
	probe.SetInputConnection(FeatureEdges.GetOutputPort())
	probe.SetSourceData(data)
	probe.Update()
	numPoints = probe.GetOutput().GetNumberOfPoints()
	x = np.zeros(numPoints)
	y = np.zeros(numPoints)
	z = np.zeros(numPoints)
	points = np.zeros((numPoints , 3))
	field = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(fieldName))
	sampledPoints = []
	sampledField  = []
	#get the coordinates of the points on the line
	for i in range(numPoints):
		x[i],y[i],z[i] = probe.GetOutput().GetPoint(i)
		points[i,0]=x[i]
		points[i,1]=y[i]
		points[i,2]=z[i]
		if y[i] < 0:
			sampledPoints.append(points[i,:])
			sampledField.append(field[i,:])
	sampledField = np.asarray(sampledField)
	sampledPoints = np.asarray(sampledPoints)
	sortIndex = np.argsort(sampledPoints[:,0])
	sampledPoints = sampledPoints[sortIndex]
	sampledField = sampledField[sortIndex]
	return sampledPoints, sampledField







def probeOverEdges_ForYLessThanZero_FixBug(FeatureEdges, reader, fieldName):
	'''
	Han Gao temp generated
	'''
	data = reader.GetOutput()
	probe = vtk.vtkProbeFilter()
	probe.SetInputConnection(FeatureEdges.GetOutputPort())
	probe.SetSourceData(data)
	probe.Update()
	numPoints = probe.GetOutput().GetNumberOfPoints()
	x = np.zeros(numPoints)
	y = np.zeros(numPoints)
	z = np.zeros(numPoints)
	points = np.zeros((numPoints , 3))
	field = VN.vtk_to_numpy(probe.GetOutput().GetPointData().GetArray(fieldName))
	sampledPoints = []
	sampledField  = []
	#get the coordinates of the points on the line
	for i in range(numPoints):
		x[i],y[i],z[i] = probe.GetOutput().GetPoint(i)
		points[i,0]=x[i]
		points[i,1]=y[i]
		points[i,2]=z[i]
		if y[i] < 0 and np.linalg.norm(field[i,:]) > 1e-8: #1e-8 is okay! Tolerance
			sampledPoints.append(points[i,:])
			sampledField.append(field[i,:])
	sampledField = np.asarray(sampledField)
	sampledPoints = np.asarray(sampledPoints)
	sortIndex = np.argsort(sampledPoints[:,0])
	sampledPoints = sampledPoints[sortIndex]
	sampledField = sampledField[sortIndex]
	return sampledPoints, sampledField




def genVTK(caseCount, caseNameTemplate, currentPath, timeStep, Flag_RegenerateVTK):
	print ("generate vtk for ",)
	tic = time.time()
	readerAssemble = []
	for caseIdx in caseCount:
		caseFolder = currentPath+str(caseIdx)
		if Flag_RegenerateVTK == True:
			subprocess.call('source /opt/openfoam6/etc/bashrc',shell =True,cwd=caseFolder, executable='/bin/bash')
			subprocess.check_call('getBCCellSf',shell =True,cwd=caseFolder, executable='/bin/bash')
			subprocess.check_call('simpleFoam -postProcess -func wallShearStress',shell =True,cwd=caseFolder, executable='/bin/bash')
			subprocess.check_call('foamToVTK',shell =True,cwd=caseFolder, executable='/bin/bash')
		readerAssemble.append(readVTK(caseFolder + '/VTK/' + caseNameTemplate + str(caseIdx) + '_' + str(timeStep) + '.vtk'))                
		print("Current VTK path is !!! "+caseFolder + '/VTK/' + caseNameTemplate + str(caseIdx) + '_' + str(timeStep) + '.vtk')
	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in serial = ", elapseTime)
	return readerAssemble




# trival code for super-resolution 

def bcmaxid2interid(bcfield, bcmeshcoord, internalmeshcoord):
	bcmaxid = np.argmax(bcfield)
	bcmaxcoord = bcmeshcoord[bcmaxid,:]
	interid = closest_node(bcmaxcoord, internalmeshcoord)
	return interid

def bcminid2interid(bcfield, bcmeshcoord, internalmeshcoord):
	bcmaxid = np.argmin(bcfield)
	bcmaxcoord = bcmeshcoord[bcmaxid,:]
	interid = closest_node(bcmaxcoord, internalmeshcoord)
	return interid


def closest_node(node, nodes):
	nodes = np.asarray(nodes)
	dist_2 = np.sum((nodes - node)**2, axis=1)
	return np.argmin(dist_2)

def projectFineToCoarse(fineField, fineMesh, coarseMesh):
	if fineMesh.shape[0] != fineField.shape[0]:
		print(fineField.shape)
		print(fineMesh.shape)
		print('Error, fineField should be consistent with fineMesh!')
		exit()
	coarseField = np.zeros((coarseMesh.shape[0], fineField.shape[1]))
	for i in range(coarseMesh.shape[0]):
		tempIndex = closest_node(coarseMesh[i,:], fineMesh)
		coarseField[i,:] = fineField[tempIndex,:]
	return coarseField 

def projectUPFromHF2LF(fineMatrix, fineMesh,coarseMesh):
	if len(fineMatrix.shape)==1:
		coarseMatrix = np.zeros((4 * coarseMesh.shape[0]))
		#pdb.set_trace()
		for i in range(coarseMesh.shape[0]):
			tempIndex = closest_node(coarseMesh[i,:], fineMesh)
			#print(tempIndex)
			coarseMatrix[i] = fineMatrix[tempIndex]
			coarseMatrix[i + 1 * coarseMesh.shape[0]] = fineMatrix[tempIndex + 1 * fineMesh.shape[0]]
			coarseMatrix[i + 2 * coarseMesh.shape[0]] = fineMatrix[tempIndex + 2 * fineMesh.shape[0]]
			coarseMatrix[i + 3 * coarseMesh.shape[0]] = fineMatrix[tempIndex + 3 * fineMesh.shape[0]]
		return coarseMatrix
	else:
		projectFineDOFSnapeMatrixToCoarseDOFSnapeMatrix(fineMatrix, fineMesh, coarseMesh)


def projectFineDOFSnapeMatrixToCoarseDOFSnapeMatrix(fineMatrix, fineMesh, coarseMesh):
	if 4 * fineMesh.shape[0] != fineMatrix.shape[0]:
		#pdb.set_trace()
		if fineMesh.shape[0] == fineMatrix.shape[0]:
			coarseField=projectFineToCoarse(fineMatrix, fineMesh, coarseMesh)
			return coarseField
		else:
			print('Error, fineMatrix should be 4 or 1 times of the mesh for u v w p')
			exit()
	coarseMatrix = np.zeros((4 * coarseMesh.shape[0], fineMatrix.shape[1]))
	for i in range(coarseMesh.shape[0]):
		tempIndex = closest_node(coarseMesh[i,:], fineMesh)
		coarseMatrix[i,:] = fineMatrix[tempIndex,:]
		coarseMatrix[i + 1 * coarseMesh.shape[0],:] = fineMatrix[tempIndex + 1 * fineMesh.shape[0],:]
		coarseMatrix[i + 2 * coarseMesh.shape[0],:] = fineMatrix[tempIndex + 2 * fineMesh.shape[0],:]
		coarseMatrix[i + 3 * coarseMesh.shape[0],:] = fineMatrix[tempIndex + 3 * fineMesh.shape[0],:]
	return coarseMatrix


#lsun OF utility
# run 1 CFD cases
def run_stenosis(currentPath,caseIdx):
	caseFolder = currentPath+str(caseIdx)
	subprocess.check_call('simpleFoam > log',shell =True,cwd=caseFolder)



# parallal or serial run cases
def run_parallel(run_stenosis,currentPath,caseCount,nCores,parallelFlag=True):
	print ("perturb viscosity")
	if parallelFlag:
		tic = time.time()
		pool = multiprocessing.Pool(processes = nCores)
		pool.map(partial(run_stenosis, currentPath), caseCount)
		pool.close()
		print ("waiting for all C++ runs to be finished")
		pool.join()
		toc = time.time()
		elapseTime = toc - tic
		print ("elapse time in parallel = ", elapseTime)
		#np.savetxt('')
	else:
		tic = time.time()
		for caseIdx in caseCount:
			if iDebug:
				print ("propagate sample - ", str(caseIdx))
			else:
				run_stenosis(currentPath, caseIdx)        
		toc = time.time()
		elapseTime = toc - tic
		print ("elapse time in serial = ", elapseTime)
# Generate coordinate in FopenFoam
def getCenterCoordinate(currentPath,caseIdx):
	caseFolder = currentPath+str(caseIdx)
	subprocess.check_call('postProcess -func writeCellCentres',shell =True,cwd=caseFolder+'/')

