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
from PIL import Image
## (Prof.JX-W's python code)
import foamFileOperation as foamOp
from utilities import readInputData
import scipy as sp
import scipy.sparse

##########################################################################
##########################################################################
##########################################################################
def GetRawPODModes(W, SnapMatrix, Flag_Display):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	W(matrix): cell volume of mesh
	SnapMatrix: store snapshots
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains each mode by method of snapshot
	Lambda: each mode's eigenvalue
	'''
	ShapeInfo = SnapMatrix.shape
	M = ShapeInfo[1] # number of snapshot
	Ng = len(W) # Ng(int scaler): number of cell in mesh 
	W = np.matlib.repmat(np.array(W),1, 3) # Here is W for diagonolize
	W_aug = np.diag(W[0]) # Here is W[0] for diagonolize
	C = np.matmul(np.matmul(np.transpose(SnapMatrix), W_aug), SnapMatrix)   
	Lambda, A = np.linalg.eig(C)
	# Calculate the Modes
	ModeMatrix = np.matmul(SnapMatrix, A)
	ModeMatrix = ModeMatrix / np.sqrt(Lambda)
	return ModeMatrix, Lambda

def GetRawPODModesWithPressure(W, SnapMatrix, Flag_Display):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	W(matrix): cell volume of mesh
	SnapMatrix: store snapshots
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains each mode by method of snapshot
	Lambda: each mode's eigenvalue
	'''
	ShapeInfo = SnapMatrix.shape
	M = ShapeInfo[1] # number of snapshot
	Ng = len(W) # Ng(int scaler): number of cell in mesh 
	W = np.matlib.repmat(np.array(W),1, 4) # Here is W for diagonolize
	W_aug = np.diag(W[0]) # Here is W[0] for diagonolize
	C = np.matmul(np.matmul(np.transpose(SnapMatrix), W_aug), SnapMatrix)   
	Lambda, A = np.linalg.eig(C)
	# Calculate the Modes
	ModeMatrix = np.matmul(SnapMatrix, A)
	ModeMatrix = ModeMatrix / np.sqrt(Lambda)
	return ModeMatrix, Lambda


def GetSnapMatrix_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	Pr: address where you store you offline case
	M: number of snapshot for build up snapmatrix
	Ng: number of cell in mesh
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains snapshots
	'''
	SnapMatrix = np.zeros((Ng*3, M)) 
	caseCount = np.linspace(1,M,M)
	for i in caseCount:
		U_temp = foamOp.readVectorFromFile(Pr + str(i)+ '/' + str(timeStep) + '/U')
		i = int(i-1)
		SnapMatrix[0:Ng,i] = U_temp[:,0]
		SnapMatrix[Ng:2*Ng,i] = U_temp[:,1]
		SnapMatrix[2*Ng:3*Ng,i] = U_temp[:,2]
	return SnapMatrix


def GetSnapMatrix_ForSteadyCase_byindex(Pr, caseCount, Ng, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	Pr: address where you store you offline case
	M: number of snapshot for build up snapmatrix
	Ng: number of cell in mesh
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains snapshots
	'''
	M = len(caseCount)
	SnapMatrix = np.zeros((Ng*3, M)) 
	j=0
	for i in caseCount:
		U_temp = foamOp.readVectorFromFile(Pr + str(i)+ '/' + str(timeStep) + '/U')
		SnapMatrix[0:Ng,j] = U_temp[:,0]
		SnapMatrix[Ng:2*Ng,j] = U_temp[:,1]
		SnapMatrix[2*Ng:3*Ng,j] = U_temp[:,2]
		j=j+1
	return SnapMatrix


def GetSnapMatrixPressure_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	Pr: address where you store you offline case
	M: number of snapshot for build up snapmatrix
	Ng: number of cell in mesh
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains snapshots
	'''
	SnapMatrix = np.zeros((Ng, M)) 
	caseCount = np.linspace(1,M,M)
	for i in caseCount:
		P_temp = foamOp.readScalarFromFile(Pr + str(i)+ '/' + str(timeStep) + '/p')
		i = int(i-1)
		SnapMatrix[0:Ng,i] = np.copy(P_temp)
	return SnapMatrix

def GetSnapMatrixPressure_ForSteadyCase_byindex(Pr, caseCount, Ng, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	Pr: address where you store you offline case
	M: number of snapshot for build up snapmatrix
	Ng: number of cell in mesh
	Flag_Display: 1 for display, 0 for not
	Returns:
	ModeMatrix: Matrix contains snapshots
	'''
	M = len(caseCount)
	SnapMatrix = np.zeros((Ng, M)) 
	j = 0
	for i in caseCount:
		P_temp = foamOp.readScalarFromFile(Pr + str(i)+ '/' + str(timeStep) + '/p')
		SnapMatrix[0:Ng,j] = np.copy(P_temp)
		j=j+1
	return SnapMatrix

def GetGlobalDOFMatrix(Pr, M, Ng, Flag_Display, timeStep):
	pMatrix = GetSnapMatrixPressure_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep)
	vMatrix = GetSnapMatrix_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep)
	globalMatrix = np.vstack((vMatrix, (pMatrix/np.absolute(pMatrix))*np.sqrt(2*np.absolute(pMatrix))))
	return globalMatrix

def GetGlobalDOFMatrix_byindex(Pr, caseCount, Ng, timeStep):
	pMatrix = GetSnapMatrixPressure_ForSteadyCase_byindex(Pr, caseCount, Ng, timeStep)
	vMatrix = GetSnapMatrix_ForSteadyCase_byindex(Pr, caseCount, Ng, timeStep)
	globalMatrix = np.vstack((vMatrix, (pMatrix/np.absolute(pMatrix))*np.sqrt(2*np.absolute(pMatrix))))
	return globalMatrix

def GetGlobalDOFRawMatrix(Pr, M, Ng, Flag_Display, timeStep):
	pMatrix = GetSnapMatrixPressure_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep)
	vMatrix = GetSnapMatrix_ForSteadyCase(Pr, M, Ng, Flag_Display, timeStep)
	globalMatrix = np.vstack((vMatrix, pMatrix))
	return globalMatrix

def getVeloPres(Pr,timeStep,caseCount=None, Ng=None):
	if caseCount is None and Ng is None:
		U=foamOp.readVectorFromFile(Pr+'/'+str(timeStep)+'/U')
		U_=np.vstack((U[:,0], U[:,1], U[:,2]))
		p_=foamOp.readScalarFromFile(Pr+'/'+str(timeStep)+'/p')
		DOF_=np.vstack((U_, (p_/np.absolute(p_))*np.sqrt(2*np.absolute(p_))))
		return DOF_.flatten()
	else:
		return GetGlobalDOFMatrix_byindex(Pr, caseCount, Ng, timeStep)

def getScalar(Pr,timeStep,varName,caseCount=None, Ng=None):
	if caseCount is None and Ng is None:
		var=foamOp.readScalarFromFile(Pr+'/'+str(timeStep)+'/'+varName)
		return var
	else:
		SnapMatrix = np.zeros((Ng, len(caseCount))) 
		for i in caseCount:
			P_temp = foamOp.readScalarFromFile(Pr + str(i)+ '/' + str(timeStep) + '/'+varName)
			i = int(i-1)
			SnapMatrix[0:Ng,i] = np.copy(P_temp)
		return SnapMatrix



def writeVeloPres(Pr, timeStep, ModeMatrix, N=None, Flag_Display=None):
	row=ModeMatrix.shape[0]
	Ng = int(row/4)
	if N is None:
		Umodetemp = np.zeros((Ng, 3))
		Umodetemp[:,0] = ModeMatrix[0:Ng]
		Umodetemp[:,1] = ModeMatrix[Ng:2*Ng]
		Umodetemp[:,2] = ModeMatrix[2*Ng:3*Ng]
		foamOp.writeVelocityToFile(Umodetemp, Pr + '/' + str(timeStep) + '/U')
		Pmodetemp = ModeMatrix[3*Ng:4*Ng]
		Pmodetemp = (Pmodetemp/np.absolute(Pmodetemp))*0.5*np.square(Pmodetemp)
		foamOp.writeScalarToFile(Pmodetemp, Pr + '/' + str(timeStep) + '/p')
	else:
		WriteGlobalDOF2OpenFOAM(N, ModeMatrix, Pr, Flag_Display, timeStep)

def WriteMode2OpenFOAM(N, ModeMatrix, Pr, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	N: the first N mode you want to output, no more than number of snapshot
	ModeMoatrx: matrix constains mode
	Pr: Address you want to write the mode in
	Flag_Display: 1 for display, 0 for not
	'''
	caseCount = np.linspace(1,N,N)
	row,col = ModeMatrix.shape
	Ng = int(row/3)
	if N > col:
		print('error: the mode number can not be greater the snapshot total number!\n')
		exit()
	for i in caseCount:
		Umodetemp = np.zeros((Ng, 3))
		Umodetemp[:,0] = ModeMatrix[0:Ng,int(i)-1]
		Umodetemp[:,1] = ModeMatrix[Ng:2*Ng,int(i)-1]
		Umodetemp[:,2] = ModeMatrix[2*Ng:3*Ng,int(i)-1]
		foamOp.writeVelocityToFile(Umodetemp, Pr + str(i)+ '/' + str(timeStep) + '/U')

def WriteGlobalDOF2OpenFOAM(N, ModeMatrix, Pr, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	N: the first N mode you want to output, no more than number of snapshot
	ModeMoatrx: matrix constains mode
	Pr: Address you want to write the mode in
	Flag_Display: 1 for display, 0 for not
	'''
	caseCount = np.linspace(1,N,N)
	row,col = ModeMatrix.shape
	Ng = int(row/4)
	if N > col:
		print('error: the mode number can not be greater the snapshot total number!\n')
		exit()
	for i in caseCount:
		Umodetemp = np.zeros((Ng, 3))
		Umodetemp[:,0] = ModeMatrix[0:Ng,int(i)-1]
		Umodetemp[:,1] = ModeMatrix[Ng:2*Ng,int(i)-1]
		Umodetemp[:,2] = ModeMatrix[2*Ng:3*Ng,int(i)-1]
		foamOp.writeVelocityToFile(Umodetemp, Pr + str(i)+ '/' + str(timeStep) + '/U')
		Pmodetemp = ModeMatrix[3*Ng:4*Ng,int(i)-1]
		Pmodetemp = (Pmodetemp/np.absolute(Pmodetemp))*0.5*np.square(Pmodetemp)
		foamOp.writeScalarToFile(Pmodetemp, Pr + str(i)+ '/' + str(timeStep) + '/p')










def WriteScalar2OpenFOAM(N, ModeMatrix, Pr, varName, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	N: the first N mode you want to output, no more than number of snapshot
	ModeMoatrx: matrix constains mode
	Pr: Address you want to write the mode in
	Flag_Display: 1 for display, 0 for not
	'''
	caseCount = np.linspace(1,N,N)
	row,col = ModeMatrix.shape
	Ng = row
	for i in caseCount:
		Umodetemp = np.zeros((Ng, 1))
		Umodetemp[:,0] = ModeMatrix[0:Ng,int(i)-1]
		foamOp.writeScalarToFile(Umodetemp, Pr + str(i)+ '/' + str(timeStep) + '/'+varName)
		
















def WriteRawGlobalDOF2OpenFOAM(N, ModeMatrix, Pr, Flag_Display, timeStep):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	N: the first N mode you want to output, no more than number of snapshot
	ModeMoatrx: matrix constains mode
	Pr: Address you want to write the mode in
	Flag_Display: 1 for display, 0 for not
	'''
	caseCount = np.linspace(1,N,N)
	row,col = ModeMatrix.shape
	Ng = int(row/4)
	if N > col:
		print('error: the mode number can not be greater the snapshot total number!\n')
		exit()
	for i in caseCount:
		Umodetemp = np.zeros((Ng, 3))
		Umodetemp[:,0] = ModeMatrix[0:Ng,int(i)-1]
		Umodetemp[:,1] = ModeMatrix[Ng:2*Ng,int(i)-1]
		Umodetemp[:,2] = ModeMatrix[2*Ng:3*Ng,int(i)-1]
		foamOp.writeVelocityToFile(Umodetemp, Pr + str(i)+ '/' + str(timeStep) + '/U')
		Pmodetemp = ModeMatrix[3*Ng:4*Ng,int(i)-1]
		foamOp.writeScalarToFile(Pmodetemp, Pr + str(i)+ '/' + str(timeStep) + '/p')
		

def GetAugmentedWeightMatrix(W):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	W(matrix): cell volume of mesh
	Returns:
	W_aug: augmented WeighMatrix
	'''
	W = np.matlib.repmat(np.array(W),1, 3) # Here is W for diagonolize
	W_aug = np.diag(W[0]) # Here is W[0] for diagonolize
	return W_aug

def GetAugmentedWeightMatrix_Global(W):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	W(matrix): cell volume of mesh
	Returns:
	W_aug: augmented WeighMatrix
	'''
	W = np.matlib.repmat(np.array(W),1, 4) # Here is W for diagonolize
	W_aug = np.diag(W[0]) # Here is W[0] for diagonolize
	return W_aug

def GetAugmentedWeightMatrix_Global_sparse(W):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	W(matrix): cell volume of mesh
	Returns:
	W_aug: augmented WeighMatrix
	'''
	W = np.matlib.repmat(np.array(W),1, 4) # Here is W for diagonolize
	W_aug = sp.sparse.spdiags(W, 0, W.size, W.size)
	return W_aug

def GetAugWeightMatrix(W, Npara):
	W=np.matlib.repmat(np.array(W),1, Npara)
	W_aug=sp.sparse.spdiags(W, 0, W.size, W.size)
	return W_aug

def CheckModeOrthogonal(ModeMatrix, W_aug, Pr):
	'''
	Reference: POD-Galerkin reduced order methods for CFD using Finite Volume Discretization: vortex shedding around a circular cylinder
	Args:
	ModeMatirx(matrix): matrix contains mode
	W_aug(matrix): matrix contains weight
	Returns:
	W_aug: augmented WeighMatrix
	'''
	OrthogonalMatrixCheck = np.matmul(np.matmul(np.transpose(ModeMatrix), W_aug), ModeMatrix) 
	plt.imshow(OrthogonalMatrixCheck)
	plt.xlabel('mode number')
	plt.ylabel('mode number')
	plt.title('Orthogonality Check For ModeMatrix')
	plt.colorbar()
	plt.savefig(Pr + 'ModeOrthogonality.png')

