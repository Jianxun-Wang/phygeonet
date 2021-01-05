import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.stats import norm as normdistribution 
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
				   np2cuda,to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
################################################################################
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
pdffunc=normdistribution(0,1)
h=0.01
def dfdx(f,dydeta,dydxi,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h 	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdx=Jinv*(dfdxi*dydeta-dfdeta*dydxi)
	return dfdx
def dfdy(f,dxdxi,dxdeta,Jinv):
	dfdxi_internal=(-f[:,:,:,4:]+8*f[:,:,:,3:-1]-8*f[:,:,:,1:-3]+f[:,:,:,0:-4])/12/h	
	dfdxi_left=(-11*f[:,:,:,0:-3]+18*f[:,:,:,1:-2]-9*f[:,:,:,2:-1]+2*f[:,:,:,3:])/6/h
	dfdxi_right=(11*f[:,:,:,3:]-18*f[:,:,:,2:-1]+9*f[:,:,:,1:-2]-2*f[:,:,:,0:-3])/6/h
	dfdxi=torch.cat((dfdxi_left[:,:,:,0:2],dfdxi_internal,dfdxi_right[:,:,:,-2:]),3)
	dfdeta_internal=(-f[:,:,4:,:]+8*f[:,:,3:-1,:]-8*f[:,:,1:-3,:]+f[:,:,0:-4,:])/12/h	
	dfdeta_low=(-11*f[:,:,0:-3,:]+18*f[:,:,1:-2,:]-9*f[:,:,2:-1,:]+2*f[:,:,3:,:])/6/h
	dfdeta_up=(11*f[:,:,3:,:]-18*f[:,:,2:-1,:]+9*f[:,:,1:-2,:]-2*f[:,:,0:-3,:])/6/h
	dfdeta=torch.cat((dfdeta_low[:,:,0:2,:],dfdeta_internal,dfdeta_up[:,:,-2:,:]),2)
	dfdy=Jinv*(dfdeta*dxdxi-dfdxi*dxdeta)
	return dfdy
os.system('mkdir test')
OFBCCoord=Ofpp.parse_boundary_field('TemplateCase_4side/1/C')
OFLOWC=OFBCCoord[b'low'][b'value']
OFUPC=OFBCCoord[b'up'][b'value']
OFLEFTC=OFBCCoord[b'left'][b'value']
OFRIGHTC=OFBCCoord[b'right'][b'value']
leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
upX=OFUPC[:,0];upY=OFUPC[:,1]
ny=len(leftX);nx=len(lowX)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
				 lowX,lowY,upX,upY,h,True,True,
				 tolMesh=1e-10,tolJoint=0.2)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase_4side/1/C',
											   ['TemplateCase_4side/1/f'],
												[0,1,0,1],0.0,False)
OFGT=np.reshape(np.loadtxt('TemplateCase_4side/GT.txt'),(ny,nx,1000), order='F')
OFFI=np.reshape(np.loadtxt('TemplateCase_4side/FI.txt'),(ny,nx,1000), order='F') 
OFKL=np.reshape(np.loadtxt('TemplateCase_4side/KLMode.txt'),(ny,nx,10), order='F') 
Coef=np.loadtxt('TemplateCase_4side/KLCoef.txt') 
eigval=np.loadtxt('TemplateCase_4side/eigVal.txt')
eigval=eigval/eigval.sum()
OFX=OFPic[:,:,0]
OFY=OFPic[:,:,1]
GT=OFGT*0
FI=OFFI*0
KL=OFKL*0
for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
		idx_min=np.where(dist == dist.min())
		GT[j,i,:]=OFGT[idx_min[0][0],idx_min[1][0],:]
		FI[j,i,:]=OFFI[idx_min[0][0],idx_min[1][0],:]
		KL[j,i,:]=OFKL[idx_min[0][0],idx_min[1][0],:]
Fontsize=20
for i in range(10):
	fig0=plt.figure()
	ax=plt.subplot(1,1,1)
	_,cbar=visualize2D(ax,myMesh.x,
				   myMesh.y,
				   KL[:,:,i],'vertical')
	ax.set_aspect('equal')
	setAxisLabel(ax,'p')
	ax.tick_params(axis='x',labelsize=Fontsize)
	ax.tick_params(axis='y',labelsize=Fontsize)
	cbar.ax.tick_params(labelsize=Fontsize)
	ax.set_title('mode '+str(i+1),fontsize=Fontsize)
	ax.set_xlabel(xlabel=r'$x$',fontsize=Fontsize)
	ax.set_ylabel(ylabel=r'$y$',fontsize=Fontsize)
	fig0.tight_layout(pad=0.5)
	plt.savefig('test/KLMode'+str(i+1)+'.png',bbox_inches='tight')
	plt.close(fig0)
trainSize=1000
batchSize=1
NvarInput=1
NvarOutput=1
nEpochs=1
lr=0.001
Ns=1
nu=0.01
model=torch.load('./Result/87252error0.08.pth').to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],10)
tol=[1,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.055,0.052,0.051,0.05]
train_set=[]
for i in range(trainSize):
	sample=[]
	sample.append(FI[:,:,i])
	sample.append(GT[:,:,i])
	train_set.append(sample)		
training_data_loader=DataLoader(dataset=train_set,
								batch_size=batchSize,shuffle=False)
[dydeta,dydxi,dxdxi,dxdeta,Jinv]=\
to4DTensor([myMesh.dydeta_ho,myMesh.dydxi_ho,
			myMesh.dxdxi_ho,myMesh.dxdeta_ho,
			myMesh.Jinv_ho])
Fontsize=22
scalefactor=1
input_scale_factor=500
def train(epoch,I):
	eV=0
	eVIndiviudal=[]
	evalTime=[]
	for i in range(trainSize):
		input=torch.from_numpy(train_set[i][0].reshape([1,1,train_set[i][0].shape[0],train_set[i][0].shape[1]])).to('cuda').float()
		truth=torch.from_numpy(train_set[i][1].reshape([1,1,train_set[i][0].shape[1],train_set[i][0].shape[1]])).to('cuda').float()
		tic=time.time()
		output=model(input/input_scale_factor)
		evalTime.append(time.time()-tic)
		outputV=udfpad(output)	
		eV=eV+torch.sqrt(criterion(truth,outputV/scalefactor)/criterion(truth,truth*0)).item()
		eVIndiviudal.append(torch.sqrt(criterion(truth,outputV/scalefactor)/criterion(truth,truth*0)).item())
	print("eV Loss is", (eV/len(training_data_loader)))
	eVIndiviudal=np.asarray(eVIndiviudal)
	error_2_sort=1*eVIndiviudal
	ID=error_2_sort[:256].argsort()
	ID_list=[ID[i] for i in range(18)]
	Er=[]
	weightPDF=[]
	for i in range(1000):
		input=torch.from_numpy(train_set[i][0].reshape([1,1,train_set[i][0].shape[0],train_set[i][0].shape[1]])).to('cuda').float()
		truth=torch.from_numpy(train_set[i][1].reshape([1,1,train_set[i][0].shape[1],train_set[i][0].shape[1]])).to('cuda').float()
		output=model(input/input_scale_factor)
		outputV=udfpad(output)
		print(i)
		print(eVIndiviudal[i])
		print(torch.sqrt(criterion(truth,outputV/scalefactor)/criterion(truth,truth*0)).item())
		print(Coef[:,i])
		pdf_temp=0
		for c in range(10):
			pdf_temp=pdf_temp+pdffunc.pdf(Coef[c,i])*eigval[c]
		weightPDF.append(pdf_temp)
		Er.append(eVIndiviudal[i])
		current_CNN=outputV[0,0,:,:].cpu().detach().numpy()/scalefactor
		current_trueth=truth[0,0,:,:].cpu().detach().numpy()
		cmax=(current_CNN.max()+current_trueth.max())/2
		cmin=(current_CNN.min()+current_trueth.min())/2
	return evalTime,eVIndiviudal,(eV/len(training_data_loader))
EV=[]
TotalstartTime=time.time()
I=0
for epoch in range(1,nEpochs+1):
	tic=time.time()
	evalTime,eVIndiviudal,ev=train(epoch,I)
	print('Time used = ',time.time()-tic)	
TimeSpent=time.time()-TotalstartTime
eVIndiviudal=np.asarray(eVIndiviudal)
EV=np.asarray(EV)
evalTime=np.asarray(evalTime)
np.savetxt('test/evalTime.txt',evalTime)
np.savetxt('test/EV.txt',EV)
np.savetxt('test/eVIndiviudal.txt',
			eVIndiviudal)
pdb.set_trace()

