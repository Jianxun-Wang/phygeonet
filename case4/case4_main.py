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
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, FixGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
				   np2cuda,to4DTensor
from model import USCNN
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
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
OFX=OFPic[:,:,0]
OFY=OFPic[:,:,1]
GT=OFGT*0
FI=OFFI*0
for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
		idx_min=np.where(dist == dist.min())
		GT[j,i,:]=OFGT[idx_min[0][0],idx_min[1][0],:]
		FI[j,i,:]=OFFI[idx_min[0][0],idx_min[1][0],:]
trainSize=256
batchSize=32
shuffleFlag=True
dir_name=str(trainSize)+'_'+str(batchSize)+'shuffle'+str(shuffleFlag)
NvarInput=1
NvarOutput=1
nEpochs=150000
lr=0.001
Ns=1
nu=0.01
essentialBC=10
model=USCNN(h,nx,ny,NvarInput,NvarOutput).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],essentialBC)
tol=[1,0.8,0.5,0.4,0.3,0.2,0.1,0.09,0.08,0.07,0.06,0.055,0.052,0.051,0.05]
train_set=[]
for i in range(trainSize):
	sample=[]
	sample.append(FI[:,:,i])
	sample.append(GT[:,:,i])
	train_set.append(sample)		
training_data_loader=DataLoader(dataset=train_set,
								batch_size=batchSize,shuffle=shuffleFlag)
[dydeta,dydxi,dxdxi,dxdeta,Jinv]=\
to4DTensor([myMesh.dydeta_ho,myMesh.dydxi_ho,
			myMesh.dxdxi_ho,myMesh.dxdeta_ho,
			myMesh.Jinv_ho])
os.system('mkdir '+dir_name)
scalefactor=1
input_scale_factor=500
def train(epoch,I):
	startTime=time.time()
	eVtemp=0
	for iteration, batch in enumerate(training_data_loader):
		[input,truth]=to4DTensor(batch)
		output_temp=model(input/input_scale_factor)
		outputV_temp=udfpad(output_temp)
		eVtemp=eVtemp+torch.sqrt(criterion(truth,outputV_temp/scalefactor)/criterion(truth,truth*0)).item()
	if (eVtemp/len(training_data_loader))<tol[I]:
		torch.save(model,dir_name+'/'+ str(epoch)+'error'+str(tol[I])+'.pth')
	mRes=0
	eV=0
	for iteration, batch in enumerate(training_data_loader):
		[input,truth]=to4DTensor(batch)
		optimizer.zero_grad()
		output=model(input/input_scale_factor)
		outputV=udfpad(output)	
		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
		continuity=(d2vdy2+d2vdx2)+input*scalefactor;
		loss=criterion(continuity,continuity*0)
		loss.backward()
		optimizer.step()
		loss_mass=criterion(continuity, continuity*0).item()
		mRes+=loss_mass
		eV=eV+torch.sqrt(criterion(truth,outputV/scalefactor)/criterion(truth,truth*0)).item()
	print('Epoch is ',epoch)
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("eV Loss is", (eVtemp/len(training_data_loader)))
	print("Case set = ",dir_name)
	return (mRes/len(training_data_loader)),(eVtemp/len(training_data_loader))
MRes=[]
EV=[]
TotalstartTime=time.time()
I=0
for epoch in range(1,nEpochs+1):
	tic=time.time()
	mres,ev=train(epoch,I)
	print('Time used = ',time.time()-tic)
	MRes.append(mres)
	EV.append(ev)
	if ev<tol[I]:
		I=I+1
		EV_temp=np.asarray(EV)
		MRes_temp=np.asarray(MRes)
		np.savetxt(dir_name+'/'+str(epoch)+'_'+'EV.txt',EV)
		np.savetxt(dir_name+'/'+str(epoch)+'_'+'MRes.txt',MRes)
		if I==len(tol):
			break
TimeSpent=time.time()-TotalstartTime
plt.figure()
plt.plot(MRes,'-*',label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig(dir_name+'/'+'convergence.png',bbox_inches='tight')
plt.figure()
plt.plot(EV,'-x',label=r'$e_v$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig(dir_name+'/'+'error.png',bbox_inches='tight')
EV=np.asarray(EV)
MRes=np.asarray(MRes)
np.savetxt(dir_name+'/'+'EV.txt',EV)
np.savetxt(dir_name+'/'+'MRes.txt',MRes)
np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)
np.savetxt(dir_name+'/'+'seed.txt',np.zeros([2,2])+torch.seed())