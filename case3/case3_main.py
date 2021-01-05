import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.utils.data import DataLoader
import time
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
from model import USCNN,USCNNSepPhi,USCNNSep,DDBasic
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
nxOF=50
nyOF=50
scalarList=[-0.1,0.0,0.1] 
SolutionList=[]
MeshList=[]
for scalar in scalarList:
	OFcaseName_='./TemplateCase'+str(scalar)
	nx=nxOF+2;ny=nyOF+2;R=0.5;L=0;l=0.5;h=0.01
	idx=np.asarray(range(1,nx-1,1))
	idy=np.asarray(range(1,ny-1,1))
	leftY=np.linspace(-l/2-L/2,l/2+L/2,ny)
	rightY=np.linspace(-l/2-L/2,l/2+L/2,ny)
	leftX=[]; rightX=[]
	for i in leftY:
		if i>-l/2 and i<l/2:
			leftX.append(+np.cos(2*np.pi*i)*scalar-R)
			rightX.append(-np.cos(2*np.pi*i)*scalar+R)
		else:
			leftX.append(-R);rightX.append(R)
	leftX=np.asarray(leftX)
	rightX=np.asarray(rightX)
	lowX=np.linspace(-R,R,nx); lowY=lowX*0-l/2-L/2
	upX=lowX; upY=lowY+l+L
	myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
					lowX,lowY,upX,upY,h,False,True,'./Mesh'+str(scalar)+'.pdf')

	MeshList.append(myMesh)
	OFLF=np.zeros([nyOF,3])
	OFLB=np.zeros([nyOF,3])
	OFRF=np.zeros([nyOF,3])
	OFRB=np.zeros([nyOF,3])
	OFLF[:,2]=OFRF[:,2]=0;OFLB[:,2]=OFRB[:,2]=0.01;
	OFLF[:,0]=leftX[idy];OFLF[:,1]=leftY[idy]
	OFLB[:,0]=leftX[idy];OFLB[:,1]=leftY[idy]
	OFRF[:,0]=rightX[idy];OFRF[:,1]=rightY[idy]
	OFRB[:,0]=rightX[idy];OFRB[:,1]=rightY[idy]
	writeLeftAndRightToFile(OFLF,OFLB,OFRF,OFRB,OFcaseName_+'/system/blockMeshDict')
	OFPic=convertOFMeshToImage(OFcaseName_+'/3200/C',[OFcaseName_+'/3200/U',OFcaseName_+'/3200/p'],
				[0,1,0,1],0.0,False)
	OFU=OFPic[:,:,2]
	OFV=OFPic[:,:,3]
	OFP=OFPic[:,:,4]
	SolutionList.append(OFPic[:,:,2:])
batchSize=len(scalarList)
NvarInput=2
NvarOutput=1
nEpochs=100000
lr=0.001
Ns=1
nu=0.01
model=USCNNSep(h,nx,ny,NvarInput,NvarOutput,'kaiming').to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
train_set=VaryGeoDataset_PairedSolution(MeshList,SolutionList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
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
def train(epoch):
	startTime=time.time()
	xRes=0
	yRes=0
	mRes=0
	eU=0
	eV=0
	eP=0
	for iteration, batch in enumerate(training_data_loader):
		[JJInv,coord,xi,
		eta,J,Jinv,
		dxdxi,dydxi,
		dxdeta,dydeta,
		Utrue,Vtrue,Ptrue]=\
			 to4DTensor(batch)
		solutionTruth=torch.cat([Utrue,Vtrue,Ptrue],axis=1)
		optimizer.zero_grad()
		output=model(coord)
		output_pad=udfpad(output)
		outputU=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		outputV=output_pad[:,1,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		outputP=output_pad[:,2,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		XR=torch.zeros([batchSize,1,ny,nx]).to('cuda')
		YR=torch.zeros([batchSize,1,ny,nx]).to('cuda')
		MR=torch.zeros([batchSize,1,ny,nx]).to('cuda')
		for j in range(batchSize):
			outputU[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,0,-1,:].reshape(1,nx-2*padSingleSide) 
			outputU[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0 
			outputU[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0			
			outputU[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0
			outputU[j,0,0,0]=0.5*(outputU[j,0,0,1]+outputU[j,0,1,0])
			outputU[j,0,0,-1]=0.5*(outputU[j,0,0,-2]+outputU[j,0,1,-1])
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx-2*padSingleSide)  
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=0.4
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0			
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 
			outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0  
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx-2*padSingleSide)  
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny-2*padSingleSide,1) 	
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny-2*padSingleSide,1)  
			outputP[j,0,0,0]=0.5*(outputP[j,0,0,1]+outputP[j,0,1,0])
			outputP[j,0,0,-1]=0.5*(outputP[j,0,0,-2]+outputP[j,0,1,-1])		
		dudx=dfdx(outputU,dydeta,dydxi,Jinv)
		d2udx2=dfdx(dudx,dydeta,dydxi,Jinv)
		dudy=dfdy(outputU,dxdxi,dxdeta,Jinv)
		d2udy2=dfdy(dudy,dxdxi,dxdeta,Jinv)
		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
		dpdx=dfdx(outputP,dydeta,dydxi,Jinv)
		d2pdx2=dfdx(dpdx,dydeta,dydxi,Jinv)
		dpdy=dfdy(outputP,dxdxi,dxdeta,Jinv)
		d2pdy2=dfdy(dpdy,dxdxi,dxdeta,Jinv)
		continuity=dudx+dvdy
		momentumX=outputU*dudx+outputV*dudy
		forceX=-dpdx+nu*(d2udx2+d2udy2)
		Xresidual=momentumX-forceX   
		momentumY=outputU*dvdx+outputV*dvdy
		forceY=-dpdy+nu*(d2vdx2+d2vdy2)
		Yresidual=momentumY-forceY		
		loss=(criterion(Xresidual,Xresidual*0)+\
		  criterion(Yresidual,Yresidual*0)+\
		  criterion(continuity,continuity*0))
		loss.backward()
		optimizer.step()
		loss_xm=criterion(Xresidual, Xresidual*0)
		loss_ym=criterion(Yresidual, Yresidual*0)
		loss_mass=criterion(continuity, continuity*0)
		xRes+=loss_xm.item()
		yRes+=loss_ym.item()
		mRes+=loss_mass.item()
		eU=eU+torch.sqrt(criterion(Utrue,output[:,0:1,:,:])/criterion(Utrue,Utrue*0))
		eV=eV+torch.sqrt(criterion(Vtrue,output[:,1:2,:,:])/criterion(Vtrue,Vtrue*0))
		eP=eP+torch.sqrt(criterion(Ptrue,output[:,2:3,:,:])/criterion(Ptrue,Ptrue*0))
		if epoch%5000==0 or epoch%nEpochs==0:
			torch.save(model, str(epoch)+'.pth')
			for j in range(batchSize):
				fig1=plt.figure()
				ax=plt.subplot(2,3,1)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							Utrue[j,:,:,:].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('Physics Domain '+'U FV')
				ax.set_xticks([-0.25,0.25])
				ax=plt.subplot(2,3,2)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							Vtrue[j,:,:,:].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('Physics Domain '+'V FV')
				ax.set_xticks([-0.25,0.25])
				ax=plt.subplot(2,3,3)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							Ptrue[j,:,:,:].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('Physics Domain '+'P FV')
				ax.set_xticks([-0.25,0.25])
				ax=plt.subplot(2,3,4)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							outputU[j,0,1:-1,1:-1].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('U CNN')
				ax.set_xticks([-0.25,0.25])
				ax=plt.subplot(2,3,5)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							outputV[j,0,1:-1,1:-1].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('V CNN')
				ax.set_xticks([-0.25,0.25])
				ax=plt.subplot(2,3,6)
				visualize2D(ax,coord[j,0,1:-1,1:-1].cpu().detach().numpy(),
							coord[j,1,1:-1,1:-1].cpu().detach().numpy(),
							outputP[j,0,1:-1,1:-1].cpu().detach().numpy())
				setAxisLabel(ax,'p')
				ax.set_title('P CNN')
				ax.set_xticks([-0.25,0.25])
				fig1.tight_layout(pad=1)
				fig1.savefig('epoch'+str(epoch)+'Para'+str(j)+'UVPCFD.pdf',bbox_inches='tight')
				plt.close(fig1)
	print('Epoch is ',epoch)
	print("xRes Loss is", (xRes/len(training_data_loader)))
	print("yRes Loss is", (yRes/len(training_data_loader)))
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("eU Loss is", (eU/len(training_data_loader)))
	print("eV Loss is", (eV/len(training_data_loader)))
	print("eP Loss is", (eP/len(training_data_loader)))
	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   (mRes/len(training_data_loader)), (eU.item()/len(training_data_loader)),\
		   (eV.item()/len(training_data_loader)), (eP.item()/len(training_data_loader))
XRes=[];YRes=[];MRes=[]
EU=[];EV=[];EP=[]
TotalstartTime=time.time()
for epoch in range(1,nEpochs+1):
	xres,yres,mres,eu,ev,ep=train(epoch)
	XRes.append(xres)
	YRes.append(yres)
	MRes.append(mres)
	EU.append(eu)
	EV.append(ev)
	EP.append(ep)
TimeSpent=time.time()-TotalstartTime
plt.figure()
plt.plot(XRes,'-o',label='X-momentum Residual')
plt.plot(YRes,'-x',label='Y-momentum Residual')
plt.plot(MRes,'-*',label='Continuity Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
plt.figure()
plt.plot(EU,'-o',label=r'$e_u$')
plt.plot(EV,'-x',label=r'$e_v$')
plt.plot(EP,'-*',label=r'$e_p$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
EU=np.asarray(EU)
EV=np.asarray(EV)
EP=np.asarray(EP)
XRes=np.asarray(XRes)
YRes=np.asarray(YRes)
MRes=np.asarray(MRes)
np.savetxt('EU.txt',EU)
np.savetxt('EV.txt',EV)
np.savetxt('EP.txt',EP)
np.savetxt('XRes.txt',XRes)
np.savetxt('YRes.txt',YRes)
np.savetxt('MRes.txt',MRes)
np.savetxt('TimeSpent.txt',np.zeros([2,2])+TimeSpent)