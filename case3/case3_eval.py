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
import tikzplotlib
sys.path.insert(0, '../source')
from dataset import VaryGeoDataset, VaryGeoDataset_PairedSolution
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
from model import USCNN,USCNNSepPhi,USCNNSep,DDBasic
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
nxOF=50
nyOF=50
scalarList=[-0.1,-0.075,-0.05,-0.025,0.0,0.025,0.05,0.075,0.1]
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
model = torch.load('./Result/20000.pth')
model = model.cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
test_set=VaryGeoDataset_PairedSolution(MeshList,SolutionList)
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
startTime=time.time()
VelocityMagnitudeErrorRecord=[]
PErrorRecord=[]
for i in range(len(scalarList)):
	[JJInv,coord,xi,
	eta,J,Jinv,
	dxdxi,dydxi,
	dxdeta,dydeta,
	Utrue,Vtrue,Ptrue]=\
		to4DTensor(test_set[i])
	solutionTruth=torch.cat([Utrue,Vtrue,Ptrue],axis=1)
	coord=coord.reshape(coord.shape[1],coord.shape[0],coord.shape[2],coord.shape[3])
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
	Vmag_True=torch.sqrt(Utrue**2+Vtrue**2)
	Vmag_=torch.sqrt(output[0,0,:,:]**2+output[0,1,:,:]**2)
	VelocityMagnitudeErrorRecord.append(torch.sqrt(criterion(Vmag_True,Vmag_)/criterion(Vmag_True,Vmag_True*0)))
	PErrorRecord.append(torch.sqrt(criterion(Ptrue,output[0,2,:,:])/criterion(Ptrue,Ptrue*0)))
	xylabelsize=20
	xytickssize=20
	titlesize  =20
	fig1=plt.figure()
	ax=plt.subplot(2,2,1)
	_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
				coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
				Vmag_True.cpu().detach().numpy(),'vertical',
				[Vmag_True.cpu().detach().numpy().min()*0,
				 Vmag_True.cpu().detach().numpy().max()*0+0.5])
	ax.set_title('CFD Velocity', fontsize = titlesize)
	cbar.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
	ax.set_xticks([-0.5,0.5])
	ax.tick_params(axis='x',labelsize=xytickssize)
	ax.tick_params(axis='y',labelsize=xytickssize)
	cbar.ax.tick_params(labelsize=xytickssize)
	ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	cbar.remove()
	ax=plt.subplot(2,2,2)
	_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
				coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
				Ptrue.cpu().detach().numpy(),'vertical',
				[Ptrue.cpu().detach().numpy().min()*0,
				 Ptrue.cpu().detach().numpy().max()*0+0.3])
	ax.set_title('CFD Pressure', fontsize = titlesize)
	cbar.set_ticks([0,0.1,0.2,0.3])
	ax.set_xticks([-0.5,0.5])
	ax.tick_params(axis='x',labelsize=xytickssize)
	ax.tick_params(axis='y',labelsize=xytickssize)
	cbar.ax.tick_params(labelsize=xytickssize)
	ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	cbar.remove()
	ax=plt.subplot(2,2,3)
	_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
				coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
				Vmag_.cpu().detach().numpy(),'vertical',
				[Vmag_True.cpu().detach().numpy().min()*0,
				 Vmag_True.cpu().detach().numpy().max()*0+0.5])
	ax.set_title('PhyGeoNet Velocity', fontsize = titlesize)
	cbar.set_ticks([0,0.1,0.2,0.3,0.4,0.5])
	ax.set_xticks([-0.5,0.5])
	ax.tick_params(axis='x',labelsize=xytickssize)
	ax.tick_params(axis='y',labelsize=xytickssize)
	cbar.ax.tick_params(labelsize=xytickssize)
	ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	cbar.remove()
	ax=plt.subplot(2,2,4)
	_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
				coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
				outputP[0,0,1:-1,1:-1].cpu().detach().numpy(),'vertical',
				[Ptrue.cpu().detach().numpy().min()*0,
				 Ptrue.cpu().detach().numpy().max()*0+0.3])
	ax.set_title('PhyGeoNet Pressure', fontsize = titlesize)
	cbar.set_ticks([0,0.1,0.2,0.3])
	ax.set_xticks([-0.5,0.5])
	ax.tick_params(axis='x',labelsize=xytickssize)
	ax.tick_params(axis='y',labelsize=xytickssize)
	cbar.ax.tick_params(labelsize=xytickssize)
	ax.tick_params(axis='both',which='both',bottom=False,left=False,top=False,labelbottom=False,labelleft=False)
	cbar.remove()
	fig1.tight_layout(pad=1)
	fig1.savefig('Para'+str(i)+'UVPCFD.pdf',bbox_inches='tight')
	fig1.savefig('Para'+str(i)+'UVPCFD.png',bbox_inches='tight')
	plt.close(fig1)
VErrorNumpy=np.asarray([i.cpu().detach().numpy() for i in VelocityMagnitudeErrorRecord])
PErrorNumpy=np.asarray([i.cpu().detach().numpy() for i in PErrorRecord])
plt.figure()			
plt.plot(np.linspace(-0.1,0.1,9),VErrorNumpy,'-x',label='Velocity Error')
plt.plot(np.linspace(-0.1,0.1,9),PErrorNumpy,'-o',label='Pressure Error')
plt.legend()
plt.xlabel('Stenosis scaler')
plt.ylabel('Error')
plt.savefig('Error.pdf',bbox_inches='tight')
tikzplotlib.save('Error.tikz')





			
			



















































'''
			dudx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputU[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputU[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2udx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dudx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dudx)*dydxi[j:j+1,0:1,2:-2,2:-2])
			dvdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputV[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputV[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			d2vdx2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdxi(dvdx)*dydeta[j:j+1,0:1,2:-2,2:-2]-\
			       model.convdeta(dvdx)*dydxi[j:j+1,0:1,2:-2,2:-2])

			dudy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputU[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputU[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2udy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dudy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dudy)*dxdeta[j:j+1,0:1,2:-2,2:-2])
			dvdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputV[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputV[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])
			d2vdy2=Jinv[j:j+1,0:1,2:-2,2:-2]*(model.convdeta(dvdy)*dxdxi[j:j+1,0:1,2:-2,2:-2]-\
			     model.convdxi(dvdy)*dxdeta[j:j+1,0:1,2:-2,2:-2])

			dpdx=Jinv[j:j+1,0:1,:,:]*(model.convdxi(outputP[j:j+1,0:1,:,:])*dydeta[j:j+1,0:1,:,:]-\
			     model.convdeta(outputP[j:j+1,0:1,:,:])*dydxi[j:j+1,0:1,:,:])
			dpdy=Jinv[j:j+1,0:1,:,:]*(model.convdeta(outputP[j:j+1,0:1,:,:])*dxdxi[j:j+1,0:1,:,:]-\
			     model.convdxi(outputP[j:j+1,0:1,:,:])*dxdeta[j:j+1,0:1,:,:])

			continuity=dudx[:,:,2:-2,2:-2]+dudy[:,:,2:-2,2:-2];
			#u*dudx+v*dudy
			momentumX=outputU[j:j+1,:,2:-2,2:-2]*dudx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdx
			#-dpdx+nu*lap(u)
			forceX=-dpdx[0:,0:,2:-2,2:-2]+nu*(d2udx2+d2udy2)
			# Xresidual
			Xresidual=momentumX[0:,0:,2:-2,2:-2]-forceX   

			#u*dvdx+v*dvdy
			momentumY=outputU[j:j+1,:,2:-2,2:-2]*dvdx+\
			          outputV[j:j+1,:,2:-2,2:-2]*dvdy
			#-dpdy+nu*lap(v)
			forceY=-dpdy[0:,0:,2:-2,2:-2]+nu*(d2vdx2+d2vdy2)
			# Yresidual
			Yresidual=momentumY[0:,0:,2:-2,2:-2]-forceY 
			'''