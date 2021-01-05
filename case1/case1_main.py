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
from scipy.interpolate import interp1d
import tikzplotlib

sys.path.insert(0, '../source')
from dataset import VaryGeoDataset
from pyMesh import hcubeMesh, visualize2D, plotBC, plotMesh,setAxisLabel,\
                   np2cuda,to4DTensor
from model import USCNN,USCNNSepPhi,USCNNSep,DDBasic
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
h=0.01
OFBCCoord=Ofpp.parse_boundary_field('TemplateCase_simpleVessel/3200/C')
OFLOWC=OFBCCoord[b'low'][b'value']
OFUPC=OFBCCoord[b'up'][b'value']
OFLEFTC=OFBCCoord[b'left'][b'value']
OFRIGHTC=OFBCCoord[b'rifht'][b'value']

leftX=OFLEFTC[:,0];leftY=OFLEFTC[:,1]
lowX=OFLOWC[:,0];lowY=OFLOWC[:,1]
rightX=OFRIGHTC[:,0];rightY=OFRIGHTC[:,1]
upX=OFUPC[:,0];upY=OFUPC[:,1]
ny=len(leftX);nx=len(lowX)
myMesh=hcubeMesh(leftX,leftY,rightX,rightY,
	             lowX,lowY,upX,upY,h,True,True,
	             tolMesh=1e-10,tolJoint=1e-2)
####
batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=1
lr=0.001
Ns=1
nu=0.01
#model=USCNNSep(h,nx,ny,NvarInput,NvarOutput,'ortho').to('cuda')
model=torch.load('./Result/15000.pth')
model=model.to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
####
MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase_simpleVessel/3200/C',
	                                           ['TemplateCase_simpleVessel/3200/U',
	                                            'TemplateCase_simpleVessel/3200/p'],
	                                            [0,1,0,1],0.0,False)
OFX=OFPic[:,:,0]
OFY=OFPic[:,:,1]
OFU=OFPic[:,:,2]
OFV=OFPic[:,:,3]
OFP=OFPic[:,:,4]
OFU_sb=np.zeros(OFU.shape)
OFV_sb=np.zeros(OFV.shape)
OFP_sb=np.zeros(OFP.shape)
fcnn_P=np.zeros(OFU.shape)
fcnn_U=np.zeros(OFV.shape)
fcnn_V=np.zeros(OFP.shape)
fcnn=np.load('comparison_160000iter.npz')
fcnn_P_=fcnn['p_NN'].reshape(OFU_sb.shape)
fcnn_U_=fcnn['u_NN'].reshape(OFU_sb.shape)
fcnn_V_=fcnn['v_NN'].reshape(OFU_sb.shape)
fcnn_X=fcnn['x_coord'].reshape(OFU_sb.shape)
fcnn_Y=fcnn['y_coord'].reshape(OFU_sb.shape)
for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-fcnn_X)**2+(myMesh.y[j,i]-fcnn_Y)**2
		idx_min=np.where(dist == dist.min())
		fcnn_U[j,i]=fcnn_U_[idx_min]
		fcnn_V[j,i]=fcnn_V_[idx_min]
		fcnn_P[j,i]=fcnn_P_[idx_min]
for i in range(nx):
	for j in range(ny):
		dist=(myMesh.x[j,i]-OFX)**2+(myMesh.y[j,i]-OFY)**2
		idx_min=np.where(dist == dist.min())
		OFU_sb[j,i]=OFU[idx_min]
		OFV_sb[j,i]=OFV[idx_min]
		OFP_sb[j,i]=OFP[idx_min]
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
		[JJInv,coord,xi,eta,J,Jinv,dxdxi,dydxi,dxdeta,dydeta]=to4DTensor(batch)
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
			outputU[j,0,0,0]=1*(outputU[j,0,0,1])
			outputU[j,0,0,-1]=1*(outputU[j,0,0,-2])
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=output[j,1,-1,:].reshape(1,nx-2*padSingleSide) 
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1					   		
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=0 					    			
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=0 					    
			outputV[j,0,0,0]=1*(outputV[j,0,0,1])
			outputV[j,0,0,-1]=1*(outputV[j,0,0,-2])
			outputP[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 								  
			outputP[j,0,:padSingleSide,padSingleSide:-padSingleSide]=output[j,2,0,:].reshape(1,nx-2*padSingleSide)      
			outputP[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=output[j,2,:,-1].reshape(ny-2*padSingleSide,1)    	
			outputP[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=output[j,2,:,0].reshape(ny-2*padSingleSide,1)     
			outputP[j,0,0,0]=1*(outputP[j,0,0,1])
			outputP[j,0,0,-1]=1*(outputP[j,0,0,-2])
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
		continuity=dudx+dvdy;
		momentumX=outputU*dudx+outputV*dudy
		forceX=-dpdx+nu*(d2udx2+d2udy2)
		Xresidual=momentumX-forceX   
		momentumY=outputU*dvdx+outputV*dvdy
		forceY=-dpdy+nu*(d2vdx2+d2vdy2)
		Yresidual=momentumY-forceY
		loss=(criterion(Xresidual,Xresidual*0)+\
		  criterion(Yresidual,Yresidual*0)+\
		  criterion(continuity,continuity*0))
		#loss.backward()
		optimizer.step()
		loss_xm=criterion(Xresidual, Xresidual*0)
		loss_ym=criterion(Yresidual, Yresidual*0)
		loss_mass=criterion(continuity, continuity*0)
		xRes+=loss_xm.item()
		yRes+=loss_ym.item()
		mRes+=loss_mass.item()
		CNNUNumpy=outputU[0,0,:,:].cpu().detach().numpy()
		CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
		CNNPNumpy=outputP[0,0,:,:].cpu().detach().numpy()
		eU=eU+np.sqrt(calMSE(OFU_sb,CNNUNumpy)/calMSE(OFU_sb,OFU_sb*0))
		eV=eV+np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))
		eP=eP+np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0))
		eVmag=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(CNNUNumpy**2+CNNVNumpy**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))
		eVmag_FCNN=np.sqrt(calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(fcnn_U**2+fcnn_V**2))/calMSE(np.sqrt(OFU_sb**2+OFV_sb**2),np.sqrt(OFU_sb**2+OFV_sb**2)*0))
		print('VelMagError_CNN=',eVmag)
		print('VelMagError_FCNN=',eVmag_FCNN)
		print('P_err_CNN=',np.sqrt(calMSE(OFP_sb,CNNPNumpy)/calMSE(OFP_sb,OFP_sb*0)))
		print('P_err_FCNN=',np.sqrt(calMSE(OFP_sb,fcnn_P)/calMSE(OFP_sb,OFP_sb*0)))
	print('Epoch is ',epoch)
	print("xRes Loss is", (xRes/len(training_data_loader)))
	print("yRes Loss is", (yRes/len(training_data_loader)))
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("eU Loss is", (eU/len(training_data_loader)))
	print("eV Loss is", (eV/len(training_data_loader)))
	print("eP Loss is", (eP/len(training_data_loader)))
	if epoch%5000==0 or epoch%nEpochs==0:
		torch.save(model, str(epoch)+'.pth')
		fig0=plt.figure()
		ax=plt.subplot(2,3,1)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(fcnn_U[1:-1,1:-1]**2+\
			           		   fcnn_V[1:-1,1:-1]**2),'vertical',[0,1.3])
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('FCNN '+'Velocity')

		ax=plt.subplot(2,3,2)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(outputU[0,0,1:-1,1:-1].cpu().detach().numpy()**2+\
			           		   outputV[0,0,1:-1,1:-1].cpu().detach().numpy()**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Velocity')
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])

		ax=plt.subplot(2,3,3)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(OFU_sb[1:-1,1:-1]**2+\
			           		   OFV_sb[1:-1,1:-1]**2),'vertical',[0,1.3])
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Velocity')
		

		ax=plt.subplot(2,3,4)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           fcnn_P[1:-1,1:-1],'vertical',[0,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('FCNN '+'Pressure')
		
		ax=plt.subplot(2,3,5)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputP[0,0,1:-1,1:-1].cpu().detach().numpy(),'vertical',[0,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Pressure')

		ax=plt.subplot(2,3,6)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFP_sb[1:-1,1:-1],'vertical',[0,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Pressure')
		fig0.tight_layout(pad=1)
		fig0.savefig(str(epoch)+'VelMagAndPressureFCNN.pdf',bbox_inches='tight')
		plt.close(fig0)

		fig0=plt.figure()
		ax=plt.subplot(2,2,1)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(outputU[0,0,1:-1,1:-1].cpu().detach().numpy()**2+\
			           		   outputV[0,0,1:-1,1:-1].cpu().detach().numpy()**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])		
		ax=plt.subplot(2,2,2)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputP[0,0,1:-1,1:-1].cpu().detach().numpy(),'vertical',[0,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('PhyGeoNet '+'Pressure')
		ax.set_aspect(1.3)
		ax=plt.subplot(2,2,3)
		_,cbar=visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           np.sqrt(OFU_sb[1:-1,1:-1]**2+\
			           		   OFV_sb[1:-1,1:-1]**2),'vertical',[0,1.3])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Velocity')
		ax.set_aspect(1.3)
		cbar.set_ticks([0,0.3,0.6,0.9,1.3])
		ax=plt.subplot(2,2,4)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFP_sb[1:-1,1:-1],'vertical',[0,1.5])
		setAxisLabel(ax,'p')
		ax.set_title('CFD '+'Pressure')
		ax.set_aspect(1.3)
		fig0.tight_layout(pad=1)
		fig0.savefig(str(epoch)+'VelMagAndPressureCNN.pdf',bbox_inches='tight')
		plt.close(fig0)
	return (xRes/len(training_data_loader)), (yRes/len(training_data_loader)),\
		   (mRes/len(training_data_loader)), (eU/len(training_data_loader)),\
		   (eV/len(training_data_loader)), (eP/len(training_data_loader))

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
tikzplotlib.save('convergence.tikz')

plt.figure()
plt.plot(EU,'-o',label=r'$u$')
plt.plot(EV,'-x',label=r'$v$')
plt.plot(EP,'-*',label=r'$p$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
tikzplotlib.save('error.tikz')
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