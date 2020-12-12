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
from model import USCNN
from readOF import convertOFMeshToImage,convertOFMeshToImage_StructuredMesh
from bfinverse.util.utility.foamFileOperation import writeLeftAndRightToFile
from sklearn.metrics import mean_squared_error as calMSE
import Ofpp
h=0.01
OFBCCoord=Ofpp.parse_boundary_field('TemplateCase/30/C')
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
	             tolMesh=1e-10,tolJoint=1)
batchSize=1
NvarInput=2
NvarOutput=1
nEpochs=1500
lr=0.001
Ns=1
nu=0.01
model=USCNN(h,nx,ny,NvarInput,NvarOutput).to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
padSingleSide=1
udfpad=nn.ConstantPad2d([padSingleSide,padSingleSide,padSingleSide,padSingleSide],0)
MeshList=[]
MeshList.append(myMesh)
train_set=VaryGeoDataset(MeshList)
training_data_loader=DataLoader(dataset=train_set,
	                            batch_size=batchSize)
OFPicInformative=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase/30/C',
	                                           ['TemplateCase/30/T'],
	                                            [0,1,0,1],0.0,False)
OFPic=convertOFMeshToImage_StructuredMesh(nx,ny,'TemplateCase/30/C',
	                                           ['TemplateCase/30/T'],
	                                            [0,1,0,1],0.0,False)
OFX=OFPic[:,:,0]
OFY=OFPic[:,:,1]
OFV=OFPic[:,:,2]
OFV_sb=OFV
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
		outputV=output_pad[:,0,:,:].reshape(output_pad.shape[0],1,
			                                output_pad.shape[2],
			                                output_pad.shape[3])
		for j in range(batchSize):
			outputV[j,0,-padSingleSide:,padSingleSide:-padSingleSide]=0 
			outputV[j,0,:padSingleSide,padSingleSide:-padSingleSide]=1					   		
			outputV[j,0,padSingleSide:-padSingleSide,-padSingleSide:]=1 					    			
			outputV[j,0,padSingleSide:-padSingleSide,0:padSingleSide]=1 					    
			outputV[j,0,0,0]=0.5*(outputV[j,0,0,1]+outputV[j,0,1,0])
			outputV[j,0,0,-1]=0.5*(outputV[j,0,0,-2]+outputV[j,0,1,-1])
		dvdx=dfdx(outputV,dydeta,dydxi,Jinv)
		d2vdx2=dfdx(dvdx,dydeta,dydxi,Jinv)
		dvdy=dfdy(outputV,dxdxi,dxdeta,Jinv)
		d2vdy2=dfdy(dvdy,dxdxi,dxdeta,Jinv)
		continuity=(d2vdy2+d2vdx2);
		loss=criterion(continuity,continuity*0)
		loss.backward()
		optimizer.step()
		loss_mass=criterion(continuity, continuity*0)
		mRes+=loss_mass.item()
		CNNVNumpy=outputV[0,0,:,:].cpu().detach().numpy()
		eV=eV+np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))
	print('Epoch is ',epoch)
	print("mRes Loss is", (mRes/len(training_data_loader)))
	print("eV Loss is", (eV/len(training_data_loader)))
	if epoch%5000==0 or epoch%nEpochs==0 or np.sqrt(calMSE(OFV_sb,CNNVNumpy)/calMSE(OFV_sb,OFV_sb*0))<0.1:
		torch.save(model, str(epoch)+'.pth')
		fig1=plt.figure()
		ax=plt.subplot(1,2,1)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           outputV[0,0,1:-1,1:-1].cpu().detach().numpy(),'horizontal',[0,1])
		setAxisLabel(ax,'p')
		ax.set_title('CNN '+r'$T$')
		ax.set_aspect('equal')
		ax=plt.subplot(1,2,2)
		visualize2D(ax,coord[0,0,1:-1,1:-1].cpu().detach().numpy(),
			           coord[0,1,1:-1,1:-1].cpu().detach().numpy(),
			           OFV_sb[1:-1,1:-1],'horizontal',[0,1])
		setAxisLabel(ax,'p')
		ax.set_aspect('equal')
		ax.set_title('FV '+r'$T$')
		fig1.tight_layout(pad=1)
		fig1.savefig(str(epoch)+'T.pdf',bbox_inches='tight')
		plt.close(fig1)		
	return (mRes/len(training_data_loader)),(eV/len(training_data_loader))
			
			







			
			

MRes=[]
EV=[]
TotalstartTime=time.time()

for epoch in range(1,nEpochs+1):
	mres,ev=train(epoch)
	MRes.append(mres)
	EV.append(ev)
	if ev<0.1:
		break
TimeSpent=time.time()-TotalstartTime

plt.figure()
plt.plot(MRes,'-*',label='Equation Residual')
plt.xlabel('Epoch')
plt.ylabel('Residual')
plt.legend()
plt.yscale('log')
plt.savefig('convergence.pdf',bbox_inches='tight')
tikzplotlib.save('convergence.tikz')

plt.figure()
plt.plot(EV,'-x',label=r'$e_v$')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.yscale('log')
plt.savefig('error.pdf',bbox_inches='tight')
tikzplotlib.save('error.tikz')

EV=np.asarray(EV)
MRes=np.asarray(MRes)
np.savetxt('EV.txt',EV)
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
