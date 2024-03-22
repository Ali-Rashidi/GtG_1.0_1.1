# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:40:48 2023

@author: Ali
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:50:12 2023

@author: Ali
"""

import time
import open3d as o3d
import random
import copy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from functions import *
import torch
torch.backends.cudnn.benchmark=True
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64 "
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.optim import AdamW
import numpy as np
import math
from torch_geometric.data import Data
import cma
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")

st_episode = 0


episodes = 40000
explore_episode=0
capacity=10000
sigma_start = 0.1
sigma_step = 0.005
min_sample_per_epoch=10
max_samples_per_epoch=100
inference = False
k=10
load_params = False
save_params = False
lr=1e-5

num_epochs = 1
btch=2000

num_components = 1

num_train_objects=10
num_val_objects = 0
update_int = 40
num_objects = 10
solver_itr = 10
augment_active = False
num_augment = 18
sigma=0
buffer = ReplayBuffer(capacity=capacity)


## Network Config :
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

GtG = holyGrail().to(device)

print( " GtG Params : " + str ( sum(p.numel() for p in GtG.parameters())))

l_R = []
l_A = []
real_reward=[]
rrma=[]

val_reward=[]
val_rrma = []

bin_reward=[]
val_bin_reward=[]

E=[]

r2_list=[]
r3_list=[]
r4_list=[]

val_r2_list=[]
val_r3_list=[]
val_r4_list=[]


if False:
    graphs = torch.load("graphs_s.pt")
    rewards = torch.load("rewards_s.pt")
    actions = torch.load("actions_s.pt")
    
    for bbb in range(len(graphs)):
        GAR = (graphs[bbb] , actions[bbb] , rewards[bbb])
        buffer.append(GAR)
        data = graphs[bbb].xyz.float().cpu().numpy()
        if rewards[bbb]>0.8:
            aug = augmentationOnTheFly(data, actions[bbb], rewards[bbb] , num_augment=num_augment)
            for i in range(len(aug)):
                buffer.append(aug[i])
            


if load_params:
    GtG.load_state_dict(torch.load('GtG.pth'))
    graphs = torch.load("graphs_s.pt")
    rewards = torch.load("rewards_s.pt")
    actions = torch.load("actions_s.pt")
    
    for bbb in range(len(graphs)):
        GAR = (graphs[bbb] , actions[bbb] , rewards[bbb])
        buffer.append(GAR)

        

    #print(len(buffer))          
            
    # print(f"len buffer :{len(buffer)}")
    
    l_R = np.load("l_R.npy").tolist()
    real_reward=np.load("real_reward.npy").tolist()
    rrma=np.load("rrma.npy").tolist()
    
    bin_reward = np.load("bin_reward.npy").tolist()

    r2_list=np.load("r2.npy").tolist()
    r3_list=np.load("r3.npy").tolist()
    r4_list=np.load("r4.npy").tolist()
    
    num_passed = int(len(l_R))
    st_episode = int(num_passed * (update_int))

loss_function_GtG =  nn.MSELoss()
optimizer_GtG = AdamW(GtG.parameters(), lr=lr) 





pie=math.pi
sys.path.append('..')

########### code for api connection:
sim.simxFinish(-1)
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:
    print ('Connected to remote API server')
else:
    sys.exit('FAILED!')

#######################################################################################################
#### INITIALIZATION
#######################################################################################################
back1= o3d.io.read_point_cloud("back11.ply")
back2= o3d.io.read_point_cloud("back22.ply")
back3= o3d.io.read_point_cloud("back33.ply")

ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)

## origin
_ , origin = sim.simxGetObjectHandle(clientID,'origin',sim.simx_opmode_blocking)

## objects
_ , DISC =sim.simxGetObjectHandle(clientID,'RG2',sim.simx_opmode_blocking)

_ , cube =sim.simxGetObjectHandle(clientID,'Cuboid',sim.simx_opmode_blocking)
_ , cylinder =sim.simxGetObjectHandle(clientID,'mug',sim.simx_opmode_blocking)
_ , cup =sim.simxGetObjectHandle(clientID,'cup',sim.simx_opmode_blocking)
_ , bowl =sim.simxGetObjectHandle(clientID,'glass',sim.simx_opmode_blocking)
_ , ant =sim.simxGetObjectHandle(clientID,'ketchup',sim.simx_opmode_blocking)
_ , hexapod =sim.simxGetObjectHandle(clientID,'pawn',sim.simx_opmode_blocking)
_ , dumbbell =sim.simxGetObjectHandle(clientID,'fish',sim.simx_opmode_blocking)
_ , phone =sim.simxGetObjectHandle(clientID,'pipe',sim.simx_opmode_blocking)
_ , gear =sim.simxGetObjectHandle(clientID,'shampoo',sim.simx_opmode_blocking)
_ , doll =sim.simxGetObjectHandle(clientID,'hold',sim.simx_opmode_blocking)

    # positions 
_ , pos_DISC_0 =sim.simxGetObjectPosition(clientID, DISC ,origin ,sim.simx_opmode_streaming)    
    
_ , pos_cube_0 =sim.simxGetObjectPosition(clientID, cube ,origin ,sim.simx_opmode_streaming)
_ , pos_cylinder_0 =sim.simxGetObjectPosition(clientID, cylinder ,origin ,sim.simx_opmode_streaming)
_ , pos_cup_0 =sim.simxGetObjectPosition(clientID, cup ,origin ,sim.simx_opmode_streaming)
_ , pos_bowl_0 =sim.simxGetObjectPosition(clientID, bowl ,origin ,sim.simx_opmode_streaming)
_ , pos_ant_0 =sim.simxGetObjectPosition(clientID, ant ,origin ,sim.simx_opmode_streaming)
_ , pos_hexapod_0 =sim.simxGetObjectPosition(clientID, hexapod ,origin ,sim.simx_opmode_streaming)
_ , pos_dumbbell_0 =sim.simxGetObjectPosition(clientID, dumbbell ,origin ,sim.simx_opmode_streaming)
_ , pos_phone_0 =sim.simxGetObjectPosition(clientID, phone ,origin ,sim.simx_opmode_streaming)
_ , pos_gear_0 =sim.simxGetObjectPosition(clientID, gear ,origin ,sim.simx_opmode_streaming)
_ , pos_doll_0 =sim.simxGetObjectPosition(clientID, doll ,origin ,sim.simx_opmode_streaming)

time.sleep(0.1)




_ , pos_cube_0 =sim.simxGetObjectPosition(clientID, cube ,origin ,sim.simx_opmode_buffer)
pos_cube_0 = [-2.174999475479126, -2.0500004291534424, 0.04072770103812218+0.01]
ret=sim.simxSetObjectPosition(clientID,cube,origin,pos_cube_0,sim.simx_opmode_oneshot)

_ , pos_cylinder_0 =sim.simxGetObjectPosition(clientID, cylinder ,origin ,sim.simx_opmode_buffer)
pos_cylinder_0 = [-1.9723215103149414, -1.8744534254074097, 0.036282822489738464+0.01]
ret=sim.simxSetObjectPosition(clientID,cylinder,origin,pos_cylinder_0,sim.simx_opmode_oneshot)

_ , pos_cup_0 =sim.simxGetObjectPosition(clientID, cup ,origin ,sim.simx_opmode_buffer)
pos_cup_0 = [-1.7724988460540771, -1.876222014427185, 0.02938552387058735+0.01]
ret=sim.simxSetObjectPosition(clientID,cup,origin,pos_cup_0,sim.simx_opmode_oneshot)

_ , pos_bowl_0 =sim.simxGetObjectPosition(clientID, bowl ,origin ,sim.simx_opmode_buffer)
pos_bowl_0 = [-1.9281619787216187, -1.5464086532592773, 0.05156950652599335+0.01]
ret=sim.simxSetObjectPosition(clientID,bowl,origin,pos_bowl_0,sim.simx_opmode_oneshot)

_ , pos_ant_0 =sim.simxGetObjectPosition(clientID, ant ,origin ,sim.simx_opmode_buffer)
pos_ant_0 = [-2.1764090061187744, -1.550096035003662, 0.08076424896717072+0.01]
ret=sim.simxSetObjectPosition(clientID,ant,origin,pos_ant_0,sim.simx_opmode_oneshot)

_ , pos_hexapod_0 =sim.simxGetObjectPosition(clientID, hexapod ,origin ,sim.simx_opmode_buffer)
pos_hexapod_0 = [-2.229705333709717, -1.8401964902877808, 0.022270258516073227+0.01]
ret=sim.simxSetObjectPosition(clientID,hexapod,origin,pos_hexapod_0,sim.simx_opmode_oneshot)

_ , pos_dumbbell_0 =sim.simxGetObjectPosition(clientID, dumbbell ,origin ,sim.simx_opmode_buffer)
pos_dumbbell_0 =[-1.5080348253250122, -1.7359908819198608, 0.02771659754216671+0.01]
ret=sim.simxSetObjectPosition(clientID,dumbbell,origin,pos_dumbbell_0,sim.simx_opmode_oneshot)

_ , pos_phone_0 =sim.simxGetObjectPosition(clientID, phone ,origin ,sim.simx_opmode_buffer)
pos_phone_0 = [-1.7765891551971436, -1.5975565910339355, 0.049156058579683304+0.01]
ret=sim.simxSetObjectPosition(clientID,phone,origin,pos_phone_0,sim.simx_opmode_oneshot)

_ , pos_gear_0 =sim.simxGetObjectPosition(clientID, gear ,origin ,sim.simx_opmode_buffer)
pos_gear_0 = [-1.9736990928649902, -2.1018004417419434, 0.08564354479312897+0.01]
ret=sim.simxSetObjectPosition(clientID,gear,origin,pos_gear_0,sim.simx_opmode_oneshot)

_ , pos_doll_0 =sim.simxGetObjectPosition(clientID, doll ,origin ,sim.simx_opmode_buffer)
pos_doll_0 = [-1.3792705535888672, -1.543818712234497, 0.032729584723711014+0.01]
ret=sim.simxSetObjectPosition(clientID,doll,origin,pos_doll_0,sim.simx_opmode_oneshot)



## vision setup
_ , cam1 = sim.simxGetObjectHandle(clientID,'Vision_sensora',sim.simx_opmode_blocking)
_ , cam2 = sim.simxGetObjectHandle(clientID,'Vision_sensorb',sim.simx_opmode_blocking)
_ , cam3 = sim.simxGetObjectHandle(clientID,'Vision_sensorc',sim.simx_opmode_blocking)
#_ , cam4 = sim.simxGetObjectHandle(clientID,'Vision_sensord',sim.simx_opmode_blocking)
returnCode,resolution1,rgb_img1=sim.simxGetVisionSensorImage(clientID,cam1,0,sim.simx_opmode_streaming)
returnCode,resolution1,d_img1=sim.simxGetVisionSensorDepthBuffer(clientID,cam1,sim.simx_opmode_streaming)
returnCode,resolution2,rgb_img2=sim.simxGetVisionSensorImage(clientID,cam2,0,sim.simx_opmode_streaming)
returnCode,resolution2,d_img2=sim.simxGetVisionSensorDepthBuffer(clientID,cam2,sim.simx_opmode_streaming)
returnCode,resolution3,rgb_img3=sim.simxGetVisionSensorImage(clientID,cam3,0,sim.simx_opmode_streaming)
returnCode,resolution3,d_img3=sim.simxGetVisionSensorDepthBuffer(clientID,cam3,sim.simx_opmode_streaming)
#returnCode,resolution4,rgb_img4=sim.simxGetVisionSensorImage(clientID,cam4,0,sim.simx_opmode_streaming)
#returnCode,resolution4,d_img4=sim.simxGetVisionSensorDepthBuffer(clientID,cam4,sim.simx_opmode_streaming)

## Proximity Sensor
_ , prox = sim.simxGetObjectHandle(clientID,'Proximity',sim.simx_opmode_blocking)

## IK setup
_ , ik = sim.simxGetObjectHandle(clientID,'Dummy',sim.simx_opmode_blocking)
_ , z_axis = sim.simxGetObjectHandle(clientID,'z_axis',sim.simx_opmode_blocking)
_ , y_axis = sim.simxGetObjectHandle(clientID,'y_axis',sim.simx_opmode_blocking)
_ , x_axis = sim.simxGetObjectHandle(clientID,'x_axis',sim.simx_opmode_blocking)
_ , r_axis = sim.simxGetObjectHandle(clientID,'r_axis',sim.simx_opmode_blocking)
## Time Setup
t= time.time()
######################################################################################################3
ret =sim.simxSetJointTargetPosition(clientID,z_axis,0,sim.simx_opmode_streaming)
ret =sim.simxSetJointTargetPosition(clientID,y_axis,0,sim.simx_opmode_streaming)
ret =sim.simxSetJointTargetPosition(clientID,x_axis,0,sim.simx_opmode_streaming)
ret =sim.simxSetJointTargetPosition(clientID,r_axis,0,sim.simx_opmode_streaming)

a=3

###initialization



p_origin=[0,0,0]
#ret=sim.simxSetObjectPosition(clientID,origin,origin,p_origin,sim.simx_opmode_oneshot)

o_origin=[0,0,0]
#ret=sim.simxSetObjectOrientation(clientID,origin,origin,o_origin,sim.simx_opmode_oneshot)
_ , pos = sim.simxGetObjectPosition(clientID,ik, origin,sim.simx_opmode_streaming)
_ , pos = sim.simxGetJointPosition(clientID,z_axis,sim.simx_opmode_streaming)
#print(pos_cylinder1)


RG2_open(clientID , string_data="RG2_open")
gripper_move(clientID, 0 , 0 , 1 , 0, x_axis , y_axis , z_axis , r_axis)
time.sleep(0.5)
a , b , c , d ,e =sim.simxReadProximitySensor(clientID,prox,sim.simx_opmode_streaming)

bin_train=[]
bin_val=[]

import time
for ii in range(st_episode , episodes):
    print(f"******************* episode: {ii+1} ********************")
    t0=time.time()

    if ii>=10000 and ii<12000:
        random_action = True
        action_type="Random"
        sigma = 0.02
    
    elif ii>=20000 and ii<22000:
        random_action = True
        action_type="Random"
        sigma = 0.02
    
    elif ii>=30000 and ii<32000:
        random_action = True
        action_type="Random"
        sigma = 0.02
    
    else:
        random_action=False
        action_type = "Deterministic"
        
        
    random_angle = np.random.random() * 2*pie
    
    if ii%1== 0 :
        ret = sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        time.sleep(0.3)
        ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        #ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        #ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        #time.sleep(0.1)
        # ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        # ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        # ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        # ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
        
    ## objects to initial position :
    ret=sim.simxSetObjectPosition(clientID,cube,origin,pos_cube_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,cylinder,origin,pos_cylinder_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,cup,origin,pos_cup_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,bowl,origin,pos_bowl_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,ant,origin,pos_ant_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,hexapod,origin,pos_hexapod_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,dumbbell,origin,pos_dumbbell_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,phone,origin,pos_phone_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,gear,origin,pos_gear_0,sim.simx_opmode_oneshot)
    ret=sim.simxSetObjectPosition(clientID,doll,origin,pos_doll_0,sim.simx_opmode_oneshot)
  
    if ii%num_objects == 0 :
        loc = [ 0 , 0 , pos_cube_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = cube
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)


    elif ii%num_objects == 1 :
        loc = [ 0 , 0 , pos_cylinder_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = cylinder
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

    elif ii%num_objects == 2 :
        loc = [ 0 , 0 , pos_cup_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = cup
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

    elif ii%num_objects == 3 :
        loc = [ 0 , 0 , pos_bowl_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = bowl
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
    elif ii%num_objects == 4 :
        loc = [ 0 , 0 , pos_ant_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = ant
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
    elif ii%num_objects == 5 :
        loc = [ 0 , 0 , pos_hexapod_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = hexapod
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

    
    elif ii%num_objects == 6 :
        loc = [ 0 , 0 , pos_dumbbell_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = dumbbell
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

    
    elif ii%num_objects == 7 :
        loc = [ 0 , 0 , pos_phone_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = phone
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

    
    elif ii%num_objects == 8 :
        loc = [ 0 , 0 , pos_gear_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = gear
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
    elif ii%num_objects == 9 :
        loc = [ 0 , 0 , pos_doll_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = doll
        ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
        _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

       
     
    

    object_height=[]
    object_x=[]
    object_y=[]
    #object_height.append(pos_cylinder1[2])
    # Trial :
        
    #1) Gripper is open and at 0 , 0 , 1 :
    RG2_open(clientID , string_data="RG2_open")
    gripper_move(clientID, 0 , 0 , 1 , 0, x_axis , y_axis , z_axis , r_axis)
    
    #2) Generate graph and feed it to DisActRew :
    
    time.sleep(0.03)
    
    pcd , data = get_pcd(clientID=clientID , cam1 = cam1 , cam2 = cam2 , cam3 =cam3 , back1 =back1, back2=back2 , back3=back3 ,plot=False, est_nmls=False , down_smp=True , vox_sz=0.005)
    #o3d.visualization.draw_geometries([pcd])
    while data.shape[0] == 0 :
        ret = sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
        time.sleep(0.03)
        ret = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot)
      
        if ii%num_objects == 0 :
            loc = [ 0 , 0 , pos_cube_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = cube
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)


        elif ii%num_objects == 1 :
            loc = [ 0 , 0 , pos_cylinder_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = cylinder
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        elif ii%num_objects == 2 :
            loc = [ 0 , 0 , pos_cup_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = cup
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        elif ii%num_objects == 3 :
            loc = [ 0 , 0 , pos_bowl_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = bowl
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

            
        elif ii%num_objects == 4 :
            loc = [ 0 , 0 , pos_ant_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = ant
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

            
        elif ii%num_objects == 5 :
            loc = [ 0 , 0 , pos_hexapod_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = hexapod
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
        elif ii%num_objects == 6 :
            loc = [ 0 , 0 , pos_dumbbell_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = dumbbell
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
        elif ii%num_objects == 7 :
            loc = [ 0 , 0 , pos_phone_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = phone
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

        
        elif ii%num_objects == 8 :
            loc = [ 0 , 0 , pos_gear_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = gear
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

            
        elif ii%num_objects == 9 :
            loc = [ 0 , 0 , pos_doll_0[2] ]
            ort =[0,0,random_angle]
            cylinder1 = doll
            ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
            ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
            _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)

           
        
        

        pcd , data = get_pcd(clientID=clientID , cam1 = cam1 , cam2 = cam2 , cam3 =cam3 , back1 =back1, back2=back2 , back3=back3 ,plot=False, est_nmls=False , down_smp=True , vox_sz=0.005)
    g = G_maker(data)
    Graph = from_networkx(g).to(device)
    
    GtG.eval()
    

    
    action_set , _  = random_search_random_topk_picking(Graph , data , GtG , num_samples=50000 ,k=k , random_action=random_action,variance=sigma)
  
    GtG = GtG.to(device)
    for qq in range(num_components):

        RG2_open(clientID , string_data="RG2_open")
        gripper_move(clientID, 0 , 0 , 1 , 0, x_axis , y_axis , z_axis , r_axis)
    
        
        time.sleep(0.01)
        
        action = action_set

        action = action.unsqueeze(0).to(device) 
            
        ba = torch.zeros(Graph.xyz.float().shape[0] , device = device , dtype=torch.int64)
        est_reward , feat_vec  = GtG(Graph.xyz.float() , Graph.edge_index , ba, action)
        xx , yy , zz , rr = action.detach().cpu().numpy()[0]
        
        rr = float("{:.3f}".format(rr))
        xx = float("{:.3f}".format(xx))
        yy = float("{:.3f}".format(yy))
        zz = float("{:.3f}".format(zz))
        zz = float("{:.3f}".format(zz))
        rr = float("{:.3f}".format(rr))
        if zz<=0.01:
            zz=0.01

        
        action = torch.tensor((xx,yy,zz,rr) , dtype=torch.float32 , device=device).unsqueeze(0)
        
        print(f"action: {xx , yy , zz , rr}")
        r1 = reward3(data, np.array([xx,yy,zz]))
        if (abs(xx)>0.99*1) or (abs(yy)>0.99*1) or (abs(zz)>0.99*1)  :
            r2=0
            r3=0

            r4=0
            
        else:
            
            
            #3) Go to action position :
            gripper_move(clientID, xx , yy , zz , rr, x_axis , y_axis , z_axis , r_axis)
            time.sleep(1.0)
                
                #4) Check to see if proximity sensor is activated :
            a , b , c , d ,e =sim.simxReadProximitySensor(clientID,prox,sim.simx_opmode_buffer)
            time.sleep(0.02)
            #print(b)
        
            r2 = 0
            r3 = 0
            r4 = 0

        
            if True :
                
                #5) Close the gripper :
                if b:
                    RG2_close(clientID , string_data="RG2_open")
                    time.sleep(0.5)
                    r3=1
                    #6) Pick-up the object for 1 meter :
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    gripper_move(clientID, xx , yy , zz+1 , rr, x_axis , y_axis , z_axis , r_axis)
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    time.sleep(0.5)
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    _ , pos_cylinder1 =sim.simxGetObjectPosition(clientID, cylinder1 ,origin ,sim.simx_opmode_buffer)
                    object_height.append(pos_cylinder1[2])
                    object_x.append(pos_cylinder1[0])
                    object_y.append(pos_cylinder1[1])
                    r2 = min( reward2(min(object_height) , max(object_height)) , 1)*b

                    if abs(min(object_height)- max(object_height))>1.1 :
                        r2=0
                        r4=0
        
        #7) Construct reward value   
        reward = min(( 0.1*r3 + 0.3*r1 + 0.6*r2) , 1)#*
        print("r : " , float("{:.3f}".format(reward)) ," ; r^ :" , float("{:.3f}".format(est_reward.detach().cpu().numpy()[0][0])) , " ; feat_vec normL1: " ,float("{:.3f}".format(feat_vec.norm(1))) ," ; action_type: " , action_type )
        reward_error = abs(reward - est_reward.detach().cpu().numpy()[0][0])
        
        if reward>0.7:
            binary_rew=1
        else:
            binary_rew=0
            
        if True:
            real_reward.append(reward)
            bin_train.append(binary_rew)
            reward = torch.tensor([reward] , dtype=torch.float).to(device)
            Graph = Data(xyz=Graph.xyz.float(), edge_index=Graph.edge_index ,  num_nodes=Graph.xyz.float().shape[0] )
            GAR = (Graph , action , reward)
            buffer.append(GAR)
            
            if augment_active :
                aug = augmentationOnTheFly(data, action, reward , num_augment=num_augment)
                for i in range(len(aug)):
                    buffer.append(aug[i])
            
            
        
            
    losses_R = []
    losses_A = []
    gtg_recent=GtG.state_dict()

    samples_per_epoch = max(min_sample_per_epoch , len(buffer))
    samples_per_epoch = min(samples_per_epoch , max_samples_per_epoch)
    samples_per_epoch = btch

    if (ii+1)%update_int==0:

            for i in range(num_epochs):
                batch0=buffer.sample(batch_size = samples_per_epoch)
                batch0 = zip(*batch0)
                G , A , R = batch0
                G1 = DataLoader (G , batch_size = 1 ,shuffle=False)
                
                r_true=torch.zeros(samples_per_epoch)
                r_hat=torch.zeros(samples_per_epoch)
                for idx , G in enumerate(G1):
                    GtG.train()
                    ba = torch.zeros(G.xyz.float().shape[0] , device = device , dtype=torch.int64)
                    predicted_reward , _ = GtG(G.xyz.float() , G.edge_index , ba , A[idx])
                    target_reward = R [idx]
                    r_hat[idx]= predicted_reward[0]
                    r_true[idx] = target_reward
                loss_R = loss_function_GtG(r_hat, r_true)
                optimizer_GtG.zero_grad()
                loss_R.backward(retain_graph=True)

                optimizer_GtG.step()

            
                losses_R.append(loss_R.item())
                

            
            print("GtG")
            for name, param in GtG.named_parameters():
                if param.grad is not None:
                    print(f'Parameter: {name}, Gradient norm: {param.grad.norm(1).item()}')
            
            l_R.append(sum(losses_R)/len(losses_R) )
    
            
            plt.figure(figsize=(40,40))
            plt.subplot(3,1,1)
            plt.plot(l_R)    
            plt.title ( " Loss R ")
            
            plt.subplot(3,1,2)
            rr=real_reward[- int(update_int) :]
            print("train avg reward = " ,sum(rr)/len(rr) )
            rrma.append(sum(rr)/len(rr))
            print("max train reward = " , max(rrma))
            
            rr=bin_train[- int(update_int) :]
            print("train avg  binary reward = " ,sum(rr)/len(rr) )
            bin_reward.append(sum(rr)/len(rr))
            print("max binary reward = " , max(bin_reward))
            plt.plot(rrma)    
            plt.title ( " Mean Continous Reward")
            
            plt.subplot(3,1,3)
            plt.plot(bin_reward)    
            plt.title ( " Mean Binary Reward")
            
            
            plt.show()
            

            
            
            
            
            if save_params:
                
                rrma_array = np.array(rrma)
                l_R_array = np.array(l_R)
                rr_array = np.array(rr)
                real_reward_array = np.array(real_reward)
                val_reward_array = np.array(val_reward)
                val_rrma_array = np.array(val_rrma)
                
                r2_array = np.array(r2_list)
                r3_array = np.array(r3_list)
                r4_array = np.array(r4_list)
                
                val_r2_array = np.array(val_r2_list)
                val_r3_array = np.array(val_r3_list)
                val_r4_array = np.array(val_r4_list)
                
                bin_reward_array = np.array(bin_reward)
                val_bin_reward_array = np.array(val_bin_reward)
                
                gtgname = 'GtG'+str(ii+1)+".pth"
                torch.save(GtG.state_dict(), gtgname)
                torch.save(GtG.state_dict(), 'GtG.pth')
                
                np.save("bin_reward.npy" , bin_reward_array)

                
                np.save("rrma.npy" , rrma_array )
                np.save("l_R.npy" , l_R_array)
                np.save("rr.npy" , rr_array)
                np.save("real_reward.npy" , real_reward_array)
                
                np.save("r2.npy" , r2_array)
                np.save("r3.npy" , r3_array)
                np.save("r4.npy" , r4_array)
                

                
                graphs_s , actions_s , rewards_s = zip(*buffer.all_buffer())

                torch.save(graphs_s, 'graphs_s.pt')
                
                torch.save(actions_s, 'actions_s.pt')
                
                torch.save(rewards_s, 'rewards_s.pt')
                
                rep_rew = np.zeros(len(rewards_s))
                
                for u in range(len(rewards_s)):
                    rep_rew[u] = rewards_s[u].cpu().numpy()
                plt.figure()
                plt.hist(rep_rew)
                plt.title("Distribution of Replay Buffer Rewards")
                plt.show()
                
                          
        
