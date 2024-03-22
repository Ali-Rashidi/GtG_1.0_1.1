# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:15:38 2023

@author: Ali
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 21:50:13 2023

@author: Ali
"""

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
import warnings
warnings.filterwarnings("ignore")

num_series=1
pp=2
k=3
st_episode = 0
num_test=20
#batch_size = 128
episodes = 20000
explore_episode=2000
capacity=10000
sigma_start = 0.2
sigma_step = 0.005
min_sample_per_epoch=10
max_samples_per_epoch=100
inference = False

load_params = True
save_params = False
lr=1e-12 #2000 aval 1e-4 , bad -6 , az 7500 -7 and buffer faghat motenaseb ba khata
#az 9100 -9
num_epochs = 10
num_components = 1

num_train_objects=10
num_val_objects = 0
update_int = 10
num_objects = 10
solver_itr = 10
augment_active = False
num_augment = 18
sigma=0
buffer = ReplayBuffer(capacity=capacity)
#sim.simxSetObjectIntParameter(clientID, cube2, 3004, 0, sim.simx_opmode_blocking)

## Network Config :
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

GtG = holyGrail().to(device)
#
print( " GtG Params : " + str ( sum(p.numel() for p in GtG.parameters())))
GtG.load_state_dict(torch.load('GtG_best.pth'))
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
#### INIRIALIZATION
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

_ , test_object =sim.simxGetObjectHandle(clientID,'test_barClamp',sim.simx_opmode_blocking)
#_ , test_object =sim.simxGetObjectHandle(clientID,'test_bakingSoda',sim.simx_opmode_blocking)
#_ , test_object =sim.simxGetObjectHandle(clientID,'test_turbine',sim.simx_opmode_blocking)
#_ , test_object =sim.simxGetObjectHandle(clientID,'test_dog',sim.simx_opmode_blocking)
#_ , test_object =sim.simxGetObjectHandle(clientID,'test_chickenSoup',sim.simx_opmode_blocking)

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

_ , pos_test_object_0 =sim.simxGetObjectPosition(clientID, test_object ,origin ,sim.simx_opmode_streaming)
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


_ , pos_test_object_0 =sim.simxGetObjectPosition(clientID, test_object ,origin ,sim.simx_opmode_buffer)
test_object_0 = [2.121164321899414, 2.075348138809204, 0.02563353255391121 +0.03]
ret=sim.simxSetObjectPosition(clientID,test_object,origin,pos_doll_0,sim.simx_opmode_oneshot)


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



# p_cube1=[0 , 0 , 5e-2]
# ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,p_cube1,sim.simx_opmode_oneshot)

# o_cube1=[0,0,0]
# ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,o_cube1,sim.simx_opmode_oneshot)


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
sr=[]
rrr=[]
import time
for ttt in range(num_series):
    print(f'test num :{ttt+1}')
    for ii in range(num_test):
        print(f"******************* episode: {ii+1} ********************")
        t0=time.time()
    
        random_action=False
        action_type = "Deterministic"
            
        ang_list = np.arange(0 , np.pi , (np.pi/num_test) )
        random_angle = ang_list[ii]
        
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
        # ret=sim.simxSetObjectPosition(clientID,cube,origin,pos_cube_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,cylinder,origin,pos_cylinder_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,cup,origin,pos_cup_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,bowl,origin,pos_bowl_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,ant,origin,pos_ant_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,hexapod,origin,pos_hexapod_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,dumbbell,origin,pos_dumbbell_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,phone,origin,pos_phone_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,gear,origin,pos_gear_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,doll,origin,pos_doll_0,sim.simx_opmode_oneshot)
        # ret=sim.simxSetObjectPosition(clientID,test_object,origin,pos_test_object_0,sim.simx_opmode_oneshot)
      
        
        loc = [ 0 , 0 , pos_test_object_0[2] ]
        ort =[0,0,random_angle]
        cylinder1 = test_object
        #ret=sim.simxSetObjectPosition(clientID,cylinder1,origin,loc,sim.simx_opmode_oneshot)
        #ret=sim.simxSetObjectOrientation(clientID,cylinder1,origin,ort,sim.simx_opmode_oneshot)
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
        
        pcd , data , pcd1_d , pcd2_d , pcd3_d= get_pcd(clientID=clientID , cam1 = cam1 , cam2 = cam2 , cam3 =cam3 , back1 =back1, back2=back2 , back3=back3 ,plot=False, est_nmls=False , down_smp=False , vox_sz=0.005)
        point_cloud = pcd
        
        o3d.visualization.draw_geometries([pcd])
        t0=time.time()
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
        print(f"time: {time.time()-t0}")
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")


        mask = np.where(labels==np.random.choice(max_label+1))
        
        data = data [mask]
        data = data[:,:3]

        mean=np.mean(data,axis=0)
        mean =np.array([float(mean[0]),float(mean[1]),0]) 
        data=data-mean
        
        
        g = G_maker(data)
        Graph = from_networkx(g).to(device)
        #print(nx.info(g))
        #nx.draw(g)
        #Graph = Data(xyz=Graph.xyz.float(), edge_index=Graph.edge_index)
        GtG.eval()
        
        #action_set , reward_set = random_search(Graph , data , GtG , num_samples=20000 , random_action=random_action)
        #print(reward_set)
        
        #action_set , _  = random_search(Graph , data , GtG , num_samples=50000 , random_action=random_action,variance=sigma)
        t0=time.time()
        action_set , _  = random_search_random_topk_picking(Graph , data , GtG , num_samples=50000 ,k=k , random_action=random_action,variance=sigma,rank=True,pp=pp)
        print(f"action time: {time.time()-t0}")
        
        GtG = GtG.to(device)
        for qq in range(num_components):
    
            RG2_open(clientID , string_data="RG2_open")
            gripper_move(clientID, 0 , 0 , 1 , 0, x_axis , y_axis , z_axis , r_axis)
            
            #2) Generate graph and feed it to DisActRew :
            
            time.sleep(0.01)
            
            action = action_set
            #print(action)
            #print(f"init action {action}")
            action = action.unsqueeze(0).to(device) 
                
            ba = torch.zeros(Graph.xyz.float().shape[0] , device = device , dtype=torch.int64)
            est_reward , feat_vec  = GtG(Graph.xyz.float() , Graph.edge_index , ba, action)
            xx , yy , zz , rr = action.detach().cpu().numpy()[0]
            
            rr = float("{:.3f}".format(rr))
            xx = float("{:.3f}".format(xx)) + mean[0]
            yy = float("{:.3f}".format(yy)) + mean[1]
            zz = float("{:.3f}".format(zz))
            if zz==0.01:
                zz=0.015
    
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
                time.sleep(1)
                    
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
                        #r4 = min((abs(min(object_x) - max(object_x)) + abs(min(object_y) - max(object_y)))*10 , 1)
                        #r4 = max(r2 , r4)
                        
                        #print(object_height)
                        if abs(min(object_height)- max(object_height))>1.1 :
                            r2=0
                            r4=0
            
            #7) Construct reward value   
            reward = min(( 0.1*r3 + 0.3*r1 + 0.6*r2) , 1)#*np.random.normal(1 , 0.05) , 1) #+ 0.15*r3 # can be scalar or vector
            print("r : " , float("{:.3f}".format(reward)) ," ; r^ :" , float("{:.3f}".format(est_reward.detach().cpu().numpy()[0][0])) , " ; feat_vec normL1: " ,float("{:.3f}".format(feat_vec.norm(1))) ," ; action_type: " , action_type )
            reward_error = abs(reward - est_reward.detach().cpu().numpy()[0][0])
            
            if reward>0.7:
                binary_rew=1
            else:
                binary_rew=0
                
            break
            real_reward.append(reward)
            bin_train.append(binary_rew)
    print(f"Success: {sum(bin_train)/num_test}")
    print(f"Mean Reward: {sum(real_reward)/num_test}")
    sr.append(sum(bin_train)/num_test)
    rrr.append(sum(real_reward)/num_test)
    real_reward=[]
    bin_train=[]

print(sr ,  np.mean(sr) , np.sqrt(np.var(sr)))
print(rrr, np.mean(rrr) , np.sqrt(np.var(rrr)))


        


            
            
            

            
        
                 
                          
        
