# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 10:37:07 2023

@author: Ali
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import sim
import time
import open3d as o3d
import random
import copy
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from collections import deque, namedtuple
from torch import nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW
from sklearn.neighbors import NearestNeighbors

## functions

def get_pcd(clientID, cam1 , cam2 , cam3  , back1 , back2 , back3 , est_nmls=False , down_smp=False , vox_sz=0.02 , plot=False):
    w = 256
    h = w
    fx = 221.7
    fy=fx
    cx=128
    cy=cx

    w = 256
    h = w
    fx = 221.7
    fy=fx
    cx=128
    cy=cx
    
    
    a = 16.5
    b= -19  
    
    a = -1
    b= -1
    
    s=0.5/2.451232142516857e-05 
    
    c=-1
    d =1
    
    e =1.04
    f = 0.95
    g = 2
    
    e =0.985
    f =1.03
    g =1
    
    i =1
    j =-0.07
    k = 1 
    
    x1 = 0.5*e/ s *c
    y1 = 0.0*g /s *d
    z1 = 0.40000000059604645*f /s
    
    x2 = -0.2500000596046448/s *c
    y2 = 0.4330126941204071/s *d
    z2 = 0.4000000059604645/s
    
    x3 = -0.24999992549419403/s *c
    y3 = -0.4330127537250519/s *d
    z3 = 0.4000000059604645/s
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx,fy, cx, cy)
    intrinsic.intrinsic_matrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    cam_in = o3d.camera.PinholeCameraParameters()
    cam_in.intrinsic = intrinsic
    
    
    ## Camera Output:
    returnCode,resolution,rgb_img1=sim.simxGetVisionSensorImage(clientID,cam1,0,sim.simx_opmode_buffer)  
    returnCode,resolution,d_img1=sim.simxGetVisionSensorDepthBuffer(clientID,cam1,sim.simx_opmode_buffer)
    returnCode,resolution,rgb_img2=sim.simxGetVisionSensorImage(clientID,cam2,0,sim.simx_opmode_buffer)  
    returnCode,resolution,d_img2=sim.simxGetVisionSensorDepthBuffer(clientID,cam2,sim.simx_opmode_buffer)
    returnCode,resolution,rgb_img3=sim.simxGetVisionSensorImage(clientID,cam3,0,sim.simx_opmode_buffer)  
    returnCode,resolution,d_img3=sim.simxGetVisionSensorDepthBuffer(clientID,cam3,sim.simx_opmode_buffer)
    #returnCode,resolution,rgb_img4=sim.simxGetVisionSensorImage(clientID,cam4,0,sim.simx_opmode_buffer)  
    #returnCode,resolution,d_img4=sim.simxGetVisionSensorDepthBuffer(clientID,cam4,sim.simx_opmode_buffer)
   
    
    ## rgb
    
    rgb_img1=np.array(rgb_img1, dtype=np.uint8)
    rgb_img1=np.reshape(rgb_img1, (resolution[0],resolution[1],3))
    rgb_img2=np.array(rgb_img2, dtype=np.uint8)
    rgb_img2=np.reshape(rgb_img2, (resolution[0],resolution[1],3))
    rgb_img3=np.array(rgb_img3, dtype=np.uint8)
    rgb_img3=np.reshape(rgb_img3, (resolution[0],resolution[1],3))
    #rgb_img4=np.array(rgb_img4, dtype=np.uint8)
    #rgb_img4=np.reshape(rgb_img4, (resolution[0],resolution[1],3))
    
    
    ## Depth
    d_img1=np.array(d_img1,dtype=np.float32)
    d_img1=np.reshape(d_img1 , (resolution[0],resolution[1],1))
    d_img2=np.array(d_img2,dtype=np.float32)
    d_img2=np.reshape(d_img2 , (resolution[0],resolution[1],1))
    d_img3=np.array(d_img3,dtype=np.float32)
    d_img3=np.reshape(d_img3 , (resolution[0],resolution[1],1))
    #d_img4=np.array(d_img4,dtype=np.float32)
    #d_img4=np.reshape(d_img4 , (resolution[0],resolution[1],1))
    
    if plot :
        plt.figure()
        plt.subplot(321)
        plt.imshow(rgb_img1 , origin='lower')
        plt.title("RGB")
        
        plt.subplot(322)
        plt.imshow(d_img1 , origin='lower',cmap='gray')
        plt.title("Depth")
        
        plt.subplot(323)
        plt.imshow(rgb_img2 , origin='lower')
        plt.title("RGB")
        
        plt.subplot(324)
        plt.imshow(d_img2 , origin='lower',cmap='gray')
        plt.title("Depth")
        
        plt.subplot(325)
        plt.imshow(rgb_img3 , origin='lower')
        plt.title("RGB")
        
        plt.subplot(326)
        plt.imshow(d_img3 , origin='lower',cmap='gray')
        plt.title("Depth")
        
        plt.show()
        
    ### Creating PointClouds:
    rgbd1=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_img1), o3d.geometry.Image((d_img1).astype(np.float32)) , convert_rgb_to_intensity=False)
    pcd1=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, cam_in.intrinsic)
    #pcd1.transform([ [1,0,0,0] , [0,-1,0,0] ,[0,0,-1,0] , [0,0,0,1]])
    dist1=pcd1.compute_point_cloud_distance(back1)
    dist1=np.asarray(dist1)
    ind1=np.where(dist1>0.0000000001)[0]
    pcd11=pcd1.select_by_index(ind1)
    
    
    
    # pcd1_s = copy.deepcopy(pcd11)
    # pcd1_s.scale(s, center=(0,0,0))
    pcd1_s = pcd11
    pcd1_r = copy.deepcopy(pcd11)
    R1 = pcd1_r.get_rotation_matrix_from_xyz((-2.2268928050994873*i+j, 0.0, -1.5707963705062866*k))
    pcd1_r.rotate(np.transpose(R1), center=(0,0,0))
    pcd1_d = copy.deepcopy(pcd1_r).translate((x1, y1 , z1))
    pcd1_d.scale(s, center=(0,0,0))
    # pcd1_r.transform([ [1,0,0,0] , [0,1,0,0] ,[0,0,1,0] , [0,0,0,1]])
    
    rgbd2=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_img2), o3d.geometry.Image((d_img2).astype(np.float32)) , convert_rgb_to_intensity=False)
    pcd2=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, cam_in.intrinsic )
    #pcd2.transform([ [1,0,0,0] , [0,1,0,0] ,[0,0,-1,0] , [0,0,0,1]])
    dist2=pcd2.compute_point_cloud_distance(back2)
    dist2=np.asarray(dist2)
    ind2=np.where(dist2>0.0000000001)[0]
    pcd22=pcd2.select_by_index(ind2)
    
    # pcd2_s = copy.deepcopy(pcd22)
    # pcd2_s.scale(s, center=(0,0,0))
    pcd2_s = pcd22  
    pcd2_r = copy.deepcopy(pcd22)
    R2 =  pcd2_r.get_rotation_matrix_from_xyz((-2.268928050994873, -5.960464477539063e-08, 0.5235989093780518))
    pcd2_r.rotate(np.transpose(R2), center= (0,0,0))
    pcd2_d = copy.deepcopy(pcd2_r).translate((x2, y2, z2))
    pcd2_d.scale(s, center=(0,0,0))
    
    # pcd2_r.transform([ [1,0,0,0] , [0,1,0,0] ,[0,0,1,0] , [0,0,0,1]])
             
    rgbd3=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_img3), o3d.geometry.Image((d_img3).astype(np.float32)) , convert_rgb_to_intensity=False)
    pcd3=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd3, cam_in.intrinsic)
    #pcd3.transform([ [1,0,0,0] , [0,1,0,0] ,[0,0,-1,0] , [0,0,0,1]])
    
    dist3=pcd3.compute_point_cloud_distance(back3)
    dist3=np.asarray(dist3)
    ind3=np.where(dist3>0.0000000001)[0]
    pcd33=pcd3.select_by_index(ind3)
    
    #o3d.visualization.draw_geometries([pcd])
    #pcd.transform([ [1,0,0,0] , [0,-1,0,0] ,[0,0,-1,0] , [0,0,0,1]])
    pcd3_s = copy.deepcopy(pcd33)
    pcd3_s.scale(s, center=(0,0,0))
    pcd3_s = pcd33
    pcd3_r = copy.deepcopy(pcd33)
    R3 = pcd3_r.get_rotation_matrix_from_xyz((-2.268928050994873, 1.4901161193847656e-08, 2.6179940700531006))
    pcd3_r.rotate(np.transpose(R3), center=(0,0,0))
    pcd3_d = copy.deepcopy(pcd3_r).translate((x3, y3, z3))
    pcd3_d.scale(s, center=(0,0,0))
    
    #pcd3_r.transform([ [1,0,0,0] , [0,1,0,0] ,[0,0,1,0] , [0,0,0,1]])
    pcd = o3d.geometry.PointCloud()
    data1 = np.round(np.asarray(pcd1_d.points),3)
    rgb1 = np.asarray(pcd1_d.colors)
    data1=np.concatenate([data1 , rgb1],axis=1)
    data2 = np.round(np.asarray(pcd2_d.points),3)
    rgb2 = np.asarray(pcd2_d.colors)
    data2=np.concatenate([data2 , rgb2],axis=1)
    data3 = np.round(np.asarray(pcd3_d.points),3)
    rgb3 = np.asarray(pcd3_d.colors)
    data3=np.concatenate([data3 , rgb3],axis=1)
    data = np.concatenate([data1,data2,data3],axis=0)
    #data = np.concatenate([data1,data2],axis=0)
    #data = np.concatenate([data1],axis=0)
    data[:,0]=-data[:,0]
    data[:,2]=np.round(data[:,2],2)
    
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data[:,3:])
    
    if down_smp:
        pcd = pcd.voxel_down_sample(voxel_size=vox_sz)
    data = np.concatenate([np.asarray(pcd.points) , np.asarray(pcd.colors)], axis=1)
    #o3d.visualization.draw_geometries()
    #print(data.shape)
    if down_smp:
        if data.shape[0]>=100:
            ind = np.random.choice(data.shape[0] , 100 , replace=False)
            data=data[ind]
    
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data[:,3:])
    
    
    if est_nmls :
        pcd.estimate_normals()
        nmls= np.asarray(pcd.normals)
        data=np.concatenate((data,nmls),axis=1)
        
    #print(data.shape)
    return pcd , data[:,:3] #, pcd1_d , pcd2_d , pcd3_d

from sklearn.neighbors import kneighbors_graph
def custom_draw_geometry_with_rotation(pcd , lines):
    vis = o3d.visualization.Visualizer()

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0,10.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd,lines],
                                                              rotate_view)
    
def pcd_2_graph(pcd , degree=5):
    centroids=np.asarray(pcd.points)
    A = kneighbors_graph(centroids, degree)
    points = centroids.tolist()
    lines = np.transpose(np.nonzero(A.toarray()))
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    o3d.visualization.draw_geometries([line_set,pcd])
    return [line_set,pcd]
    #o3d.visualization.draw_geometries([line_set])
    
class ReplayBuffer:
    
    def __init__(self , capacity):
        self.buffer = deque(maxlen=capacity) #    it's like a list but manages its contents automaticly
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self , batch_size):
        return random.sample(self.buffer , batch_size)
    
    def all_buffer (self):
        return self.buffer
    
    
class RLDataset(IterableDataset):
    
    def __init__ (self , buffer , sample_size = 200):
        self.buffer = buffer
        self.sample_size = sample_size
        
    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience # returns by request of pytorch

## note : feature haro concat kon , shayad behtar bashe edge feature ro bikhial shi
def G_maker(data , dist_th=0.04 , node_degree = 5 , dist_base = False , deg_base=True):
    points=data
    num_points = points.shape[0]
    xyz = points[:, :3] # and normals ### faghat xyz
    #normals = points[:, 3:]
    
    # Create a graph and add nodes with features xyz and normals
    graph = nx.Graph()
    for i in range(num_points):
        node_features = {'xyz': xyz[i]}
        graph.add_node(i, **node_features)
    
    # Add edges with features euclidean distance and cosine of angle between normals
    if dist_base:
        for i in range(num_points):
            for j in range(i+1, num_points):
                distance = np.linalg.norm(xyz[i] - xyz[j])
                #angle = np.dot(normals[i], normals[j]) / (np.linalg.norm(normals[i]) * np.linalg.norm(normals[j]))
                #edge_features = {'distance': distance, 'angle': angle}
                if distance <= dist_th :
                    #graph.add_edge(i, j, **edge_features)
                    graph.add_edge(i, j)
    
    if deg_base:
        kdtree = NearestNeighbors(n_neighbors=node_degree+1, algorithm='kd_tree')
        kdtree.fit(xyz)
        for i, point in enumerate(xyz):
            _, neighbor_indices = kdtree.kneighbors([point], node_degree+1)
            
            for neighbor_index in neighbor_indices[0]:
                if i != neighbor_index:
                    graph.add_edge(i, neighbor_index)
        
    return graph  
  
def reward1(data , gripper , alpha=12 , beta=0.25):  
    m = np.mean(data[:,:3] , axis=0)
    xyz = gripper[:3]
    d=m-xyz
    d=np.linalg.norm(d)
    r = 1 / (1+np.exp(alpha * (d-beta)))+0.05
    
    return r

def reward2(z0 , zm , z_th=1 , alpha=15 , beta=0.9):
    l = zm-z0
    L = np.abs(z_th-l)
    
    #r = 1 / (1+np.exp(alpha * (L-beta)))
    r = np.abs(l)
    return r

def reward3(data , gripper,b=10):  
    xyz = gripper[:3]
    dist = data-xyz
    dist=np.linalg.norm(dist,axis=1)
    min_dist = min(dist)
    r = 1 / (1+min_dist*b)
    
    return r

def RG2_close(clientID , string_data="RG2_open"):
    ret = sim.simxSetIntegerSignal(clientID, string_data , 0, sim.simx_opmode_blocking)
    #print("closing >< ")
    
def RG2_open(clientID , string_data="RG2_open"):
    ret = sim.simxSetIntegerSignal(clientID, string_data , 1, sim.simx_opmode_blocking)
    #print("opening >   < ")
    
def gripper_move(clientID, x , y , z , r , x_axis , y_axis , z_axis , r_axis):
    
    zz=-1+z
    ret4 =sim.simxSetJointTargetPosition(clientID,r_axis, r ,sim.simx_opmode_blocking)
    #time.sleep(0.3)
    ret1 =sim.simxSetJointTargetPosition(clientID,x_axis, x ,sim.simx_opmode_blocking)
    ret2 =sim.simxSetJointTargetPosition(clientID,y_axis, y ,sim.simx_opmode_blocking)
    ret3 =sim.simxSetJointTargetPosition(clientID,z_axis, zz ,sim.simx_opmode_blocking)
    
    
  
############# Networks ########################    
import torch
from torch.nn import Linear , Dropout 
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn.norm import LayerNorm
#from torch_geometric.nn.pool import global_mean_pool as GAP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch.nn.init as init



def random_search(Graph , data , r_network , num_samples=20000 , device=device , num_dec = 2 , random_action=False , variance=0.04):
    
    rng = 0.02
    
    x_min_data = np.min(data[:,0])
    x_min = x_min_data - rng
    x_min = np.round(x_min , num_dec)

    x_max_data = np.max(data[:,0])
    x_max = x_max_data + rng
    x_max = np.round(x_max , num_dec)
    
    y_min_data = np.min(data[:,1])
    y_min = y_min_data -rng
    y_min = np.round(y_min , num_dec)
    
    y_max_data = np.max(data[:,1])
    y_max = y_max_data + rng
    y_max = np.round(y_max , num_dec)
    
    z_min_data = np.min(data[:,2])
    z_min = max( 0.01 , z_min_data -rng )
    z_min = np.round(z_min , num_dec)
    
    z_max_data = np.max(data[:,2])
    z_max = z_max_data
    z_max = np.round(z_max , num_dec)
    
    #print(x_min , x_min_data , x_max , x_max_data)
    ori_list = np.arange(0 , np.pi , (np.pi/18) )
    x_list = np.arange(x_min,x_max+0.01 , 0.01)
    y_list = np.arange(y_min,y_max+0.01 , 0.01)
    z_list = np.arange(z_min,z_max+0.01 , 0.01)
    #print(z_list)
    
    if len(x_list) *  len(y_list) *  len(z_list) *  len(ori_list) <= num_samples:
        print("exhaustive search")
        x,y,z,theta = np.meshgrid(x_list , y_list , z_list , ori_list ,  indexing='ij')
        
        x_data = x.flatten().reshape(-1,1)
        y_data = y.flatten().reshape(-1,1)
        z_data = z.flatten().reshape(-1,1)
        action_ori= theta.flatten().reshape(-1,1)

    
        
    else:
        print("random search")
        action_ori = np.random.choice(ori_list , num_samples).reshape(-1,1)
        x_data = np.random.choice(x_list , num_samples).reshape(-1,1)
        y_data = np.random.choice(y_list , num_samples).reshape(-1,1)
        z_data = np.random.choice(z_list , num_samples).reshape(-1,1)
        
    action_po = np.concatenate((x_data , y_data , z_data , action_ori) , axis = 1)
    
    #print(action_po.shape)
    action_po = torch.tensor(action_po , dtype = torch.float32 , device = device ) 
    
    #feat_vector = feat_vector.to(device).to(torch.float32).detach()
    
    r_network.eval()
    ba = torch.zeros(Graph.xyz.float().shape[0] , device = device , dtype=torch.int64)

    reward_tensor , _  = r_network(Graph.xyz.float() , Graph.edge_index , ba, action_po)
    
    best_ind = torch.argmax(reward_tensor)
    
    best_point = action_po[best_ind]
    
    if random_action:
        print("noisy action")
        best_point = best_point + (torch.normal(0, variance , best_point.size() , device=device)*torch.tensor([1,1,1,np.pi],device=device).resize(4))
    best_reward = reward_tensor[best_ind]
    
    return best_point , best_reward



def random_search_random_topk_picking(Graph , data , r_network , num_samples=20000,k=10 , device=device , num_dec = 2 , random_action=False , variance=0.04,rank=False,pp=1):
    
    rng = 0.02

    x_min_data = np.min(data[:,0])
    x_min = x_min_data - rng
    x_min = np.round(x_min , num_dec)

    x_max_data = np.max(data[:,0])
    x_max = x_max_data + rng
    x_max = np.round(x_max , num_dec)
    
    y_min_data = np.min(data[:,1])
    y_min = y_min_data -rng
    y_min = np.round(y_min , num_dec)
    
    y_max_data = np.max(data[:,1])
    y_max = y_max_data + rng
    y_max = np.round(y_max , num_dec)
    
    z_min_data = np.min(data[:,2])
    z_min = max( 0.01 , z_min_data -rng )
    z_min = np.round(z_min , num_dec)
    
    z_max_data = np.max(data[:,2])
    z_max = z_max_data
    z_max = np.round(z_max , num_dec)
    
    #print(x_min , x_min_data , x_max , x_max_data)
    ori_list = np.arange(0 , np.pi , (np.pi/18) )
    x_list = np.arange(x_min,x_max+0.01 , 0.01)
    y_list = np.arange(y_min,y_max+0.01 , 0.01)
    z_list = np.arange(z_min,z_max+0.01 , 0.01)
    #print(z_list)
    
    if len(x_list) *  len(y_list) *  len(z_list) *  len(ori_list) <= num_samples:
        #print("exhaustive search")
        x,y,z,theta = np.meshgrid(x_list , y_list , z_list , ori_list ,  indexing='ij')
        
        x_data = x.flatten().reshape(-1,1)
        y_data = y.flatten().reshape(-1,1)
        z_data = z.flatten().reshape(-1,1)
        action_ori= theta.flatten().reshape(-1,1)

    
        
    else:
        print("random search")
        action_ori = np.random.choice(ori_list , num_samples).reshape(-1,1)
        x_data = np.random.choice(x_list , num_samples).reshape(-1,1)
        y_data = np.random.choice(y_list , num_samples).reshape(-1,1)
        z_data = np.random.choice(z_list , num_samples).reshape(-1,1)
        
    action_po = np.concatenate((x_data , y_data , z_data , action_ori) , axis = 1)
    
    #print(action_po.shape)
    action_po = torch.tensor(action_po , dtype = torch.float32 , device = device ) 
    
    #feat_vector = feat_vector.to(device).to(torch.float32).detach()
    
    r_network.eval()
    ba = torch.zeros(Graph.xyz.float().shape[0] , device = device , dtype=torch.int64)

    reward_tensor , _  = r_network(Graph.xyz.float() , Graph.edge_index , ba, action_po)
    
    ind=torch.argsort(-reward_tensor , dim=0)
    ind = ind.squeeze(1)
    ind = ind[:k]
    k_rew = reward_tensor[ind]
    k_w = (k_rew / torch.sum(k_rew)).detach().cpu().numpy().reshape(-1)
    
    
    
    #based on reward
    #picked_element = np.random.choice(k , p=k_w)
    
    # ##random
    picked_element = np.random.choice(k)
    
    #rank
    # if rank:
    #     picked_element=pp-1
    print(f" picking {picked_element+1}th best action")
    
    best_ind = ind[picked_element]
    
    best_point = action_po[best_ind]
    
    if random_action:
        print("noisy action")
        best_point = best_point + (torch.normal(0, variance , best_point.size() , device=device)*torch.tensor([1,1,1,np.pi],device=device).resize(4))
    best_reward = reward_tensor[best_ind]
    
    return best_point , best_reward


def top_k_grasp(Graph , data , r_network , num_samples=20000 , device=device , num_dec = 2 , k = 5):
    
    rng = 0.02
    
    x_min_data = np.min(data[:,0])
    x_min = x_min_data - rng
    x_min = np.round(x_min , num_dec)

    x_max_data = np.max(data[:,0])
    x_max = x_max_data + rng
    x_max = np.round(x_max , num_dec)
    
    y_min_data = np.min(data[:,1])
    y_min = y_min_data -rng
    y_min = np.round(y_min , num_dec)
    
    y_max_data = np.max(data[:,1])
    y_max = y_max_data + rng
    y_max = np.round(y_max , num_dec)
    
    z_min_data = np.min(data[:,2])
    z_min = max( 0.01 , z_min_data -rng )
    z_min = np.round(z_min , num_dec)
    
    z_max_data = np.max(data[:,2])
    z_max = z_max_data
    z_max = np.round(z_max , num_dec)
    
    #print(x_min , x_min_data , x_max , x_max_data)
    ori_list = np.arange(0 , np.pi , (np.pi/18) )
    x_list = np.arange(x_min,x_max+0.01 , 0.01)
    y_list = np.arange(y_min,y_max+0.01 , 0.01)
    z_list = np.arange(z_min,z_max+0.01 , 0.01)
    
    if len(x_list) *  len(y_list) *  len(z_list) *  len(ori_list) <= num_samples:
        print("exhaustive search")
        x,y,z,theta = np.meshgrid(x_list , y_list , z_list , ori_list ,  indexing='ij')
        
        x_data = x.flatten().reshape(-1,1)
        y_data = y.flatten().reshape(-1,1)
        z_data = z.flatten().reshape(-1,1)
        action_ori= theta.flatten().reshape(-1,1)

    
        
    else:
        print("random search")
        action_ori = np.random.choice(ori_list , num_samples).reshape(-1,1)
        x_data = np.random.choice(x_list , num_samples).reshape(-1,1)
        y_data = np.random.choice(y_list , num_samples).reshape(-1,1)
        z_data = np.random.choice(z_list , num_samples).reshape(-1,1)
        
    action_po = np.concatenate((x_data , y_data , z_data , action_ori) , axis = 1)
    
    #print(action_po.shape)
    action_po = torch.tensor(action_po , dtype = torch.float32 , device = device ) 
    
    #feat_vector = feat_vector.to(device).to(torch.float32).detach()
    
    r_network.eval()
    ba = torch.zeros(Graph.xyz.float().shape[0] , device = device , dtype=torch.int64)

    reward_tensor , _  = r_network(Graph.xyz.float() , Graph.edge_index , ba, action_po)
    
    best_ind = torch.argsort(-reward_tensor[:,0])
    best_ind = best_ind[:k]
    
    # topk_indices = torch.topk(reward_tensor, k=10).indices
    # print( topk_indices)
    
    best_points = action_po[best_ind]
    
    
    return best_points.cpu()
    






node_feature_size = 3  # Dimension of node features
embedding_size = 128  

descriptor_size = 128  # Dimension of the descriptor output
action_size = 5       # Dimension of the action input
hidden_size = 128      # Dimension of hidden layers in the Rmlp subnet

class holyGrail(nn.Module):
    def __init__(self):
        super(holyGrail, self).__init__()
        self.initial_conv = GCNConv(node_feature_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        
        # self.do = nn.Dropout(0.5)

        self.ln1 = nn.LayerNorm(descriptor_size)  # Layer normalization
        self.ln2 = nn.LayerNorm(descriptor_size)
        # self.ln3 = nn.LayerNorm(descriptor_size)
        
        self.afc1 = nn.Linear(action_size , 8*action_size)
        self.afc2 = nn.Linear(8*action_size , 8 * action_size)
        self.afc3 = nn.Linear(8*action_size , descriptor_size)
        
        self.fc1 = nn.Linear(descriptor_size, 64)
        self.fc2 = nn.Linear(64, 1)
        #self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        #self.fc3 = nn.Linear(hidden_size, 1)  # Output layer
        
        # Xavier initialization for better stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        #nn.init.xavier_uniform_(self.fc1.bias)
        nn.init.xavier_uniform_(self.afc1.weight)
        #nn.init.xavier_uniform_(self.afc1.bias)
        nn.init.xavier_uniform_(self.afc2.weight)
        #nn.init.xavier_uniform_(self.afc2.bias)
        nn.init.xavier_uniform_(self.afc3.weight)
        #nn.init.xavier_uniform_(self.afc3.bias)


        # Layer normalization
        #self.ln_2 = nn.LayerNorm(action_size)
        # self.ln_2 = nn.LayerNorm(4*action_size)
        self.ln_3 = nn.LayerNorm(4*action_size)
        self.ln_4 = nn.LayerNorm(64)
        self.ln_1 = nn.LayerNorm(8 * action_size)
        
        
    def forward(self, x, edge_index, batch_index , a):
        
        hidden0 = self.initial_conv(x, edge_index)
    
        hidden1 = self.conv1(hidden0, edge_index)
        hidden1 = F.leaky_relu(hidden1)  # Apply layer normalization
        hidden1 = hidden1 + hidden0  # Skip connection
        
        hidden2 = self.conv2(hidden1, edge_index)
        hidden2 = F.leaky_relu(hidden2)
        hidden2 = hidden2 + hidden1
        
        hidden3 = self.conv3(hidden2, edge_index)
        hidden3 = F.leaky_relu(hidden3)
        hidden3 = hidden3 + hidden2
        
             
        x = global_max_pool(hidden3 , batch_index)
        #x = self.ln_1(x)
        
        V = x.flatten().unsqueeze(0)
        
        # Combine descriptor and action
        V = V.repeat(a.size(0), 1)
        V = self.ln2(V)
        
        sin_theta = (torch.sin(a[:,3])/20).reshape(-1,1)
        cos_theta = (torch.cos(a[:, 3 ])/20).reshape(-1,1)
        a = torch.cat([a[:,0].reshape(-1,1) , a[:,1].reshape(-1,1) , a[:,2].reshape(-1,1) , sin_theta , cos_theta] , dim=1).to(device).reshape(-1 , 5)

        #a=self.ln_2(a)
        a = self.afc1(a)
        a = F.leaky_relu(a)
        a = self.ln_1(a)
        
        a = self.afc2(a)
        a = F.leaky_relu(a)
        #a = self.ln_2(a)
        
        a = self.afc3(a)
        a = F.leaky_relu(a)
        a = self.ln1(a)
        
        x = V * a
        
        # x = self.do(x)
        
        # x = self.ln3(x)
        
        
        # Pass through layers with activation functions and layer normalization
        
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.ln_4(x)
        
        x = self.fc2(x)
        #x = F.sigmoid(x)

        
        # Output layer
        reward = x
            
            
            
        return reward , V

        


def rot_around_z (data , theta):
    Q = np.matrix( [])

def augmentationOnTheFly(data , action , reward , num_augment=18 , device=device):
    from torch_geometric.data import Data
    
    if data.shape[0]>200:
        ind = np.random.choice(data.shape[0] , 200)
        data=data[ind]
    
    reward = reward[0].cpu().detach().item()
    action = action.reshape(-1,1).cpu().numpy()[:,0]
    rot_list = np.arange((np.pi/num_augment) , np.pi , (np.pi/num_augment) )
    new_GAR=[]
    for q in range(num_augment-1):
        X = (data[:,0] * np.cos(rot_list[q]) - data[:,1] * np.sin(rot_list[q])).reshape(-1,1)
        Y = (data[:,0] * np.sin(rot_list[q]) + data[:,1] * np.cos(rot_list[q])).reshape(-1,1)
        Z = data[:,2].reshape(-1,1)
        
        G_x = np.round((action[0] * np.cos(rot_list[q]) - action[1] * np.sin(rot_list[q])),2)
        G_y = np.round((action[0] * np.sin(rot_list[q]) + action[1] * np.cos(rot_list[q])),2)
        G_z = (action[2])
        G_t = (action[3] + rot_list[q] - int((action[3] + rot_list[q])/(2*np.pi) )*(2*np.pi))
        
        if G_t > np.pi:
            G_t = G_t - np.pi
        new_data = np.concatenate([X,Y,Z] , axis=1)
        new_g = G_maker(new_data)
        new_graph = from_networkx(new_g).to(device)
        
        new_graph = Data(xyz=new_graph.xyz.float(), edge_index=new_graph.edge_index ,  num_nodes=new_graph.xyz.float().shape[0] )
        #new_action = np.concatenate([G_x , G_y , G_z , G_t] , axis=1).reshape(1,-1)
        new_action = torch.tensor([G_x , G_y , G_z , G_t] , dtype=torch.float).reshape(1,-1).to(device)
        new_reward = torch.tensor([reward] , dtype=torch.float).to(device)
        
        new_gar = (new_graph , new_action , new_reward)
        new_GAR.append(new_gar)
        
    return new_GAR



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

# Function to create a gripper visualization with lines aligned with the Z-axis

# inputs : Object , Actions

def visualize_gripper(object_data , actions):

    w=0.1
    d=0.05

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    gripper_points = np.array([
        [-w/2 , 0 , 0 ],
        [-w/2 , 0 , d ],
        [w/2 , 0 , 0 ],
        [w/2 , 0 , d ],
        [0 , 0 , d],
        [0 , 0 , 2*d ]
    ])

    for i in range(actions.shape[0]):
      x,y,z,rotation_angle = actions[i]

      # Rotation matrix for the gripper orientation
      rotation_matrix = np.array([
          [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
          [np.sin(rotation_angle), np.cos(rotation_angle), 0],
          [0, 0, 1]
      ])

      # Transform gripper points to world coordinates
      gripper_points_world = np.dot(gripper_points, rotation_matrix.T) + [x, y, z]

      # Gripper finger connections
      finger1 = [gripper_points_world[0], gripper_points_world[1]]
      finger2 = [gripper_points_world[2], gripper_points_world[3]]
      hor_connection = [gripper_points_world[1] , gripper_points_world[3] ]
      ver_connection = [gripper_points_world[4], gripper_points_world[5]]

      # Create a 3D plot
      

      # Plot the gripper fingers and top connection
      ax.add_collection3d(Line3DCollection([finger1], colors='b'))
      ax.add_collection3d(Line3DCollection([finger2], colors='b'))
      ax.add_collection3d(Line3DCollection([hor_connection], colors='b'))
      ax.add_collection3d(Line3DCollection([ver_connection], colors='b'))
      #ax.scatter(x, y, z, c='r', marker='o', s=50)
      # Set axis limits

    #x , y , z = object_data.T
    #ax.scatter(x,y,z,c='g') 
    
    ax.set_xlim([-0.1, 0.1])
    ax.set_ylim([-0.1, 0.1])
    ax.set_zlim([0,0.2])

      # Set axis labels
      #ax.set_xlabel('X')
      #ax.set_ylabel('Y')
      #ax.set_zlabel('Z')

      # Show the plot
    plt.show()
