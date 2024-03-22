# Grasp the Graph (GtG 1.0 & GtG 1.1): A Super Light Graph-RL Framework for Robotic Grasping

Welcome to the repository for the "Grasp the Graph" (GtG) framework! This project implements the concepts outlined in the GtG paper. We utilize CoppeliaSim as our simulation environment, along with key dependencies such as PyTorch, PyTorch Geometric, and Open3D.

The model is trained in a scenario featuring a single object within the scene. We employ 3 RGB-D cameras strategically positioned with a 120-degree separation to capture a comprehensive point cloud representation of the object.

Our test results exhibit remarkable generalization and robustness to previously unseen objects. Even with just a single camera, the performance of our model remains commendably stable. Notably, while the model is trained solely on scenes featuring single objects, it demonstrates the ability to operate in scenarios with multiple objects, provided they do not occlude each other. (Note: this capability is not covered in the paper.)

For your convenience, we provide the checkpoint file named "GtG_best.pth". Further details can be found in the paper, available on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10412387) or [ResearchGate](https://www.researchgate.net/profile/Ali-Rashidi-Moghadam/publication/377819311_Grasp_the_Graph_GtG_A_Super_Light_Graph-RL_Framework_for_Robotic_Grasping/links/65ca348f1e1ec12eff8a5659/Grasp-the-Graph-GtG-A-Super-Light-Graph-RL-Framework-for-Robotic-Grasping.pdf), and on our [Google Site](https://sites.google.com/view/grasp-the-graph-gtg/home).

# Results

## Training Result

![TrainingPlot](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/Slide11.PNG)



## Test Result - Single Object

3 Cameras:
![3Cameras](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/13.gif)


2 Cameras:
![2Cameras](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/14.gif)


1 Camera:
![1Camera](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/15.gif)


Overall Performance on test Objects:
![Overall Performance on test Objects](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/16.gif)


## Test Result - Multi Object


![Grasping in Multi Object Scene](https://github.com/Ali-Rashidi/GtG_1.0_1.1/blob/main/17h.gif)


# References
- Paper (IEEE Xplore): [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10412387)
- Paper (ResearchGate): [ResearchGate](https://www.researchgate.net/profile/Ali-Rashidi-Moghadam/publication/377819311_Grasp_the_Graph_GtG_A_Super_Light_Graph-RL_Framework_for_Robotic_Grasping/links/65ca348f1e1ec12eff8a5659/Grasp-the-Graph-GtG-A-Super-Light-Graph-RL-Framework-for-Robotic-Grasping.pdf)
- Google Site: [Google Site](https://sites.google.com/view/grasp-the-graph-gtg/home)

# Contact
For any inquiries or feedback, feel free to reach out to me at [AliRashidiMoghadam@gmail.com].
