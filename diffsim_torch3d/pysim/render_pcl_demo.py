import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import time
import json
import sys
import gc
import numpy as np
import os
from datetime import datetime

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

#demo_length = 30
demo_length = 50
#demo_length = 45
#demo_length = 50
step = 1
out_dir = 'demo_pcl_frames'
num_points = 5000

def plot_pointcloud(points, title=""):
    x, z, y = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, y, s=0.15)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(100, -60)
    #plt.savefig(title)
    #plt.clf()
    plt.show()

def sim_objs_to_pcls(sim_dir='default_out'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(0, demo_length, step):
        mesh_fnames = sorted([f for f in os.listdir('%s/out0'%sim_dir) if '%04d'%i in f])
        all_verts = []
        all_faces = []
        vert_count = 0
        for j, f in enumerate(mesh_fnames[:1] + mesh_fnames[2:]):
        #for j, f in enumerate(mesh_fnames):
        #for j, f in enumerate(mesh_fnames[:1]):
            verts, faces, aux = load_obj(os.path.join(sim_dir, "out0", f))
            faces_idx = faces.verts_idx + vert_count
            verts = verts
            vert_count += len(verts)
            all_verts.append(verts)
            all_faces.append(faces_idx)
        mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
        sample_pcl = sample_points_from_meshes(mesh, num_points)
        np.save(os.path.join(out_dir, '%03d.npy'%i), sample_pcl)
        #plot_pointcloud(sample_pcl)
        #break
if __name__ == '__main__':
    sim_objs_to_pcls()
