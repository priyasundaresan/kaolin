import os
import sys

import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import gc
import numpy as np
from datetime import datetime

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.io import load_obj

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


with open('conf/rigidcloth/sysid/start_cube.json','r') as f:
	config = json.load(f)

# PYTORCH3D
device = torch.device("cuda:0")
ref_verts, ref_faces, _ = load_obj("meshes/rigidcloth/sysid/ref_m0.02_stiff0.15.obj")
ref_faces_idx = ref_faces.verts_idx.to(device)
ref_verts = ref_verts.to(device)
ref_mesh = Meshes(verts=[ref_verts], faces=[ref_faces_idx])

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1


def reset_sim(sim, epoch):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def get_loss(sim):
    reg  = torch.norm(param_g, p=2)*0.001 
    #loss = 0
    #node_number = ref_verts.shape[0]
    #for i in range(node_number):
    #    loss += torch.norm(ref_verts[i]-(sim.cloths[0].mesh.nodes[i].x.to(device)))**2
    #loss /= node_number

    verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)

    curr_mesh = Meshes(verts=[verts], faces=[faces])

    sample_trg = sample_points_from_meshes(ref_mesh, 1000)
    sample_src = sample_points_from_meshes(curr_mesh, 1000)

    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    loss = loss_chamfer
    return loss.cpu() + reg
    #return loss.cpu()


def run_sim(steps, sim):
    stiff_inp, mass_inp = torch.relu(param_g).cpu()
    print(stiff_inp, mass_inp)
    if stiff_inp == 0 or mass_inp == 0:
        return float('inf')

    for node in sim.obstacles[1].curr_state_mesh.nodes:
    	node.m *= mass_inp

    orig_stretch = sim.cloths[0].materials[0].stretching 
    sim.cloths[0].materials[0].stretching = orig_stretch*stiff_inp

    for step in range(20):
        arcsim.sim_step()
    
    loss = get_loss(sim)
    
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    while True:
        steps = 20
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(steps, sim)
        en0 = time.time()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))

        if loss < 0.005 or loss == float('inf'):
            print("final params (stiffness, mass): ", param_g, ", (ref: 0.15, 0.02)")
            break
        
        optimizer.step()
        epoch = epoch + 1
    return param_g, loss, epoch+1
# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 2
    sim=arcsim.get_sim()
    
    #param_g = torch.tensor([0.5,0.5],dtype=torch.float64, requires_grad=True)
    
    out_dir = 'exps_stiff_mass'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    results = []
    cur_step = 0
    #for i in np.linspace(0.1,1.1,5):
    #    for j in np.linspace(0.05,0.4,5):
    for i in np.linspace(0.15,1.1,5):
        for j in np.linspace(0.02,0.4,5):
            pprint.pprint(results)
            param_g = torch.tensor([i,j],dtype=torch.float64, requires_grad=True)
            lr = 0.015
            optimizer = torch.optim.Adam([param_g],lr=lr)
            try:
                result, loss, iters = do_train(cur_step,optimizer,sim)
                if loss != float('inf'):
                    results.append([i,j] + result.squeeze().tolist() + [loss.item()] + [iters])
                os.system('mv %s ./%s/run%d'%(out_path, out_dir,cur_step))
                os.system('mkdir %s'%out_path)
                save_config(config, out_path+'/conf.json')
                cur_step += 1
            except:
                print("errored", i, j)
                continue
            np.save('%s/results.npy'%out_dir, results)
    pprint.pprint(results)
    results = np.array(results)
    np.save('%s/results.npy'%out_dir, results)
                                 
print("done")
