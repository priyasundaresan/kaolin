import os
import math
import cv2
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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import kaolin as kal
from pytorch3d.renderer import (
    look_at_view_transform,
)

if len(sys.argv)==1:
	out_path = 'default_out'
else:
	out_path = sys.argv[1]
if not os.path.exists(out_path):
	os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


with open('conf/rigidcloth/sysid/start.json','r') as f:
	config = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# PYTORCH3D
texture_res = 2
batch_size = 1
image_size_x = image_size_y = 200
device = torch.device("cuda:0")
ref_mesh = kal.io.obj.import_mesh('meshes/rigidcloth/sysid/ref_pinned_less_stiff.obj', with_materials=True)
#ref_mesh = kal.io.obj.import_mesh('meshes/rigidcloth/sysid/ref_pinned_more_stiff.obj', with_materials=True)
ref_faces = ref_mesh.faces.cuda()
nb_faces = len(ref_faces)
uvs = ref_mesh.uvs.cuda().unsqueeze(0)
face_uvs_idx = ref_mesh.face_uvs_idx.cuda()
face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
face_uvs.requires_grad = False
texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                         requires_grad=False)
texture_map[:,1,:,:] = 0.5
print(texture_map.shape)

#theta = torch.rand(1, device='cuda') * 2 * math.pi
#phi = (torch.rand(1, device='cuda') - 0.5) * math.pi
#distance = torch.rand(1, device='cuda') * 0.1 + 0.5

theta = torch.Tensor([math.pi]).cuda()
phi = torch.Tensor([math.pi/2]).cuda()
distance = torch.Tensor([1.55]).cuda()

x = torch.cos(theta) * distance 
y = torch.sin(theta) * distance
z = torch.sin(phi) * distance - 1.25
cam_pos = torch.stack([x, y, z], dim=-1)
look_at = torch.zeros([1, 3], device='cuda')
cam_up = torch.tensor([[0., 0, 1]], device='cuda')

cam_transform = kal.render.camera.generate_transformation_matrix(cam_pos, look_at, cam_up).cuda()
cam_proj = kal.render.camera.generate_perspective_projection(45, image_size_y / image_size_x).cuda().unsqueeze(0)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']

scalev=1

def reset_sim(sim, epoch):
	arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def get_render_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device).unsqueeze(0)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device).long()
    return cloth_verts, cloth_faces

def get_image(verts, faces):
    face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(verts.repeat(batch_size, 1, 1), \
                faces, cam_proj, camera_transform=cam_transform
            )
    face_attributes = [
            face_uvs.repeat(batch_size, 1, 1, 1),
            torch.ones((batch_size, nb_faces, 3, 1), device='cuda')
        ]
    image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
        image_size_x, image_size_y, face_vertices_camera[:, :, :, -1],
        face_vertices_image, face_attributes, face_normals[:, :, -1])
    texture_coords, mask = image_features
    image = kal.render.mesh.texture_mapping(texture_coords,
                                            texture_map.repeat(batch_size, 1, 1, 1),
                                            mode='bilinear')
    image = torch.clamp(image * mask, 0., 1.)
    return image

def get_loss(sim, epoch):
    reg  = torch.norm(param_g, p=2)*0.001
    loss = 0

    curr_mesh_verts, curr_mesh_faces = get_render_mesh_from_sim(sim)
    curr_image = get_image(curr_mesh_verts, curr_mesh_faces)

    ref_mesh_verts = ref_mesh.vertices.cuda().unsqueeze(0)
    ref_image = get_image(ref_mesh_verts, ref_faces)

    loss_rgb = (torch.abs(curr_image - ref_image)).mean()
    #loss = loss_rgb*0.0 + loss_silhouette*1.0
    loss = loss_rgb
    visualization = np.hstack((curr_image[0].detach().cpu().numpy(), ref_image[0].detach().cpu().numpy()))
    cv2.imwrite('%s/epoch%03d.jpg'%(out_path, epoch), visualization*255)
    return loss.cpu() + reg

def run_sim(steps, sim, epoch):
    output_processed = param_g
    print("param_g:", output_processed)
    orig = sim.cloths[0].materials[0].stretching
    sim.cloths[0].materials[0].stretching = orig*output_processed.cpu()
    for step in range(steps):
        print(step)
        arcsim.sim_step()
    loss = get_loss(sim, epoch)
    return loss

def do_train(cur_step,optimizer,sim):
    epoch = 0
    while True:
        steps = 40
        
        reset_sim(sim, epoch)
        
        st = time.time()
        loss = run_sim(steps, sim, epoch)
        en0 = time.time()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        #if loss < 0.016:
        if loss < 0.01:
        #if loss < 0.02:
            break
        
        en1 = time.time()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
        
        
        optimizer.step()
        epoch = epoch + 1
# break

with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
    tot_step = 1
    sim=arcsim.get_sim()
    
    param_g = torch.tensor(0.5,dtype=torch.float64, requires_grad=True)
    
    lr = 0.1
    #lr = 0.15
    optimizer = torch.optim.Adam([param_g],lr=lr)
    for cur_step in range(tot_step):
    	do_train(cur_step,optimizer,sim)

print("done")
