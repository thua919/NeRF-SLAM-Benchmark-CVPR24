import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = '0'
import time
import copy
import torch.nn.functional as F
from tqdm import tqdm
import random

#from thop import profile
#from thop import clever_format

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.dataset import get_dataset
from src.utils.pose_utils import matrix_to_axis_angle,at_to_transform_matrix, matrix_to_quaternion,qt_to_transform_matrix
from src.NeRFs.keyframe_global import KeyFrameDatabase
from src.NeRFs.representaion import NeRFs
from src.NeRFs.representaion_hybrid import NeRFs_hybrid
from src.utils.mesher_utils import extract_mesh
from src.utils.vis_utils import plot_depth_color
from src.utils.mesh_utils_eslam import Mesher
from src.tools.pose_eval import pose_evaluation


class standard_SLAM():
    '''
    benchmark NeRFs for SLAM main class.
    '''
    def __init__(self, config, writer=None):
        self.config = config
        device = self.config['device']
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.writer = writer
        # set bound
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(self.device)
        
        # set pose
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.pose_gt = self.load_gt_pose()
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
        
        elif self.config['training']['rot_rep'] == 'quat':
            print('Using quaternion as rotation representation')
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix

        # set keyframe
        num_kf = int(self.dataset.num_frames // self.config['NeRFs']['tricks']['keyframe_every'] + 1)
        self.keyframeDatabase = KeyFrameDatabase(config, self.dataset.H, self.dataset.W, num_kf, self.dataset.num_rays_to_save, self.device)
        
        # set NeRFs
        
        if self.config['hybrid']:
            self.NeRF = NeRFs_hybrid(config, self.bounding_box,self.device)
        else:
            self.NeRF = NeRFs(config, self.bounding_box,self.device)
            print('encoder: ',self.NeRF.encoder,
                'geometric_render: ',self.NeRF.geometric_render,
                'coupling?: ',self.NeRF.share_encoder)
        
        self.map_optimizer = self.NeRF.optimizer
        
        # set useful parameters
        self.mapping_pix_samples = self.config['NeRFs']['tricks']['mapping_pix_samples']
        self.tracking_pix_samples = self.config['NeRFs']['tricks']['tracking_pix_samples']
        self.first_iters = self.config['NeRFs']['tricks']['first_iters']
        self.keyframe_every = self.config['NeRFs']['tricks']['keyframe_every']

        # set utils
        self.mesh_res = self.config['utils']['mesh_resolution']
        self.fig_freq = self.config['utils']['fig_freq']
        self.Mesher = Mesher(config, self.bounding_box, self.marching_cube_bound, self.dataset, self.NeRF)
        self.keyframe_dict_mesh = []

    def load_gt_pose(self):
        '''
        Load the ground truth pose
        For constantly ploting pose error
        '''
        pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            pose_gt[i] = pose
        return pose_gt
    
    
    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice
    
    def save_mesh(self, i, voxel_size=0.01): # should be <0.05
        voxel_size = self.mesh_res
        mesh_savepath = os.path.join(self.config['data']['output'], 'mesh_track{}.ply'.format(i))
        extract_mesh(self.NeRF, 
                    self.config, 
                    self.bounding_box,
                    marching_cube_bound = self.marching_cube_bound,  
                    voxel_size=voxel_size, 
                    mesh_savepath=mesh_savepath,
                    device=self.device)
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.NeRF.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')
    
    def first_frame_mapping(self, batch, n_iters=100, g_iter=0):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        item ='init'
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w
            
        # Training
        x=g_iter
        start_time = time.time()
        for i in range(n_iters):
            g_iter = x + i
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.mapping_pix_samples)
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_c = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.mapping_pix_samples, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            loss, depth_l1, psnr = self.NeRF.render_rays(rays_o, rays_d, target_d = target_d, target_c = target_c)
            loss.backward()
            self.map_optimizer.step()
            
            self.writer.add_scalar(item+'/depth_l1', depth_l1, g_iter)
            self.writer.add_scalar(item+'/psnr', psnr, g_iter)
            
            print('Iter: {}, Loss: {}'.format(i, loss.item()))
            if i%self.fig_freq == 0 or i == n_iters-1:
                output_path = self.config['data']['output']
                os.makedirs(output_path, exist_ok=True)
                plot_depth_color(self.NeRF,c2w,self.dataset.H,self.dataset.W,
                                 self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy,
                                 batch['depth'].squeeze(0).to(self.device),
                                 batch['rgb'].squeeze(0).to(self.device),
                                 output_path,g_iter,self.device)
                
        end_time = time.time()
        init_time = end_time - start_time
        print("Initilization took: ", init_time, "seconds to run")
        self.writer.add_scalar(item+'/time', init_time, g_iter)
            
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch)
        
        if self.config['utils']['first_mesh']:
            self.save_mesh(0)

        print('First frame mapping done')
    
    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]
    
    #def freeze_model(self):
    #    self.decoder = copy.deepcopy(self.NeRF.decoder)        
    #    for param in self.decoder.parameters():
    #        param.requires_grad = False          
    #    if self.NeRF.encoder == 'dense' or self.NeRF.encoder == 'hash':
    #        self.geo_embed_fn = copy.deepcopy(self.NeRF.geo_embed_fn)
    #        self.app_embed_fn = copy.deepcopy(self.NeRF.app_embed_fn)
    #        for param in self.geo_embed_fn.parameters():
    #            param.require_grad = False        
    #        if not self.NeRF.share_encoder:
    #            for param in self.app_embed_fn.parameters():
    #                param.require_grad = False      
    #    elif self.NeRF.encoder == 'tri_plane' or self.NeRF.encoder == 'tensorf':
    #        self.geo_embed = copy.deepcopy(self.NeRF.geo_embed)
    #        self.app_embed = copy.deepcopy(self.NeRF.app_embed)
    #        for elem in list(self.geo_embed):
    #            for elem_sub in elem:
    #                param.require_grad = False
    #        if not self.NeRF.share_encoder:
    #            for param in self.app_embed.parameters():
    #                param.require_grad = False
                

    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['NeRFs']['tricks']['track_lr']},
                                            {"params": cur_trans, "lr": self.config['NeRFs']['tricks']['track_lr']}])
        
        return cur_rot, cur_trans, pose_optimizer  
    
    def tracking_render(self, batch, frame_id, pre_g_iter=0):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''
        item = 'tracking'
        c2w_gt = batch['c2w'][0].to(self.device)
        g_iter = pre_g_iter

        # Initialize current pose
        #if self.config['tracking']['iter_point'] > 0:
        #    cur_c2w = self.est_c2w_data[frame_id]     
        #else: # always 0:
        cur_c2w = self.predict_current_pose(frame_id)
            
        #self.freeze_model()

        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        # Start tracking
        x = g_iter
        for i in range(self.config['NeRFs']['tricks']['tracking_iters']):
            g_iter = x + i
            self.writer.add_scalar('training/frame_idx', frame_id, g_iter) 

            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)
            #if self.config['pose_rel']:
            #    c2w_est_rel = self.matrix_from_tensor(cur_rot, cur_trans)
            #    c2w_est = c2w_est_rel @ last_c2w
            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, 
                                             self.tracking_pix_samples)
                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_c = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w_est[...,:3, -1].repeat(self.tracking_pix_samples, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            loss, depth_l1, psnr = self.NeRF.render_rays(rays_o, rays_d, target_d = target_d, target_c = target_c, tracking=True)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1
            
            if thresh > 100:
                break

            loss.backward()
            pose_optimizer.step()
        
        #if self.config['tracking']['best']:
            # Use the pose with smallest loss
        self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        #else:
            # Use the pose after the last iteration
        #    self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

       # Save relative pose of non-keyframes
        if frame_id % self.keyframe_every != 0:
            kf_id = frame_id // self.keyframe_every
            kf_frame_id = kf_id * self.keyframe_every
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
            
            # HUA: related re place
            #self.est_c2w_data[frame_id] = delta @ c2w_key
        
        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
        return g_iter

    def global_BA(self, batch, cur_frame_id, pre_g_iter=0):
        
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        item = 'mapping'
        pose_optimizer = None
        g_iter = pre_g_iter

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.keyframe_every)])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.keyframe_every)))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            #if self.config['mapping']['optim_cur']:
            cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
            pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            #else:
            #    cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
            #    pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
            #    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        x = g_iter
        for i in range(self.config['NeRFs']['tricks']['mapping_iters']):
            g_iter = x + i
            self.writer.add_scalar('training/frame_idx', cur_frame_id, g_iter)
             
            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.mapping_pix_samples)

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),
                                    max(
                                        self.mapping_pix_samples // len(self.keyframeDatabase.frame_ids), 
                                        100 #self.config['mapping']['min_pixels_cur']
                                        )
                                    )
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.keyframe_every, -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_c = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            loss, depth_l1, psnr = self.NeRF.render_rays(rays_o, rays_d, target_d = target_d, target_c = target_c)
            loss.backward(retain_graph=True)
            
            self.writer.add_scalar(item+'/depth_l1', depth_l1, g_iter)
            self.writer.add_scalar(item+'/psnr', psnr, g_iter)
            
            self.map_optimizer.step()
            self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % 5 == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                
                # So current pose is always unchanged
                #if self.config['mapping']['optim_cur']:
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                #else:
                #    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses
                #    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            #if self.config['mapping']['optim_cur']:
            print('Update current pose')
            self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
        return g_iter
    
    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.keyframe_every == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.keyframe_every
                kf_frame_id = kf_id * self.keyframe_every
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses
    
    def run(self):
        data_loader = DataLoader(self.dataset, num_workers=4)
        global g_iter
        g_iter = 0
        if self.config['data']['dataset']=='replica':
            max_iter = data_loader.dataset.frame_ids.stop
        else:
            max_iter=data_loader.dataset.frame_ids[-1]
        
        for i, batch in tqdm(enumerate(data_loader)):
            self.progress = self.NeRF.progress = i / max_iter
            
            if i == 0:
                # 1. Initilization
                self.first_frame_mapping(batch, self.first_iters, g_iter=g_iter)
                g_iter += self.first_iters
            else:
                # 2. Tracking
                track_time_start = time.time()
                g_iter = self.tracking_render(batch, i, pre_g_iter=g_iter)
                track_time_end = time.time()
                track_time = track_time_end - track_time_start
                self.writer.add_scalar('tracking/time', track_time, i) 
                
                # 3. Map when keyframe
                if i%self.keyframe_every==0:
                    map_time_start = time.time()
                    g_iter = self.global_BA(batch, i, pre_g_iter=g_iter)
                    map_time_end = time.time()
                    map_time = map_time_end - map_time_start
                    self.writer.add_scalar('mapping/time', map_time, i)
                # 4. Add keyframe
                if i % self.keyframe_every == 0:
                    self.keyframeDatabase.add_keyframe(batch)

                # meshing
                if i%self.config['utils']['mesh_freq']==0:
                    self.save_mesh(i)

                ## evaluation metric ##
                pose_relative = self.convert_relative_pose()
                results_abs=pose_evaluation(self.pose_gt, self.est_c2w_data, 1, 
                                            os.path.join(self.config['data']['output']), i)
                results_rel=pose_evaluation(self.pose_gt, pose_relative, 1, 
                                            os.path.join(self.config['data']['output']), 
                                            i, img='pose_r', name='output_relative.txt')
                
                self.writer.add_scalar('ATE/rmse', results_abs['absolute_translational_error.rmse'], i) 
                self.writer.add_scalar('ATE/rmse', results_abs['absolute_translational_error.mean'], i) 
    
        model_savepath = os.path.join(self.config['data']['output'], 'checkpoint{}.pt'.format(i)) 
        self.save_ckpt(model_savepath)
        self.save_mesh(i)
        pose_relative = self.convert_relative_pose()
        results_abs=pose_evaluation(self.pose_gt, self.est_c2w_data, 1, 
                                            os.path.join(self.config['data']['output']), i,
                                            final_flag = True)
        results_rel=pose_evaluation(self.pose_gt, pose_relative, 1, 
                                            os.path.join(self.config['data']['output']), 
                                            i, final_flag = True, 
                                            img='pose_r', name='output_relative.txt')
        