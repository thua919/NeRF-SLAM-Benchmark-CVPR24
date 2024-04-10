import torch
import torch.nn as nn
from src.NeRFs.encoder_hybrid import get_encoder, get_pos_embed
from src.NeRFs.decoder_hybrid import get_decoder
from src.utils.common_utils import intersect_with_sphere, mse2psnr
from src.utils.render_utils import alpha_compositing_weights, composite, compute_loss

class NeRFs_hybrid(nn.Module):
    def __init__(self, config, bound_box, device):
        super(NeRFs_hybrid, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.device = device
        ## 1. set resolution ##
        # we test two scenario: 
            # large 【implicit_dim】, small 【res_level】 (e.g. ESLAM [32,2])
            # small 【implicit_dim】, large 【res_level】 (e.g. co-slam [2,16])
        space_len = self.bounding_box[:,1] - self.bounding_box[:,0] #(self.bounding_box[:,1] - self.bounding_box[:,0]).coarse()
        self.geo_resolution_fine = list(map(int, (space_len / self.config['NeRFs']['space_resolution']['geo_block_size_fine']).tolist()))
        self.geo_resolution_coarse = list(map(int, (space_len / self.config['NeRFs']['space_resolution']['geo_block_size_coarse']).tolist()))
        self.app_resolution_fine = list(map(int, (space_len / self.config['NeRFs']['space_resolution']['app_block_size_fine']).tolist()))
        self.app_resolution_coarse = list(map(int, (space_len / self.config['NeRFs']['space_resolution']['app_block_size_coarse']).tolist()))
        self.implicit_dim = self.config['NeRFs']['space_resolution']['implicit_dim']
        self.res_level = self.config['NeRFs']['space_resolution']['res_level']
        
        ## 2. set pose embed (gamma) ##
        # we test under fix scenario:
            # for MLP: frequency
            # for hybrid: identity
        self.gamma = self.config['NeRFs']['tricks']['gamma']
        self.pos_embed_fn, self.pos_chanl = get_pos_embed(self.gamma)
        self.pos_embed_fn.to(self.device)
        
        ## 3. set encoder ##
        # For fair comparison, we separately encode geometry and appearance
        self.encoder_1 = 'hash' if self.config['hybrid_hash'] else 'dense'
        
        #self.encoder_1 = 'hash' # 'dense'
        self.encoder_2 = 'tri_plane'    

        # get encoder functions for encoder_1
        self.geo_embed_fn, self.geo_chanl = get_encoder(self.encoder_1, res_level=self.res_level, implicit_dim=self.implicit_dim, 
                                    resolution_fine=self.geo_resolution_fine, resolution_coarse=self.geo_resolution_coarse)
        self.geo_embed_fn.to(self.device)
        self.app_embed_fn = None
        self.app_chanl = None
        
        # get encoded features for encoder_2
        self.plane_resolution_coarse = list(map(int, (space_len / self.config['NeRFs']['space_resolution']['plane_block_size_coarse']).tolist()))
        self.geo_embed, _ = get_encoder(self.encoder_2, res_level=1, implicit_dim=self.implicit_dim, 
                                    resolution_fine= self.plane_resolution_coarse, #self.geo_resolution_coarse, 
                                    resolution_coarse= self.plane_resolution_coarse) #self.geo_resolution_coarse)
        self.app_embed = None
        
        
        ## 4. set render ##
        # noted that some render might have the learnable parameter, so we set it in the training part
        self.geometric_render = 'sdf_approx' if self.config['NeRFs']['G']['sdf_approx'] else \
                                'sdf_style' if self.config['NeRFs']['G']['sdf_style'] else \
                                'sdf_neus'
        if self.geometric_render == 'sdf_style':
            self.beta = nn.Parameter(10 * torch.ones(1)).to(self.device)
        else:
            self.beta = None
        if self.geometric_render == 'sdf_neus':
            self.anneal_end = 0.1
            self.s_var = torch.nn.Parameter(torch.tensor(1.4, dtype=torch.float32))
        
        ## 5. set decoder ##
            # if share encoder, then the geo feature is channel to app feature by [geo_feat_dim]
        geo_feat_dim = 0
        
        input_ch_1 = self.geo_chanl + self.pos_chanl
        input_ch_2 = self.implicit_dim
        input_ch = input_ch_1+input_ch_2
        self.decoder = get_decoder(device = self.device ,bound=self.bounding_box, 
                                    config = self.config, input_ch = input_ch, pos_ch=self.pos_chanl,
                                    geo_feat_dim = geo_feat_dim, hidden_dim=32, num_layers=2, beta = self.beta)
        
        #input_ch_2 = self.implicit_dim*self.res_level + self.pos_chanl
        # we only use the self.decoder_2.get_raw_sdf
                     
        ## 6. set optimizer ##
        self.map_lr = self.config['NeRFs']['tricks']['map_lr']

        #trainable_parameters_1 = [{'params': self.decoder_1.parameters()},
        #                        {'params': self.geo_embed_fn.parameters()}]
        #self.optimizer = torch.optim.Adam(trainable_parameters, lr=self.map_lr) #betas=(0.9, 0.99))
            
        planes_xy, planes_xz, planes_yz = self.geo_embed
        planes_para = []
        #c_planes_para = []
        for planes in [planes_xy, planes_xz, planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane
    
        self.optimizer = torch.optim.Adam([{'params': self.decoder.parameters(), 'lr':self.map_lr},
                                        {'params': self.geo_embed_fn.parameters(), 'lr':self.map_lr},
                                        {'params': planes_para, 'lr':self.map_lr}])                
                
 
    #### class initiliztion finished ####
        
    def sdf2weights_approx(self, sdf, z_vals, truncation):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / truncation) * torch.sigmoid(-sdf / truncation)

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + truncation, torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
 
    def raw2outputs_approx(self, raw, z_vals, truncation):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        
        weights = self.sdf2weights_approx(raw[..., 3], z_vals, truncation)
        
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    
    def raw2outputs_style(self, raw, z_vals, beta=10):   
        sdfs = raw[..., 3]
        rgbs = torch.sigmoid(raw[...,:3])
        alpha = 1. - torch.exp(-beta * torch.sigmoid(-sdfs * beta))
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=self.device)
                                        , (1. - alpha + 1e-10)], -1), -1)[:, :-1]
        
        rgb_map = torch.sum(weights[..., None] * rgbs, -2)
        weights = weights.squeeze(0).squeeze(-1)
        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var

    @torch.no_grad()
    def get_dist_bounds(self, center, ray_unit):
        dist_near, dist_far = intersect_with_sphere(center, ray_unit, radius=1.)
        dist_near.relu_()  # Distance (and thus depth) should be non-negative.
        outside = dist_near.isnan()
        dist_near[outside], dist_far[outside] = 1, 1.2  # Dummy distances. Density will be set to 0.
        return dist_near, dist_far, outside
    
    def _get_iter_cos(self, true_cos, progress=1.):
        anneal_ratio = min(progress / self.anneal_end, 1.)
        # The anneal strategy below keeps the cos value alive at the beginning of training iterations.
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)  # always non-positive
    
    def compute_neus_alphas(self, ray_unit, sdfs, gradients, dists, dist_far=None, 
                            progress=1., eps=1e-5):
        sdfs = sdfs[..., 0]  # [B,R,N]
        # SDF volume rendering in NeuS.
        inv_s = self.s_var.exp()
        true_cos = (ray_unit[..., None, :] * gradients).sum(dim=-1, keepdim=False)  # [B,R,N]
        iter_cos = self._get_iter_cos(true_cos, progress = progress)  # [B,R,N]
        # Estimate signed distances at section points
        if dist_far is None:
            dist_far = torch.empty_like(dists[..., :1, :]).fill_(1e10)  # [B,R,1,1]
        dists = torch.cat([dists, dist_far], dim=2)  # [B,R,N+1,1]
        dist_intvs = dists[..., 1:, 0] - dists[..., :-1, 0]  # [B,R,N]
        est_prev_sdf = sdfs - iter_cos * dist_intvs * 0.5  # [B,R,N]
        est_next_sdf = sdfs + iter_cos * dist_intvs * 0.5  # [B,R,N]
        prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
        next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
        alphas = ((prev_cdf - next_cdf) / (prev_cdf + eps)).clip_(0.0, 1.0)  # [B,R,N]
        # weights = render.alpha_compositing_weights(alphas)  # [B,R,N,1]
        return alphas
    
    def raw2outputs_neus(self, raw, z_vals, rays_o, rays_d, gradient, render_color=False):
        
        sdfs = raw[..., 3]
        rgbs = torch.sigmoid(raw[...,:3])
        sdfs = sdfs.unsqueeze(-1)
        
        # SDF volume rendering.
        if not render_color:
            near, far, outside = self.get_dist_bounds(rays_o, rays_d)
            alphas = self.compute_neus_alphas(ray_unit=rays_o.unsqueeze(0), sdfs=sdfs.unsqueeze(0), 
                                gradients=gradient.unsqueeze(0), 
                                dists=z_vals.unsqueeze(-1).unsqueeze(0), 
                                dist_far=far[..., None].unsqueeze(0),
                                progress=self.progress)  # [B,R,N]
        else:
            alphas = self.compute_neus_alphas(ray_unit=rays_o.unsqueeze(0), sdfs=sdfs.unsqueeze(0), 
                                        gradients=gradient.unsqueeze(0), 
                                        dists=z_vals.unsqueeze(-1).unsqueeze(0), 
                                        dist_far=None,
                                        progress=self.progress)  # [B,R,N]
        
        weights = alpha_compositing_weights(alphas)
        rgb_map = composite(rgbs.unsqueeze(0), weights)
        
        weights = weights.squeeze(0).squeeze(-1)
        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)
        return rgb_map.squeeze(0), disp_map, acc_map, weights, depth_map, depth_var    
    

    def perturbation(self, z_vals):
        """
        Add perturbation to sampled depth values on the rays.
        Args:
            z_vals (tensor): sampled depth values on the rays.
        Returns:
            z_vals (tensor): perturbed depth values on the rays.
        """
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=z_vals.device)

        return lower + (upper - lower) * t_rand
    
    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """
        self.w_sdf_fs = self.config['NeRFs']['tricks']['sdf_fs_weight']
        self.w_sdf_center = self.config['NeRFs']['tricks']['sdf_center_weight']
        self.w_sdf_tail = self.config['NeRFs']['tricks']['sdf_tail_weight']
        
        
        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses 
    
     
    #### main (F+G) inference function ####    
    def render_rays(self, rays_o, rays_d, target_d = None, target_c = None, render_img=False, tracking=False):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]
        '''
        # 5.0 if tracking then freeze the parameter
        if tracking:
            for param_groups in self.optimizer.param_groups:
                param_groups['lr'] = 0.0
        else:
            for param_groups in self.optimizer.param_groups:
                param_groups['lr'] = self.map_lr
        
        # 5.1 sample points (ESLAM proposed)
            ### For rays with gt depth: (n_stratified) free + (n_importance) near surface
        n_stratified = self.config['NeRFs']['tricks']['ray_stratified_samples']
        n_importance = self.config['NeRFs']['tricks']['ray_surface_samples']
        self.truncation = self.config['NeRFs']['space_resolution']['truncation']
        n_rays = rays_o.shape[0]
        
        z_vals = torch.empty([n_rays, n_stratified + n_importance], device=self.device)
        near = 0.0
        t_vals_uni = torch.linspace(0., 1., steps=n_stratified, device=self.device)
        t_vals_surface = torch.linspace(0., 1., steps=n_importance, device=self.device)
        
            # mask: pixels with gt depth:
        gt_mask = (target_d > 0).squeeze()
        gt_nonezero = target_d[gt_mask]
        
            # Sampling points around the gt depth (surface(n_importance) + free(n_stratified))
        gt_depth_surface = gt_nonezero.expand(-1, n_importance)
        z_vals_surface = gt_depth_surface - (1.5 * self.truncation)  + (3 * self.truncation * t_vals_surface)

        gt_depth_free = gt_nonezero.expand(-1, n_stratified)
        z_vals_free = near + 1.2 * gt_depth_free * t_vals_uni

        z_vals_nonzero, _ = torch.sort(torch.cat([z_vals_free, z_vals_surface], dim=-1), dim=-1)

                # default perturbation
        z_vals_nonzero = self.perturbation(z_vals_nonzero)
        z_vals[gt_mask] = z_vals_nonzero
            ### For rays without gt depth: uniform sampling
        if not gt_mask.all():
            with torch.no_grad():
                t_vals_uni_no_gt = torch.linspace(0., 1., steps=n_stratified+n_importance, device=self.device)
                rays_o_uni = rays_o[~gt_mask].detach()
                rays_d_uni = rays_d[~gt_mask].detach()
                det_rays_o = rays_o_uni.unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = rays_d_uni.unsqueeze(-1)  # (N, 3, 1)
                t = (self.bounding_box.unsqueeze(0) - det_rays_o)/det_rays_d  # (N, 3, 2)
                far_bb, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                far_bb = far_bb.unsqueeze(-1)
                far_bb += 0.01

                z_vals_uni = near * (1. - t_vals_uni_no_gt) + far_bb * t_vals_uni_no_gt
                z_vals_uni = self.perturbation(z_vals_uni)
                
                ## we don't do importance sampling, as suggested by co-slam, is cubursome and not very useful
                #pts_uni = rays_o_uni.unsqueeze(1) + rays_d_uni.unsqueeze(1) * z_vals_uni.unsqueeze(-1)  # [n_rays, n_stratified, 3]
                #pts_uni_nor = normalize_3d_coordinate(pts_uni.clone(), self.bound)
                #sdf_uni = decoders.get_raw_sdf(pts_uni_nor, all_planes)
                #sdf_uni = sdf_uni.reshape(*pts_uni.shape[0:2])
                #alpha_uni = self.sdf2alpha(sdf_uni, decoders.beta)
                #weights_uni = alpha_uni * torch.cumprod(torch.cat([torch.ones((alpha_uni.shape[0], 1), device=device)
                #                                        , (1. - alpha_uni + 1e-10)], -1), -1)[:, :-1]
                #z_vals_uni_mid = .5 * (z_vals_uni[..., 1:] + z_vals_uni[..., :-1])
                #z_samples_uni = sample_pdf(z_vals_uni_mid, weights_uni[..., 1:-1], n_importance, det=False, device=device)
                #z_vals_uni, ind = torch.sort(torch.cat([z_vals_uni, z_samples_uni], -1), -1)
                
                z_vals[~gt_mask] = z_vals_uni.to(torch.float32)
        
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        # 5.2 query samples' color and sdf
        raw = self.decoder(pts, self.pos_embed_fn, self.geo_embed, self.geo_embed_fn, self.app_embed_fn)
     
        # 5.25 compute param
        if self.config['test_param']:
            from thop import profile,clever_format
            if self.encoder == 'freq':
                inputs = (pts, self.pos_embed_freq_fn)
            elif self.encoder == 'dense' or self.encoder == 'hash':
                inputs = (pts, self.pos_embed_fn, self.geo_embed_fn, self.app_embed_fn)
            elif self.encoder == 'tri_plane':
                inputs = (pts, self.pos_embed_fn, self.geo_embed, self.app_embed)
            elif self.encoder == 'tensorf':
                planes_xy, planes_xz, planes_yz, lines_z, lines_y, lines_x = self.geo_embed
                all_geo_planes = planes_xy, planes_xz, planes_yz
                all_geo_lines = lines_z, lines_y, lines_x
                if self.share_encoder:
                    inputs = (pts, self.pos_embed_fn, all_geo_planes, all_geo_lines)
                else:
                    planes_c_xy, planes_c_xz, planes_c_yz, lines_c_z, lines_c_y, lines_c_x = self.app_embed
                    all_c_planes = planes_c_xy, planes_c_xz, planes_c_yz 
                    all_c_lines = lines_c_z, lines_c_y, lines_c_x
                    inputs = (pts, self.pos_embed_fn, all_geo_planes, all_c_planes, all_geo_lines, all_c_lines)
            
            model_params_bytes = 0
            for param_group in self.optimizer.param_groups:
                params = param_group['params']
                param_group_bytes = sum(p.numel() * p.element_size() for p in params)
                model_params_bytes += param_group_bytes

            model_params_MB = model_params_bytes / (1024 * 1024)
            print('=======================================================')
            print(f"Total model parameters in MB: {model_params_MB:.2f} MB")
            print('=======================================================')

            #macs, params = profile(self.decoder, inputs = inputs)
            #print('decoder Params(Mb):', params/1000000)
            
            #macs, params = clever_format([macs, params], "%.3f") 
            #print('FLOPs(total):', macs)
            #print('for pts number:', pts.shape[0]*pts.shape[1]) 
            #params = float(params.rstrip('K'))*0.001
            #macs = float(macs.rstrip('G'))*1000/(pts.shape[0]*pts.shape[1])*1000
            #print('Params(Mb): ', params)
            #print('FLOPs(x103): ', macs)
            assert False
        
        # 5.3 start geometric redering:
        raw = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
        if self.geometric_render == 'sdf_approx':
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs_approx(raw, z_vals, self.truncation)
        elif self.geometric_render == 'sdf_style':
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs_style(raw, z_vals, self.beta)
        #elif self.geometric_render == 'sdf_neus':
            # firstly: need to compute gradient
        #    requires_grad = pts.requires_grad
        #    with torch.enable_grad():
        #        pts.requires_grad_(True)
        #        if self.encoder == 'freq':
        #            raw_tmp = self.decoder(pts, self.pos_embed_freq_fn)
        #        elif self.encoder == 'dense' or self.encoder == 'hash':
        #            raw_tmp = self.decoder(pts, self.pos_embed_fn, self.geo_embed_fn, self.app_embed_fn)
        #        elif self.encoder == 'tri_plane':
        #            raw_tmp = self.decoder(pts, self.pos_embed_fn, self.geo_embed, self.app_embed)
        #        elif self.encoder == 'tensorf':
        #            raw_tmp = self.decoder(pts, self.pos_embed_fn, all_geo_planes, all_c_planes , all_geo_lines, all_c_lines)
        #        gradient = torch.autograd.grad(raw_tmp.sum(), pts, create_graph=True)[0]
        #    pts.requires_grad_(requires_grad)
        #    gradient = gradient.detach()
        #    rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs_neus(raw, z_vals, rays_o, rays_d, gradient, render_color=True)
        
        if render_img:
            return rgb_map, depth_map
                   
        ## 5.4 compute loss
            # 1. sdf loss
        sdf = raw[..., -1]
        sdf_loss = self.sdf_losses(sdf[gt_mask], z_vals[gt_mask], target_d[gt_mask].squeeze(-1))
            # 2. color loss
        color_loss = compute_loss(rgb_map.squeeze(0), target_c)
        psnr = mse2psnr(color_loss)
        if self.config['verbose']:
            print('psnr: ', psnr.item())
        self.color_weight = self.config['NeRFs']['tricks']['color_weight']
            # 3. depth loss
        depth_loss = compute_loss(depth_map[gt_mask], target_d[gt_mask].squeeze(-1))
        depth_l1 = compute_loss(depth_map[gt_mask], target_d[gt_mask].squeeze(-1),loss_type='l1')
        if self.config['verbose']:
            print('depth_l1: ', depth_l1.item())
        self.depth_weight = self.config['NeRFs']['tricks']['depth_weight']
            # 4. total loss
        loss = self.color_weight * color_loss + self.depth_weight * depth_loss + sdf_loss    
         
        return loss, depth_l1, psnr
    
    def query_sdf(self, pts):
        raw = self.decoder(pts, self.pos_embed_fn, self.geo_embed, self.geo_embed_fn, self.app_embed_fn)
        return raw[..., -1]
    
    def query_color(self, pts):
        raw = self.decoder(pts, self.pos_embed_fn, self.geo_embed, self.geo_embed_fn, self.app_embed_fn)
        return torch.sigmoid(raw[..., :3])

            
                
            
            
        

        
        
        

