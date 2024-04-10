import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.common_utils import normalize_3d_coordinate


class Grid_Decoder(nn.Module):
    def __init__(self, device, bound, config, input_ch=3, pos_ch=3, geo_feat_dim=15, hidden_dim=32, num_layers=2,beta=None):
        super(Grid_Decoder, self).__init__()
        self.config = config
        self.bounding_box = bound
        self.input_ch = input_ch
        self.pos_ch = pos_ch
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if beta is not None:
            self.beta = beta
        self.coupling = config['NeRFs']['tricks']['share_encoder']
        self.geo_feat_dim = geo_feat_dim
        if (self.geo_feat_dim==0) and (self.coupling):
            self.coupling_base = True
            self.coupling = False
        else:
            self.coupling_base = False
        
        sdf_net = self.get_sdf_decoder()
        self.sdf_net = sdf_net.to(device)
        color_net = self.get_color_decoder()
        self.color_net = color_net.to(device)
        
    def get_sdf_decoder(self):
        sdf_net = []
        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = self.hidden_dim 
            if l == self.num_layers - 1:
                if self.coupling:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = 1
            else:
                out_dim = self.hidden_dim 
            
            sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers - 1:
                sdf_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(sdf_net))
    
    def get_color_decoder(self):
        color_net =  []
        for l in range(self.num_layers):
            if l == 0:
                if self.coupling:
                    # only geo feature passed to color decoder
                    in_dim = self.geo_feat_dim+self.pos_ch
                else:
                    # its own color embeding
                    in_dim = self.input_ch #+self.geo_feat_dim 
            else:
                in_dim = self.hidden_dim
            
            if l == self.num_layers - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))
        
    def forward(self, query_points, pos_embed_fn, geo_embed_fn, app_embed_fn = None):
        
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        # get feature embedding
        embed = geo_embed_fn(inputs_flat)
        embed_pos = pos_embed_fn(inputs_flat)

        if self.coupling:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1))
            sdf, geo_feat = h[...,:1], h[...,1:]
            rgb = self.color_net(torch.cat([geo_feat,embed_pos], dim=-1))
        elif self.coupling_base:
            sdf = self.sdf_net(torch.cat([embed, embed_pos], dim=-1))
            rgb = self.color_net(torch.cat([embed_pos, embed], dim=-1))
        elif (not self.coupling) and (not self.coupling_base):
            embed_color = app_embed_fn(inputs_flat)
            sdf = self.sdf_net(torch.cat([embed, embed_pos], dim=-1))
            rgb = self.color_net(torch.cat([embed_pos, embed_color], dim=-1))
        
        return torch.cat([rgb, sdf], -1)

class Decomp_Decoders(nn.Module):
    """
    Decoders for SDF and RGB.
    Args:
        c_dim: feature dimensions
        hidden_size: hidden size of MLP
        truncation: truncation of SDF
        n_blocks: number of MLP blocks
    """
    def __init__(self, device, bound, config, pos_ch, c_dim=32, geo_feat_dim=15, hidden_size=32, n_blocks=2,beta=None):
        super().__init__()
        self.bound = bound
        self.device = device
        self.pos_ch = pos_ch
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        if beta is not None:
            self.beta = beta
        self.coupling = config['NeRFs']['tricks']['share_encoder']
        self.F = 'tensorf' if config['NeRFs']['F']['choice_TensoRF'] else 'tri_plane'
        self.geo_feat_dim = geo_feat_dim
        if (self.geo_feat_dim==0) and (self.coupling):
            self.coupling_base = True
            self.coupling = False
        else:
            self.coupling_base = False
        
        if self.coupling:
            #SDF decoder
            linears = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
            output_linear = nn.Linear(hidden_size, 1+self.geo_feat_dim)
            #RGB decoder
            c_linears = nn.ModuleList(
            [nn.Linear(self.geo_feat_dim+self.pos_ch, hidden_size)] +
            [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])
            c_output_linear = nn.Linear(hidden_size, 3)
        else:
            #SDF decoder
            linears = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size) for i in range(n_blocks - 1)])
            output_linear = nn.Linear(hidden_size, 1)
            #RGB decoder
            c_linears = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size)] +
                [nn.Linear(hidden_size, hidden_size)  for i in range(n_blocks - 1)])
            c_output_linear = nn.Linear(hidden_size, 3)
                
        
        self.linears = linears.to(self.device)
        self.output_linear = output_linear.to(self.device) 
        self.c_linears = c_linears.to(self.device)
        self.c_output_linear = c_output_linear.to(self.device)

    def sample_decomp_feature(self, p_nor, planes_xy, planes_xz, planes_yz, lines_z = None, lines_y = None, lines_x = None):
        """
        Sample feature from planes
        Args:
            p_nor (tensor): normalized 3D coordinates
            planes_xy (list): xy planes
            planes_xz (list): xz planes
            planes_yz (list): yz planes
        Returns:
            feat (tensor): sampled features
        """
        vgrid = p_nor[None, :, None].to(torch.float32)
        if lines_z is not None:
            coord_line = torch.stack((p_nor[..., 2], p_nor[..., 1], p_nor[..., 0]))
            coord_line = torch.stack((torch.zeros_like(coord_line), coord_line), dim=-1).detach().view(3, -1, 1, 2).to(torch.float32)

        plane_feat = []
        line_feat = []
        for i in range(len(planes_xy)):
            xy = F.grid_sample(planes_xy[i].to(self.device), vgrid[..., [0, 1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            xz = F.grid_sample(planes_xz[i].to(self.device), vgrid[..., [0, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            yz = F.grid_sample(planes_yz[i].to(self.device), vgrid[..., [1, 2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
            plane_feat.append(xy + xz + yz)
            if lines_x is not None:
                z = F.grid_sample(lines_z[i].to(self.device), coord_line[[0]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
                y = F.grid_sample(lines_y[i].to(self.device), coord_line[[1]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
                x = F.grid_sample(lines_x[i].to(self.device), coord_line[[2]], padding_mode='border', align_corners=True, mode='bilinear').squeeze().transpose(0, 1)
                line_feat.append(z + y + x)
        
        plane_feat = torch.cat(plane_feat, dim=-1)
        line_feat = torch.cat(line_feat, dim=-1) if lines_x is not None else None
        if line_feat is not None:
            feat = plane_feat*line_feat
        else:
            feat = plane_feat
        return feat

    def get_raw_sdf(self, embed_pos, p_nor, all_planes, all_lines = None):
        """
        Get raw SDF
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            sdf (tensor): raw SDF
        """
        if all_lines is None:
            planes_xy, planes_xz, planes_yz = all_planes
            feat = self.sample_decomp_feature(p_nor, planes_xy, planes_xz, planes_yz)
        else:
            planes_xy, planes_xz, planes_yz = all_planes
            lines_z, lines_y, lines_x = all_lines
            feat = self.sample_decomp_feature(p_nor, planes_xy, planes_xz, planes_yz, lines_z, lines_y, lines_x)

        h = torch.cat([embed_pos.to(feat.dtype), feat], dim=-1) #feat
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = F.relu(h, inplace=True)
        sdf_geo = torch.tanh(self.output_linear(h)).squeeze()

        return sdf_geo

    def get_raw_rgb(self, p_nor, embed_pos, all_planes = None, all_lines = None, geo_fea = None):
        """
        Get raw RGB
        Args:
            p_nor (tensor): normalized 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            rgb (tensor): raw RGB
        """
        
        if self.F == 'tri_plane':
            if self.coupling:
                h = torch.cat([embed_pos.to(geo_fea.dtype), geo_fea], dim=-1) #geo_fea
            else:
                # if couling_base, inputed all_planes is geo_planes
                # if not couling_base, inputed all_planes is color_planes
                c_planes_xy, c_planes_xz, c_planes_yz = all_planes
                c_feat = self.sample_decomp_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz)
                h = torch.cat([embed_pos.to(c_feat.dtype), c_feat], dim=-1) #c_feat
        
        elif self.F == 'tensorf':
            if self.coupling:
                h = torch.cat([embed_pos.to(geo_fea.dtype), geo_fea], dim=-1) #geo_fea
            else:
                c_planes_xy, c_planes_xz, c_planes_yz = all_planes
                c_lines_z, c_lines_y, c_lines_x = all_lines
                c_feat = self.sample_decomp_feature(p_nor, c_planes_xy, c_planes_xz, c_planes_yz, c_lines_z, c_lines_y, c_lines_x)
                h = torch.cat([embed_pos.to(c_feat.dtype), c_feat], dim=-1) #c_feat
  
        for i, l in enumerate(self.c_linears):
            h = self.c_linears[i](h)
            h = F.relu(h, inplace=True)
        rgb = self.c_output_linear(h) #torch.sigmoid(self.c_output_linear(h))
        return rgb

    def forward(self, p, pos_embed_fn, all_geo_planes, all_c_planes = None, all_geo_lines = None, all_c_lines = None):
        """
        Forward pass
        Args:
            p (tensor): 3D coordinates
            all_planes (Tuple): all feature planes
        Returns:
            raw (tensor): raw SDF and RGB
        """
        
        #p_shape = p.shape
        #p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        inputs_flat = torch.reshape(p, [-1, p.shape[-1]])
        p_nor = (inputs_flat - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])

        embed_pos = pos_embed_fn(p_nor)
        
        sdf_geo = self.get_raw_sdf(p_nor, embed_pos, all_geo_planes, all_geo_lines)
        
        if self.coupling:
            #share encoder, but no connection between geo and app
            sdf, geo_feat = sdf_geo[...,:1], sdf_geo[...,1:]
            rgb = self.get_raw_rgb(p_nor, embed_pos, geo_fea = geo_feat)
        elif self.coupling_base:
            sdf = sdf_geo.unsqueeze(-1)
            rgb = self.get_raw_rgb(p_nor, embed_pos, all_planes = all_geo_planes, all_lines = all_geo_lines, geo_fea = None)
        elif (not self.coupling) and (not self.coupling_base):
            #separate encoder, then need color embeding
            sdf = sdf_geo.unsqueeze(-1)
            rgb = self.get_raw_rgb(p_nor, embed_pos, all_planes = all_c_planes, all_lines = all_c_lines, geo_fea = None)

        raw = torch.cat([rgb, sdf], dim=-1)
        #raw = raw.reshape(*p_shape[:-1], -1)
        return raw


class Decoder(nn.Module):
    def __init__(self, device, bound, depth=4,width=256,in_dim=3,skips=[4],geo_feat_dim = None,coupling = False,beta=None):
        super().__init__()
        self.skips = skips
        self.coupling = coupling
        self.device = device
        self.bound = bound
        self.geo_feat_dim = geo_feat_dim
        if (self.geo_feat_dim==0) and (self.coupling):
            self.coupling_base = True
            self.coupling = False
        else:
            self.coupling_base = False
        width_p = 32
        if beta is not None:
            self.beta = beta
            
        pts_linears = nn.ModuleList(
            [nn.Linear(in_dim, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + in_dim, width) for i in range(depth-1)])
        self.pts_linears = pts_linears.to(self.device)
        if (not self.coupling) and (not self.coupling_base):
            pts_linears_c = nn.ModuleList(
                [nn.Linear(in_dim, width)] + [nn.Linear(width, width) if i not in self.skips else nn.Linear(width + in_dim, width) for i in range(depth-1)])
            self.pts_linears_c = pts_linears_c.to(self.device)
        
        if self.coupling:
            sdf_out = nn.Sequential(
                nn.Linear(width, width_p),#+in_dim, width),
                nn.ReLU(),
                nn.Linear(width_p, 1+self.geo_feat_dim))
            color_out = nn.Sequential(
                nn.Linear(self.geo_feat_dim+in_dim, width_p),#+in_dim, width),
                nn.ReLU(),
                nn.Linear(width_p, 3))#,nn.Sigmoid())
        else:
            sdf_out = nn.Sequential(
                nn.Linear(width, width_p),#+in_dim, width),
                nn.ReLU(),
                nn.Linear(width_p, 1))
            color_out = nn.Sequential(
                nn.Linear(width, width_p),#+in_dim, width),
                nn.ReLU(),
                nn.Linear(width_p, 3))#,nn.Sigmoid())
        
        self.sdf_out = sdf_out.to(self.device)
        self.color_out = color_out.to(self.device)
            
            #output_linear = nn.Linear(width, 4)
            #self.output_linear = output_linear.to(self.device)

    def get_values(self, x):
        h = x
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        if (not self.coupling) and (not self.coupling_base):
            h_c = x        
            for i, l in enumerate(self.pts_linears_c):
                h_c = self.pts_linears_c[i](h_c)
                h_c = F.relu(h_c)
                if i in self.skips:
                    h_c = torch.cat([x, h_c], -1)

        if self.coupling:
            sdf_out = self.sdf_out(h)
            sdf = sdf_out[:, :1]
            sdf_feat = sdf_out[:, 1:]
            
            h = torch.cat([sdf_feat, x], dim=-1)#h=sdf_feat
            rgb = self.color_out(h)
        elif self.coupling_base:
            sdf = self.sdf_out(h)
            rgb = self.color_out(h)
        elif (not self.coupling) and (not self.coupling_base):
            #outputs = self.output_linear(h) #outputs[:, :3] = torch.sigmoid(outputs[:, :3])
            sdf = self.sdf_out(h)
            rgb = self.color_out(h_c)
            
        outputs = torch.cat([rgb, sdf], dim=-1)
        return outputs

    def forward(self, query_points, pos_embed_fn):
        #pointsf = query_points.reshape(-1, 3)
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        pointsf = (inputs_flat - self.bound[:, 0]) / (self.bound[:, 1] - self.bound[:, 0])
        
        x = pos_embed_fn(pointsf)
        outputs = self.get_values(x)

        return outputs


def get_decoder(encoder, device ,bound, config, input_ch=32, pos_ch=3, geo_feat_dim=15, hidden_dim=32, num_layers=2, beta=None):
    if encoder == 'freq':
        coupling = config['NeRFs']['tricks']['share_encoder']
        decoder = Decoder(device = device,bound=bound,in_dim=input_ch,geo_feat_dim =geo_feat_dim, coupling = coupling, beta = beta)
    elif (encoder == 'dense') or (encoder == 'hash'):
        decoder = Grid_Decoder(bound=bound, config=config, input_ch=input_ch, pos_ch=pos_ch, geo_feat_dim=geo_feat_dim, 
                                hidden_dim=hidden_dim, num_layers=num_layers, device = device, beta = beta)
    elif (encoder == 'tri_plane') or (encoder == 'tensorf'):
        decoder = Decomp_Decoders(device = device, bound=bound, config=config, pos_ch=pos_ch, c_dim=input_ch, geo_feat_dim=geo_feat_dim, 
                                   hidden_size=hidden_dim, n_blocks=num_layers, beta = beta)
    return decoder