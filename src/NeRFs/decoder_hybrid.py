import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.common_utils import normalize_3d_coordinate


class hybrid_Decoder(nn.Module):
    def __init__(self, device, bound, config, input_ch=3, pos_ch=3, geo_feat_dim=15, hidden_dim=32, num_layers=2,beta=None):
        super(hybrid_Decoder, self).__init__()
        self.device = device
        self.config = config
        self.dim = config['NeRFs']['space_resolution']['implicit_dim']
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
                in_dim = self.input_ch-self.dim
                #if self.coupling:
                    # only geo feature passed to color decoder
                #    in_dim = self.geo_feat_dim+self.pos_ch
                #else:
                    # its own color embeding
                #    in_dim = self.input_ch #+self.geo_feat_dim 
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
    
    
    def forward(self, query_points, pos_embed_fn, all_geo_planes, geo_embed_fn, app_embed_fn = None):
        
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
        inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        # get feature embedding for gird
        embed = geo_embed_fn(inputs_flat)
        embed_pos = pos_embed_fn(inputs_flat)
        
        # get feature embedding for plane
        planes_xy, planes_xz, planes_yz = all_geo_planes
        feat = self.sample_decomp_feature(inputs_flat, planes_xy, planes_xz, planes_yz)
        
        #print('feat:', feat.shape)
        #print('embed:', embed.shape)
        #print('embed_pos:', embed_pos.shape)
        #print('inputs_flat', inputs_flat.shape)
        
        sdf = self.sdf_net(torch.cat([embed, feat, embed_pos], dim=-1))
        rgb = self.color_net(torch.cat([embed_pos, embed], dim=-1))
        return torch.cat([rgb, sdf], -1)


def get_decoder(device ,bound, config, input_ch=32, pos_ch=3, 
                geo_feat_dim=0, hidden_dim=32, num_layers=2, beta=None):

    decoder = hybrid_Decoder(bound=bound, config=config, input_ch=input_ch, pos_ch=pos_ch, geo_feat_dim=geo_feat_dim, 
                            hidden_dim=hidden_dim, num_layers=num_layers, device = device, beta = beta)

    return decoder