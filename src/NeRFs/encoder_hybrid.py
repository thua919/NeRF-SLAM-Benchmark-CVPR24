import torch
import numpy as np
import tinycudann as tcnn
import torch.nn as nn


class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale
    def forward(self, x):
        x = x.squeeze(0)
        x = x.to(self._B.dtype)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


def get_encoder(encoding,
                res_level, implicit_dim, 
                resolution_fine, resolution_coarse,
                log2_hashmap_size=19, input_dim=3):
    if 'dense' in encoding.lower():
        per_level_scale = np.exp2(np.log2( max(resolution_fine) / max(resolution_coarse)) / (res_level - 1))
        if res_level==1:
            per_level_scale = 1
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": res_level,
                    "n_features_per_level": implicit_dim,
                    "base_resolution": max(resolution_coarse),
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
        
    elif 'hash' in encoding.lower():
        per_level_scale = np.exp2(np.log2( max(resolution_fine) / max(resolution_coarse)) / (res_level - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": 'HashGrid',
                    "n_levels": res_level,
                    "n_features_per_level": implicit_dim,
                    "log2_hashmap_size": log2_hashmap_size,
                    "base_resolution": max(resolution_coarse),
                    "per_level_scale": per_level_scale},
                dtype=torch.float
        )
        out_dim = embed.n_output_dims

    elif 'tri_plane' or 'tensorf' in encoding.lower():
        
        #per_level_scale_x = np.exp2(np.log2( resolution_fine[0] / resolution_coarse[0]) / (res_level - 1))
        #per_level_scale_y = np.exp2(np.log2( resolution_fine[1] / resolution_coarse[1]) / (res_level - 1))
        #per_level_scale_z = np.exp2(np.log2( resolution_fine[2] / resolution_coarse[2]) / (res_level - 1))
        grid_shape = [resolution_coarse[2], resolution_coarse[1], resolution_coarse[0]]
        #grid_shape_level_scale = [per_level_scale_z, per_level_scale_y, per_level_scale_x]
        planes_xy, planes_xz, planes_yz = [], [], []
        #if 'tensorf' in encoding.lower():
        #    lines_z, lines_y, lines_x = [], [], []
        
        #for i in range(res_level):
        planes_xy.append(torch.empty([1, implicit_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
        planes_xz.append(torch.empty([1, implicit_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
        planes_yz.append(torch.empty([1, implicit_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))
            
            #if 'tensorf' in encoding.lower():
            #    lines_z.append(torch.empty([1, implicit_dim, grid_shape[0],1]).normal_(mean=0, std=0.01))
            #    lines_y.append(torch.empty([1, implicit_dim, grid_shape[1],1]).normal_(mean=0, std=0.01))
            #    lines_x.append(torch.empty([1, implicit_dim, grid_shape[2],1]).normal_(mean=0, std=0.01))    
            
        #    grid_shape = [int(grid_shape[0]*grid_shape_level_scale[0]), 
        #                  int(grid_shape[1]*grid_shape_level_scale[1]), 
        #                  int(grid_shape[2]*grid_shape_level_scale[2])]
            
        if 'tri_plane' in encoding.lower():
            embed = planes_xy, planes_xz, planes_yz
        #if 'tensorf' in encoding.lower():
        #    embed = planes_xy, planes_xz, planes_yz, lines_z, lines_y, lines_x
            
        out_dim = None
        
    return embed, out_dim


def get_pos_embed(embedding,input_dim=3, n_bins=16):
    
    if 'identity' in embedding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    elif 'blob' in embedding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", #Component type.
	            "n_bins": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Frequency encoding from iMAP
    elif 'freq' in embedding.lower():
        print('Use frequency')
        embed = GaussianFourierFeatureTransform(input_dim, mapping_size=93, scale=25)
        out_dim = 93
    return embed, out_dim
    
