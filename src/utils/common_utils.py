

import torch

def mse2psnr(x):
    '''
    MSE to PSNR
    '''
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)

def normalize_3d_coordinate(p, bound):
    """
    Normalize 3d coordinate to [-1, 1] range.
    Args:
        p: (N, 3) 3d coordinate
        bound: (3, 2) min and max of each dimension
    Returns:
        (N, 3) normalized 3d coordinate

    """
    p = p.reshape(-1, 3)
    p[:, 0] = ((p[:, 0]-bound[0, 0])/(bound[0, 1]-bound[0, 0]))*2-1.0
    p[:, 1] = ((p[:, 1]-bound[1, 0])/(bound[1, 1]-bound[1, 0]))*2-1.0
    p[:, 2] = ((p[:, 2]-bound[2, 0])/(bound[2, 1]-bound[2, 0]))*2-1.0
    return p

def intersect_with_sphere(center, ray_unit, radius=1.0):
    ctc = (center * center).sum(dim=-1, keepdim=True)  # [...,1]
    ctv = (center * ray_unit).sum(dim=-1, keepdim=True)  # [...,1]
    b2_minus_4ac = ctv ** 2 - (ctc - radius ** 2)
    dist_near = -ctv - b2_minus_4ac.sqrt()
    dist_far = -ctv + b2_minus_4ac.sqrt()
    return dist_near, dist_far