from matplotlib import pyplot as plt
import numpy as np
import torch

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.

    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def plot_depth_color(NeRF, c2w,H ,W ,fx ,fy ,cx ,cy ,gt_depth,gt_color,output,opt_iter,device):
    '''
    gt_depth, gt_color, cur_c2ws are stream of tensors
    idx is to index them
    '''
    with torch.no_grad():
        rays_o ,rays_d = get_rays(H, W, fx, fy, cx, cy, c2w, device)
        rays_o = torch.reshape(rays_o, [-1, rays_o.shape[-1]])
        rays_d = torch.reshape(rays_d, [-1, rays_d.shape[-1]])
        target_d = torch.reshape(gt_depth, [-1, 1])
        ray_batch_size = 10000 #0
        
        depth_list = []
        color_list = []
        for i in range(0, rays_d.shape[0], ray_batch_size):
            rays_d_batch = rays_d[i : i + ray_batch_size]
            rays_o_batch = rays_o[i : i + ray_batch_size]
            target_d_batch = (
                    target_d[i : i + ray_batch_size] if target_d is not None else None
                )

            color, depth = NeRF.render_rays(rays_o_batch, rays_d_batch, 
                                            target_d = target_d_batch, 
                                            render_img=True)
        
            depth_list.append(depth.double())
            color_list.append(color)
        
        depth = torch.cat(depth_list, dim=0)
        color = torch.cat(color_list, dim=0)
        
        depth_np = depth.reshape(H, W).detach().cpu().numpy()
        color_np = color.reshape(H, W, 3).detach().cpu().numpy()
        gt_depth_np = gt_depth.detach().cpu().numpy()
        gt_color_np = gt_color.detach().cpu().numpy()
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0

        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        max_depth = np.max(gt_depth_np)
        max_depth = 2 if max_depth == 0 else max_depth
        axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 0].set_title("Input Depth")
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 1].set_title("Generated Depth")
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 2].set_title("Depth Residual")
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title("Input RGB")
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title("Generated RGB")
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap="plasma")
        axs[1, 2].set_title("RGB Residual")
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(
            f"{output}/{opt_iter:04d}.jpg",
            bbox_inches="tight",
            pad_inches=0.2,
        )
        plt.clf()
        torch.cuda.empty_cache()

def render_img(self, c2w, device, gt_depth=None, **kwargs):
        """
        Renders out depth, uncertainty, and color images.

        Args:
            c (dict): feature grids.
            decoders (nn.module): decoders.
            c2w (tensor): camera to world matrix of current frame.
            device (str): device name to compute on.
            stage (str): query stage.
            gt_depth (tensor, optional): sensor depth image. Defaults to None.

        Returns:
            depth (tensor, H*W): rendered depth image.
            uncertainty (tensor, H*W): rendered uncertainty image.
            color (tensor, H*W*3): rendered color image.
        """
        with torch.no_grad():
            H = self.H
            W = self.W
            rays_o, rays_d = get_rays(
                H, W, self.fx, self.fy, self.cx, self.cy, c2w, device
            )
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            depth_list = []
            uncertainty_list = []
            color_list = []

            ray_batch_size = self.ray_batch_size
            if gt_depth is not None:
                gt_depth = gt_depth.reshape(-1)

            if "render_size" in kwargs:
                skip = rays_d.shape[0] // kwargs["render_size"]
                kwargs["skip"] = skip

            if "skip" in kwargs:
                rays_d = rays_d[:: kwargs["skip"]]
                rays_o = rays_o[:: kwargs["skip"]]

            for i in range(0, rays_d.shape[0], ray_batch_size):
                rays_d_batch = rays_d[i : i + ray_batch_size]
                rays_o_batch = rays_o[i : i + ray_batch_size]
                gt_depth_batch = (
                    gt_depth[i : i + ray_batch_size] if gt_depth is not None else None
                )
                ret = self.render_batch_ray(
                    rays_d_batch,
                    rays_o_batch,
                    device,
                    gt_depth=gt_depth_batch,
                    **kwargs
                )

                depth, uncertainty, color, extra_ret, density = ret
                depth_list.append(depth.double())
                uncertainty_list.append(uncertainty.double())
                color_list.append(color)

            depth = torch.cat(depth_list, dim=0)
            uncertainty = torch.cat(uncertainty_list, dim=0)
            color = torch.cat(color_list, dim=0)

            if "skip" not in kwargs or kwargs["skip"] == 1:
                depth = depth.reshape(H, W)
                uncertainty = uncertainty.reshape(H, W)
                color = color.reshape(H, W, 3)
            return depth, uncertainty, color