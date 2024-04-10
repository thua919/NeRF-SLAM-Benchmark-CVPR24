import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast


def alpha_compositing_weights(alphas):
    """Alpha compositing of (sampled) MPIs given their RGBs and alphas.
    Args:
        alphas (tensor [batch,ray,samples]): The predicted opacity values.
    Returns:
        weights (tensor [batch,ray,samples,1]): The predicted weight of each MPI (in [0,1]).
    """
    alphas_front = torch.cat([torch.zeros_like(alphas[..., :1]),
                              alphas[..., :-1]], dim=2)  # [B,R,N]
    with autocast(enabled=False):  # TODO: may be unstable in some cases.
        visibility = (1 - alphas_front).cumprod(dim=2)  # [B,R,N]
    weights = (alphas * visibility)[..., None]  # [B,R,N,1]
    return weights


def composite(quantities, weights):
    """Composite the samples to render the RGB/depth/opacity of the corresponding pixels.
    Args:
        quantities (tensor [batch,ray,samples,k]): The quantity to be weighted summed.
        weights (tensor [batch,ray,samples,1]): The predicted weight of each sampled point along the ray.
    Returns:
        quantity (tensor [batch,ray,k]): The expected (rendered) quantity.
    """
    # Integrate RGB and depth weighted by probability.
    quantity = (quantities * weights).sum(dim=2)  # [B,R,K]
    return quantity


def compute_loss(prediction, target, loss_type='l2'):
    '''
    Params: 
        prediction: torch.Tensor, (Bs, N_samples)
        target: torch.Tensor, (Bs, N_samples)
        loss_type: str
    Return:
        loss: torch.Tensor, (1,)
    '''

    if loss_type == 'l2':
        return F.mse_loss(prediction, target)
    elif loss_type == 'l1':
        return F.l1_loss(prediction, target)

    raise Exception('Unsupported loss type')