import einops
import torch
import torch.nn.functional as F


@torch.no_grad()
def find_flat_region(mask):
    device = mask.device
    kernel_x = torch.Tensor([[-1, 0, 1], [-1, 0, 1],
                             [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    kernel_y = torch.Tensor([[-1, -1, -1], [0, 0, 0],
                             [1, 1, 1]]).unsqueeze(0).unsqueeze(0).to(device)
    mask_ = F.pad(mask.unsqueeze(0), (1, 1, 1, 1), mode='replicate')

    grad_x = torch.nn.functional.conv2d(mask_, kernel_x)
    grad_y = torch.nn.functional.conv2d(mask_, kernel_y)
    return ((abs(grad_x) + abs(grad_y)) == 0).float()[0]


def numpy2tensor(img):
    x0 = torch.from_numpy(img.copy()).float().cuda() / 255.0 * 2.0 - 1.
    x0 = torch.stack([x0], dim=0)
    return einops.rearrange(x0, 'b h w c -> b c h w').clone()
