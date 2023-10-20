import torch
import torch.fft as fft


def Fourier_filter(x, threshold, scale):

    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold,
         ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))

    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered

from deps.ControlNet.ldm.modules.diffusionmodules.util import \
    timestep_embedding  # noqa：E501


# backbone_scale1=1.1, backbone_scale2=1.2, skip_scale1=1.0, skip_scale2=0.2
def freeu_forward(self,
                  backbone_scale1=1.,
                  backbone_scale2=1.,
                  skip_scale1=1.,
                  skip_scale2=1.):

    def forward(x,
                timesteps=None,
                context=None,
                control=None,
                only_mid_control=False,
                **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps,
                                       self.model_channels,
                                       repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()
        '''
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
        '''
        for i, module in enumerate(self.output_blocks):
            hs_ = hs.pop()

            if h.shape[1] == 1280:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
                h[:, :640] = h[:, :640] * ((backbone_scale1 - 1) * hidden_mean + 1)        
                # h[:, :640] = h[:, :640] * backbone_scale1
                hs_ = Fourier_filter(hs_, threshold=1, scale=skip_scale1)
            if h.shape[1] == 640:
                hidden_mean = h.mean(1).unsqueeze(1)
                B = hidden_mean.shape[0]
                hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
                hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3) 
                h[:, :320] = h[:, :320] * ((backbone_scale2 - 1) * hidden_mean + 1)
                # h[:, :320] = h[:, :320] * backbone_scale2
                hs_ = Fourier_filter(hs_, threshold=1, scale=skip_scale2)

            if only_mid_control or control is None:
                h = torch.cat([h, hs_], dim=1)
            else:
                h = torch.cat([h, hs_ + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

    return forward
