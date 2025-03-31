import torch
import math
from pathlib import Path

def get_pos_neg_text_embeddings(embeddings, azimuth_val, opt):
    if azimuth_val >= -90 and azimuth_val < 90:
        if azimuth_val >= 0:
            r = 1 - azimuth_val / 90
        else:
            r = 1 + azimuth_val / 90
        start_z = embeddings['front']
        end_z = embeddings['side']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['front'], embeddings['side']], dim=0)
        if r > 0.8:
            front_neg_w = 0.0
        else:
            front_neg_w = math.exp(-r * opt['front_decay_factor']) * opt['negative_w']
        if r < 0.2:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-(1-r) * opt['side_decay_factor']) * opt['negative_w']

        weights = torch.tensor([1.0, front_neg_w, side_neg_w])
    else:
        if azimuth_val >= 0:
            r = 1 - (azimuth_val - 90) / 90
        else:
            r = 1 + (azimuth_val + 90) / 90
        start_z = embeddings['side']
        end_z = embeddings['back']
        # if random.random() < 0.3:
        #     r = r + random.gauss(0, 0.08)
        pos_z = r * start_z + (1 - r) * end_z
        text_z = torch.cat([pos_z, embeddings['side'], embeddings['front']], dim=0)
        front_neg_w = opt['negative_w']
        if r > 0.8:
            side_neg_w = 0.0
        else:
            side_neg_w = math.exp(-r * opt['side_decay_factor']) * opt['negative_w'] / 2

        weights = torch.tensor([1.0, side_neg_w, front_neg_w])
    return text_z, weights.to(text_z.device)

def adjust_text_embeddings(embeddings, azimuth, opt):
    text_z_list = []
    weights_list = []
    K = 0
    for b in range(azimuth.shape[0]):
        text_z_, weights_ = get_pos_neg_text_embeddings(embeddings, azimuth[b], opt)
        K = max(K, weights_.shape[0])
        text_z_list.append(text_z_)
        weights_list.append(weights_)

    # Interleave text_embeddings from different dirs to form a batch
    text_embeddings = []
    for i in range(K):
        for text_z in text_z_list:
            # if uneven length, pad with the first embedding
            text_embeddings.append(text_z[i] if i < len(text_z) else text_z[0])
    text_embeddings = torch.stack(text_embeddings, dim=0) # [B * K, 77, 768]

    # Interleave weights from different dirs to form a batch
    weights = []
    for i in range(K):
        for weights_ in weights_list:
            weights.append(weights_[i] if i < len(weights_) else torch.zeros_like(weights_[0]))
    weights = torch.stack(weights, dim=0) # [B * K]
    return text_embeddings, weights

class sds_loss():

    def __init__(self,opt,guidance,device):

        self.opt = opt
        import diffusion_prior.sd_utils as sd
        from .if_utils import  IF
        import diffusion_prior.dds_utils as dds
        self.guidance_type = guidance
        self.guidance = {guidance: None}
        if guidance == 'SD':
            self.guidance['SD'] = sd.StableDiffusion(device, opt['fp16'], opt['vram_0'], opt['sd_version'], opt['hf_key'], opt['t_range'])
        elif guidance == 'IF':
            self.guidance['IF'] = IF(device, opt['vram_0'], opt['t_range'])
        elif guidance == 'DDS':
            self.guidance['DDS'] = dds.StableDiffusion(device, opt['fp16'], opt['vram_0'], opt['sd_version'], opt['hf_key'], opt['t_range'])
        self.embeddings = {}

        # text prompt / images
        if self.guidance is not None:
            for key in self.guidance:
                for p in self.guidance[key].parameters():
                    p.requires_grad = False
                self.embeddings[key] = {}
            self.prepare_embeddings()

    # calculate the text embs.
    @torch.no_grad()
    def prepare_embeddings(self):

        # text embeddings (stable-diffusion)
        if self.opt['text'] is not None:

            if 'SD' in self.guidance:
                self.embeddings['SD']['default'] = self.guidance['SD'].get_text_embeds([self.opt['text']])
                self.embeddings['SD']['uncond'] = self.guidance['SD'].get_text_embeds([self.opt['negative']])

                for d in ['front', 'side', 'back']:
                    self.embeddings['SD'][d] = self.guidance['SD'].get_text_embeds([f"{self.opt['text']}, {d} view"])
            if 'IF' in self.guidance:
                self.embeddings['IF']['default'] = self.guidance['IF'].get_text_embeds([self.opt['text']])
                self.embeddings['IF']['uncond'] = self.guidance['IF'].get_text_embeds([self.opt['negative']])

                for d in ['front', 'side', 'back']:
                    self.embeddings['IF'][d] = self.guidance['IF'].get_text_embeds([f"{self.opt['text']}, {d} view"])
            if 'DDS' in self.guidance:
                self.embeddings['DDS']['default'] = self.guidance['DDS'].get_text_embeds([self.opt['text']])
                self.embeddings['DDS']['uncond'] = self.guidance['DDS'].get_text_embeds([self.opt['negative']])

                for d in ['front', 'side', 'back']:
                    self.embeddings['DDS'][d] = self.guidance['DDS'].get_text_embeds([f"{self.opt['text']}, {d} view"])

    def compute(self,azim,rgb,ref_rgb=None):

        # interpolate text_z

        text_z = [self.embeddings[self.guidance_type]['uncond']] * azim.shape[0]

        text_z_comp, weights = adjust_text_embeddings(self.embeddings[self.guidance_type], azim, self.opt)
        text_z.append(text_z_comp)
        text_z = torch.cat(text_z, dim=0)
        if self.guidance_type == 'SD':
            loss = self.guidance['SD'].train_step_perpneg(text_z, weights, rgb, as_latent=False, guidance_scale=self.opt['guidance_scale'], grad_scale=self.opt['lambda_guidance'],
                                                    )
        elif self.guidance_type == 'IF':
            loss = self.guidance['IF'].train_step_perpneg(text_z, weights, rgb,
                                                                        guidance_scale=self.opt['guidance_scale'],
                                                                        grad_scale=self.opt['lambda_guidance']
                                                                        )
        elif self.guidance_type == 'DDS':
            assert ref_rgb != None

            loss, n= self.guidance['DDS'].train_step_perpneg(text_z, weights, rgb, ref_rgb, as_latent=False,
                                                          guidance_scale=self.opt['guidance_scale'],
                                                          grad_scale=self.opt['lambda_guidance'],
                                                          )

        return loss