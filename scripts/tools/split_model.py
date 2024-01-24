
import os
import torch

path = r"D:\PycharmProject\checkpoints"

pl_sd = torch.load(os.path.join(path, "last.ckpt"), map_location='cpu')
print(pl_sd.keys())
sd = pl_sd["state_dict"]
# print(sd.keys())

for k, v in sd.items():
    print(k)
    # print(v.shape)

unet_sd = {"state_dict": {k[22:]: v for k, v in sd.items() if k[:21] == 'model.diffusion_model'}}
vq_sd = {"state_dict": {k[18:]: v for k, v in sd.items() if k[:17] == 'first_stage_model'}}
cond_sd = {"state_dict": {k[17:]: v for k, v in sd.items() if k[:16] == 'cond_stage_model'}}
#
torch.save(unet_sd, os.path.join(path, 'unet.ckpt'))
torch.save(vq_sd, os.path.join(path, 'vqvae.ckpt'))
torch.save(cond_sd, os.path.join(path, 'clip.ckpt'))
