
import torch
import os

dictionary=torch.load("pre_trained_models/ldm/layout2img-openimages256/model.ckpt",map_location='cpu')

new_dict={}
keys=dictionary['state_dict'].keys()

for i,params in enumerate(keys):
    if 'ddim_sigmas' != params and 'ddim_alphas' !=params and 'ddim_alphas_prev' !=params and 'ddim_sqrt_one_minus_alphas' !=params and 'cond_stage_model' not in params:
        new_dict[params]=dictionary['state_dict'][params]

#If you want to change the conditional keys in the pretrained model to a personal conditional model with random weights.

model=torch.nn.Linear(2048,640)

for param in model.parameters():
    param.data = torch.randn(param.data.size())

s_d=model.state_dict()
keys=s_d.keys()

for i, params in enumerate(keys):
    new_dict['cond_stage_model.model.'+params]=s_d[params]


dictionary['state_dict']=new_dict

torch.save(dictionary,"pre_trained_models/ldm/layout2img-openimages256/new_model.ckpt")