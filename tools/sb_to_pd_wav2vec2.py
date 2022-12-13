import paddle
import torch
import numpy as np

## in sb file: https://github.com/speechbrain/speechbrain/blob/5063efb641285213e1fe618cdd46c1eaef9ef209/speechbrain/core.py#L988
## use torch.save(self.modules.state_dict(), 'test_init.ckpt') to save

torch_model_dict = torch.load('test_init.ckpt')


# step 1: 
def torch_paddle_param(model_state_dict):
    paddle_state_dict = {}
    for n, p in model_state_dict.items():
        print(f'-> name: {n} dim: {p.ndim}')
        
        # change norm.mean and norm variance
        name_change=True
        if 'norm.running_mean' in n:
            new_n = n.replace('norm.running_', 'norm._')
        elif 'norm.running_var' in n:
            new_n = n.replace('norm.running_var', 'norm._variance')
        else:
            name_change=False
            new_n = n
        if name_change:
            print(f"\t norm mean/var: {n} -> {new_n}")

        p = p.cpu().detach().numpy()
        # weight which is rank 2, transpose it
        if n.endswith('weight') and p.ndim == 2:
            new_p = p.T
            print(f"\t liner transpose: {n}: {p.shape} -> {new_p.shape}")
        else:
            new_p = p
        
        # text embedding layer
        if 'decoder.embed.0.weight' in n:
            new_p = p

        if 'global_cmvn.mean' in n:
            print("\t cmvn:", p, p.dtype)
        if 'global_cmvn.istd' in n:
            print("\t istd:", p, p.dtype)

        paddle_state_dict[new_n] = new_p
    return paddle_state_dict

paddle_model_state_dict = torch_paddle_param(torch_model_dict)
eq = []
for key, val in torch_model_dict.items():
    val = val.cpu().detach().numpy()
    if 'norm.running_mean' in key:
        key = key.replace('norm.running_', 'norm._')
    elif 'norm.running_var' in key:
        key = key.replace('norm.running_var', 'norm._variance')
    if 'decoder.embed.0.weight' in key:
        # text embeding
        val = val
    elif key.endswith('weight') and val.ndim == 2:
        # linear weight
        val = val.T  
    eq.append(np.allclose(val, paddle_model_state_dict[key]))         
print("all param equal:", np.array(eq).all())



# specific for sb to pd
stage_2_paddle_model_state_dict = {}

for key, val in paddle_model_state_dict.items():
    if 'wav2vec2' in key: 
        stage_2_paddle_model_state_dict[key.replace('.model.', '.')] = val
    
    elif 'enc' in key:
        tag = int(key.split('.')[1][-1])
        if tag != 1:
            stage_2_paddle_model_state_dict[key.replace(str(tag), '_'+str(tag-2))] = val
        else:
            stage_2_paddle_model_state_dict[key.replace(key.split('.')[1], key.split('.')[1][:-1])] = val
    elif 'ctc' in key:        stage_2_paddle_model_state_dict['ctc.' + key.replace('ctc_lin.w', 'ctc_lo')] = val

## 特殊转换pos_conv_embed.conv.weight_g
stage_2_paddle_model_state_dict['wav2vec2.encoder.pos_conv_embed.conv.weight_g'] = stage_2_paddle_model_state_dict['wav2vec2.encoder.pos_conv_embed.conv.weight_g'].squeeze(0).squeeze(0)

paddle.save(stage_2_paddle_model_state_dict, 'avg_1.pdparams')
