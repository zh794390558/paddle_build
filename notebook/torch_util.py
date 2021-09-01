

def torch_paddle_param(model_state_dict):
    paddle_state_dict = {}
    for n, p in model_state_dict.items():
        print(f'-> {n}    {p.ndim}')

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
            print(f"{n} -> {new_n}")

        p = p.cpu().detach().numpy()

        # weight with is rank 2, transpose it
        if n.endswith('weight') and p.ndim == 2:
            new_p = p.T
            print(f"\t{n}: {p.shape} -> {new_p.shape}")
        else:
            new_p = p

        # embedding layer
        if 'decoder.embed.0.weight' in n:
            new_p = p

        if 'global_cmvn.mean' in n:
            print("cmvn:", p, p.dtype)

        paddle_state_dict[new_n] = new_p
    return paddle_state_dict



def torch_model_grad(model):
    for key, val in model.named_parameters():
        if val.grad is None:
            continue
        print(key, "\t",  val.grad.detach().numpy())


###### register hook to get forward and backward inputs/outputs/grad_outs/grad_ins

forward_dict=OrderedDict()
backward_dict=OrderedDict()
def tensor_to_numpy(xx):
    if isinstance(xx, (list, tuple)):
        return [tensor_to_numpy(x) for x in xx]
    if isinstance(xx, dict):
        return {key: tensor_to_numpy(val) for key, val in xx}
    if xx is None:
        return None
    if isinstance(xx, torch.Tensor):
        return xx.detach().numpy()
    return xx

def forward_hook(m, ins, outs):
    global model
    for n, mod in model.named_modules():
        if m is mod:
            forward_dict[n] = {
                'inputs': tensor_to_numpy(ins),
                'outputs':  tensor_to_numpy(outs),
            }

def backward_hook(m, grad_ins, grad_outs):
    global model
    if grad_outs is None or grad_outs[0] is None:
        return

    for n, mod in model.named_modules():
        if m is mod:
            backward_dict[n] = {
                'grad_outs': tensor_to_numpy(grad_outs),
                'grad_ins': tensor_to_numpy(grad_ins),
            }


def torch_forward_backward_hook(model):
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Module):
            try:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
            except Exception as e:
                print(n, e)

