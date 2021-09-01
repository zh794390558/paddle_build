from yacs.config import CfgNode
import paddle
from collections import OrderedDict


def allclose(a, b, atol=1e-5, rtol=0.0):
    if isinstance(a, (list, tuple)):
        return all([allclose(i, j) for i, j in zip(a, b)])
    #return np.allclose(a, b, atol, rtol)
    return np.all( np.abs(a - b) < atol )



def load_conf(filepath):
   config = CfgNode.load_cfg(open(filepath, 'rt'))
   return config



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
    if isinstance(xx, paddle.Tensor):
        return xx.numpy()
    return xx


def forward_hook(m, ins, outs):
    global model
    for n, mod in model.named_sublayers():
        if m is mod:
            forward_dict[n] = {
                'inputs': tensor_to_numpy(ins),
                'outputs':  tensor_to_numpy(outs),
            }

#def backward_hook(m, grad_ins, grad_outs):
#    global model
#     if grad_outs is None or grad_outs[0] is None:
#            return
#     for n, mod in model.named_sublayers():
#        if m is mod:
#            backward_dict[n] = {
#                'grad_outs': tensor_to_numpy(grad_outs),
#                'grad_ins': tensor_to_numpy(grad_ins),
#            }
#

def paddle_forward_hook(model):
    for n, m in model.named_sublayers():
        if isinstance(m, paddle.nn.Layer):
            try:
                m.register_forward_post_hook(forward_hook)
                #m.register_backward_hook(backward_hook)
            except Exception as e:
                print(n, e)


def backwardhook(grad):
    global model
    if grad is None:
            return
    val_name = grad.name[:-5]
    for n, val in model.named_parameters():
        if val.name == val_name:
            backward_dict[n] = {
                'grad': tensor_to_numpy(grad),
            }

def paddle_backward_hook(model):
    for n, val in model.named_parameters():
        val.register_hook(backwardhook)
