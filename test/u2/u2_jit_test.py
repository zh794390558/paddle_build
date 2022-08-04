import paddle


import numpy as np
#np.set_printoptions(threshold=np.inf)

paddle.set_device('cpu')

# def flatten(out):
#     if isinstance(out, paddle.Tensor):
#         return [out]
#     flatten_out = []
#     for var in out:
#         if isinstance(var, (list, tuple)):
#             flatten_out.extend(var)
#         else:
#             flatten_out.append(var)
#     return flatten_out

from paddle.jit.layer import Layer
layer = Layer()
layer.load('chunk_wenetspeech_static/export.jit', paddle.CPUPlace())
print(dir(layer))

xs1 = paddle.full([1, 7, 80], 0.1, dtype='float32')
offset = paddle.to_tensor([0], dtype='int32')
att_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')
cnn_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')

func = getattr(layer, 'jit.forward_encoder_chunk')
# xs, att_cache, cnn_cache = layer.forward_encoder_chunk(xs1, offset, att_cache, cnn_cache)
xs, att_cache, cnn_cache = func(xs1, offset, att_cache, cnn_cache)

print('encoder out', xs.shape, xs.numpy())
print('att cache', att_cache.shape, att_cache.numpy())
print('cnn cache', cnn_cache.shape, cnn_cache.numpy())


func = getattr(layer, 'jit.ctc_activation')
ys = func(xs)
print('log_probs', ys[0].numpy())
print('log_probs shape', ys[0].shape)
