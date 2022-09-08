import paddle


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
layer.load('./export.jit', paddle.CPUPlace())
print(dir(layer))

xs1 = paddle.full([1, 67, 80], 0.1, dtype='float32')
offset = paddle.to_tensor([0], dtype='int32')
att_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')
cnn_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')

# print(xs1.numpy())
print(paddle.shape(att_cache)[0])

func = getattr(layer, 'jit.forward_encoder_chunk')
# xs, att_cache, cnn_cache = layer.forward_encoder_chunk(xs1, offset, att_cache, cnn_cache)
xs, att_cache, cnn_cache = func(xs1, offset, att_cache, cnn_cache)
print(xs.numpy())
print(att_cache.numpy())
print(cnn_cache.numpy())

