import paddle
xs1 = paddle.rand(shape=[1, 67, 80], dtype='float32')
from paddle.jit.layer import Layer
layer = Layer()
layer.load('./export.jit', paddle.CPUPlace())

offset = paddle.to_tensor([0], dtype='int32')
att_cache = paddle.zeros([0, 0, 0, 0])
cnn_cache=paddle.zeros([0, 0, 0, 0])
func = getattr(layer, 'jit.forward_encoder_chunk')
xs, att_cache, cnn_cache = func(xs1, offset, att_cache, cnn_cache)
print(xs)

