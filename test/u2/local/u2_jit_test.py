import paddle


import numpy as np
np.set_printoptions(threshold=np.inf)

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
layer.load('asr1_chunk_conformer_wenetspeech_ckpt_1.0.0a.model/export.jit', paddle.CPUPlace())
print(dir(layer))

xs1 = paddle.full([1, 7, 80], 0.1, dtype='float32')
offset = paddle.to_tensor([0], dtype='int32')
att_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')
cnn_cache = paddle.zeros([0, 0, 0, 0], dtype='float32')

func = getattr(layer, 'forward_encoder_chunk')
# xs, att_cache, cnn_cache = layer.forward_encoder_chunk(xs1, offset, att_cache, cnn_cache)
xs, att_cache, cnn_cache = func(xs1, offset, att_cache, cnn_cache)

print('encoder out', xs.shape, xs.numpy())
print('att cache', att_cache.shape, att_cache.numpy())
print('cnn cache', cnn_cache.shape, cnn_cache.numpy())


func = getattr(layer, 'ctc_activation')
ys = func(xs)
print('log_probs', ys[0].numpy())
print('log_probs shape', ys[0].shape)

#exit(0)

# print("###################")
# # ######################### infer_model.forward_attention_decoder ########################
# B = 2
# U = 8
# hyps = paddle.full(shape=[B, U], fill_value=10, dtype='int64') # hyps
# hyp_lens = paddle.full(shape=[B], fill_value=8, dtype='int64') # hyps lens
# encoder_outs = paddle.full(shape=[1, 20, 512], fill_value=1, dtype='float32') # encoder outs

# func = getattr(layer, 'forward_attention_decoder')
# out2 = func(hyps, hyp_lens, encoder_outs)
# print("decoder logits", out2[0])
# print("decoder logits", out2[0].shape)


print("###################")
# ######################### infer_model.forward_attention_decoder ########################
B = 1
U = 8 # <=7 fatal

# add sos
hyps = paddle.full(shape=[B, U], fill_value=10, dtype='int64') # hyps
sos = paddle.full(shape=[B,1], fill_value=5537, dtype='int64')
hyps = paddle.concat([sos, hyps], axis=-1)

hyp_lens = paddle.full(shape=[B], fill_value=U+1, dtype='int64') # hyps lens

encoder_outs = paddle.full(shape=[1, 20, 512], fill_value=1, dtype='float32') # encoder outs

func = getattr(layer, 'forward_attention_decoder')
out2 = func(hyps, hyp_lens, encoder_outs)
print("decoder logits", out2[0].numpy())
print("decoder logits", out2[0].shape)
print("hyps:", hyps)
print("hyps len:", hyp_lens)
