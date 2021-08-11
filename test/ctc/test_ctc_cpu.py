
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# declarative mode
import paddle.nn.functional as F
import numpy as np
import paddle

# length of the longest logit sequence
max_seq_length = 4
#length of the longest label sequence
max_label_length = 3
# number of logit sequences
batch_size = 2
# class num
class_num = 3

np.random.seed(1)
log_probs = np.array([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
                        [3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],

                        [[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
                        [5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],

                        [[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
                        [6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],

                        [[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
                        [9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],

                        [[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
                        [3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]]).astype("float32")
labels = np.array([[1, 2, 2],
                [1, 2, 2]]).astype("int32")
input_lengths = np.array([5, 5]).astype("int64")
label_lengths = np.array([3, 3]).astype("int64")

log_probs = paddle.to_tensor(log_probs)
labels = paddle.to_tensor(labels)
input_lengths = paddle.to_tensor(input_lengths)
label_lengths = paddle.to_tensor(label_lengths)

loss = F.ctc_loss(log_probs, labels,
    input_lengths,
    label_lengths,
    blank=0,
    reduction='none')
print(loss)  #[3.9179852 2.9076521]

loss = F.ctc_loss(log_probs, labels,
    input_lengths,
    label_lengths,
    blank=0,
    reduction='mean')
print(loss)  #[1.1376063]
