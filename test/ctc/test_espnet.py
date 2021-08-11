# declarative mode
import sys
import numpy as np
import paddle
paddle.set_device('cpu')
paddle.seed(1)
np.random.seed(1)

##(B=2)
#log_probs = np.array([[[4.17021990e-01, 7.20324516e-01, 1.14374816e-04],
#			[3.02332580e-01, 1.46755889e-01, 9.23385918e-02]],
#			[[1.86260208e-01, 3.45560730e-01, 3.96767467e-01],
#			[5.38816750e-01, 4.19194520e-01, 6.85219526e-01]],
#			[[2.04452246e-01, 8.78117442e-01, 2.73875929e-02],
#			[6.70467496e-01, 4.17304814e-01, 5.58689833e-01]],
#			[[1.40386939e-01, 1.98101491e-01, 8.00744593e-01],
#			[9.68261600e-01, 3.13424170e-01, 6.92322612e-01]],
#			[[8.76389146e-01, 8.94606650e-01, 8.50442126e-02],
#			[3.90547849e-02, 1.69830427e-01, 8.78142476e-01]]]).astype("float32")
#labels = np.array([[1, 2, 2],
#		[1, 2, 2]]).astype("int32")
#input_lengths = np.array([5, 5]).astype("int64")
#label_lengths = np.array([3, 3]).astype("int64")




log_probs = np.array([[[0.1000, 0.6000, 0.1000, 0.1000, 0.1000]],
                      [[0.1000, 0.1000, 0.6000, 0.1000, 0.1000]]]).astype('float32')
labels = np.array([[1, 2]]).astype('int32')
input_lengths = np.array([2 ]).astype("int64")
label_lengths = np.array([2]).astype("int64")


#log_probs = np.random.rand(2, 13, 4956).astype('float32')
#labels = np.random.rand(2, 13).astype('int32')
#input_lengths = np.random.rand(2).astype("int64")
#label_lengths = np.random.rand(2).astype("int64")


log_probs = paddle.to_tensor(log_probs, stop_gradient=False)
labels = paddle.to_tensor(labels)
input_lengths = paddle.to_tensor(input_lengths)
label_lengths = paddle.to_tensor(label_lengths)


print('-'*10)
loss = paddle.nn.CTCLoss(blank=0, reduction='sum')(log_probs, labels,
			input_lengths,
			label_lengths,
			size_average=False,
			length_average=False)
print('loss', loss)  #[1.1376063]
loss.backward()
print('grad', log_probs.grad)
print(log_probs.grad.shape)
print(log_probs.shape)
log_probs.clear_gradient()

#loss = paddle.nn.CTCLoss(blank=0, reduction='none')(log_probs, labels,
#			input_lengths,
#			label_lengths)
#print(loss)  #[3.9179852 2.9076521]
print('-'*10)
loss = paddle.nn.CTCLoss(blank=0, reduction='sum')(log_probs, labels,
			input_lengths,
			label_lengths,
			size_average=True,
			length_average=False)
print('loss', loss)  #[1.1376063]
loss.backward()
print('grad', log_probs.grad)
print(log_probs.grad.shape)
print(log_probs.shape)
log_probs.clear_gradient()

print('-'*10)
loss = paddle.nn.CTCLoss(blank=0, reduction='sum')(log_probs, labels,
			input_lengths,
			label_lengths,
			size_average=False,
			length_average=True)
print('loss', loss)  #[1.1376063]
loss.backward()
print('grad', log_probs.grad)
print(log_probs.grad.shape)
print(log_probs.shape)
log_probs.clear_gradient()

sys.exit(1)

print('-'*10)
loss = paddle.nn.CTCLoss(blank=0, reduction='sum')(log_probs, labels,
			input_lengths,
			label_lengths,
			norm_by_times=True,
			size_average=False,
			length_average=False)
print('loss', loss)  #[1.1376063]
loss.backward()
print('grad', log_probs.grad)
print(log_probs.grad.shape)
print(log_probs.shape)
log_probs.clear_gradient()
