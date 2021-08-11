# declarative mode
import numpy as np
import torch
from warpctc_pytorch import CTCLoss

torch.manual_seed(1)
np.random.seed(1)

probs = torch.FloatTensor([[[0.1, 0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1, 0.1]]]).transpose(0, 1).contiguous()
print(probs)
print(probs.shape)
labels = torch.IntTensor([1, 2])
print(labels.shape)
label_sizes = torch.IntTensor([2])
print(label_sizes.shape)
probs_sizes = torch.IntTensor([2])
print(probs_sizes.shape)

probs.requires_grad_(True)  # tells autograd to compute gradients for probs
log_probs = probs 
labels = labels
input_lengths = label_sizes
label_lengths = probs_sizes


print('-'*10, 'all false')
ctc_loss = CTCLoss(blank=0, size_average=False, length_average=False, reduce=True)
cost = ctc_loss(log_probs, labels, input_lengths, label_lengths)
cost.backward()
print('loss', cost)  #[1.1376063]
print(log_probs.grad.detach().numpy())
print(log_probs.grad.shape)
log_probs.grad = None



print('-'*10, 'size average')
ctc_loss = CTCLoss(blank=0, size_average=True, length_average=False, reduce=True)
cost = ctc_loss(log_probs, labels, input_lengths, label_lengths)
cost.backward()
print('loss', cost)  #[1.1376063]
print(log_probs.grad.detach().numpy())
log_probs.grad = None

print('-'*10, 'length_average')
ctc_loss = CTCLoss(blank=0, size_average=False, length_average=True, reduce=True)
cost = ctc_loss(log_probs, labels, input_lengths, label_lengths)
cost.backward()
print('loss', cost)  #[1.1376063]
print(log_probs.grad.detach().numpy())
log_probs.grad = None
