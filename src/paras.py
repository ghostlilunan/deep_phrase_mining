import torch

SOS_token = 1
EOS_token = 2
use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5
hidden_size = 256
gpu_idx = 6
pad_idx = 0
batch_size = 2048
num_layers = 3
learning_rate = 0.0001
bidirectional = True
beam_size = 5
