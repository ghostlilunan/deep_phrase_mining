# Special thanks to the great seq2seq yTorch tutorial
# http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
from paras import use_cuda, gpu_idx
from time import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, gru_drop_out=0.1):
        """

        :param input_size: batch_size * seq_len
        :param hidden_size: size of each hidden state
        :param batch_size: number of examples in one batch
        :param num_layers: number of layers for a GRU
        """
        super(RnnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=0, sparse=False)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=gru_drop_out, bidirectional=self.bidirectional)

    def forward(self, inputs, hidden, input_lengths):
        """

        :param inputs: seq_len * batch_size * hidden_size
        :param hidden: (num_layer * num_direction) * batch_size * hidden_size
        :return: output: seq_len * batch_size * embedding_size
                 last_hidden: 1 * batch_size * hidden_size
        """
        # print("inputs shape: {} ".format(inputs.size()))
        # print("hidden shape: {}".format(hidden.size()))
        # start = time()
        embedded = self.embedding(inputs)
        packed_input = pack_padded_sequence(embedded, input_lengths)

        # print("embedded took: {} seconds".format(time()-start))
        # start = time()
        output, hidden = self.gru(packed_input, hidden)
        # print("output: {}".format(output))
        output, output_batch_size = pad_packed_sequence(output)

        if self.bidirectional:
            return output, torch.cat([hidden[-2], hidden[-1]], dim=1).unsqueeze(0)

        # print("gru took: {} seconds".format(time()-start))
        return output, hidden[-1].unsqueeze(0)

    def init_hidden(self, batch_size, bidirectional=False):
        result = None
        if bidirectional:
            result = init.kaiming_normal(torch.zeros(self.num_layers*2, batch_size, self.hidden_size))
        else:
            result = init.kaiming_normal(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        if use_cuda:
            return Variable(result.pin_memory()).cuda(gpu_idx, async=True)
        else:
            return Variable(result)


class AttnDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout_p=0.1):
        """
        RNN Decoder with  attention
        :param hidden_size: hidden layer size
        :param output_size: output size
        :param dropout_p: dropout rate for all units
        :param max_length: the maximum length of encoder output vectors
        """
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size, sparse=False)
        self.embedding_hidden_attn = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)
        self.attn_weights = nn.Linear(self.hidden_size*2, 1)
        self.attn_weights_softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, batch_size, mask):
        """

        :param input: seq_len * batch_size
        :param hidden: (num_layer * num_direction) * batch_size * hidden_size
        :param encoder_outputs: seq_length * batch_size * embedding_size(encoder)
        :return:
        """
        # print("=================================================================")
        # print("input size: {}".format(input.size()))
        # print("encoder_outputs size: {}".format(encoder_outputs.size()))
        # print("hidden size: {}".format(hidden.size()))
        # embedded: seq_len * batch_size * hidden
        embedded = self.embedding(input)

        # print("embedded size: {}".format(embedded.size()))
        embedded = self.dropout(embedded)
        # print("After dropout, embedded size: {}".format(embedded.size()))

        # embedding_hidden: seq_len * batch_size * hidden
        embedding_hidden = self.embedding_hidden_attn(embedded, hidden[0]).view(1, batch_size, -1)
        embedding_hidden = embedding_hidden.repeat(encoder_outputs.size()[0], 1, 1)
        # print("embedding_hidden size: {}".format(embedding_hidden.size()))

        # attn_weights -> seq_len * batch_size * 1
        # print("embedding_hidden: {}".format(embedding_hidden))
        # print("encoder_outputs: {}".format(encoder_outputs))
        # print("mask: {}".format(mask))
        attn_weights = self.attn_weights(torch.cat((embedding_hidden, encoder_outputs), dim=2))
        attn_weights *= mask
        # print("attn_weights size: {}".format(attn_weights.size()))
        attn_weights = self.attn_weights_softmax(attn_weights)
        # print("After softmax, attn_weights size: {}".format(attn_weights.size()))

        # inputs_with_attn: 1 * batch_size * hidden_size
        inputs_with_attn = torch.mean(attn_weights * encoder_outputs, dim=0, keepdim=True)
        # print("inputs_with_attn: {}".format(inputs_with_attn.size()))
        inputs_with_attn = F.relu(inputs_with_attn)
        output, hidden = self.gru(inputs_with_attn, hidden)
        # print("After gru: output size: {}".format(output.size()))
        # print("After gru: hidden size: {}".format(hidden.size()))
        # print("=================================================================")
        # print()
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def init_hidden(self, batch_size):
        result = init.kaiming_normal(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            return Variable(result.pin_memory()).cuda(gpu_idx, async=True)
        else:
            return Variable(result)
