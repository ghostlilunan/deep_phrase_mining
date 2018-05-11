import torch
from torch.autograd import Variable
from paras import EOS_token, use_cuda, teacher_forcing_ratio, gpu_idx, bidirectional
import random
from time import time


def train(encoder_inputs, encoder_input_lengths, decoder_inputs, decoder_input_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    """
    train the whole dataset for one epoch
    :param encoder_inputs: sentence_length * batch_size
    :param decoder_inputs:  sentence_length * batch_size
    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion:
    :return:
    """
    # print()
    # print("******************************************************************************************")
    # print("input_variable: {}".format(encoder_inputs.size()))
    # print("target_variable: {}".format(decoder_inputs.size()))

    target_length = decoder_inputs.size()[0]-1
    batch_size = len(encoder_input_lengths)
    mask = encoder_inputs != 0
    mask = mask.unsqueeze(2).type(torch.FloatTensor)
    # print("mask: {}".format(mask))
    #
    # print("batch_size: {}".format(batch_size))
    encoder_hidden = encoder.init_hidden(batch_size, bidirectional=bidirectional)
    # print("encoder_hidden shape: {}".format(encoder_hidden.size()))

    if use_cuda:
        encoder_inputs = encoder_inputs.cuda(gpu_idx, async=True)
        decoder_inputs = decoder_inputs.cuda(gpu_idx, async=True)
        mask = mask.cuda(gpu_idx, async=True)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    # bi-direction Encoder
    # forward direction
    # encoder_start = time()
    encoder_outputs, decoder_hidden = encoder(encoder_inputs, encoder_hidden, encoder_input_lengths)
    # encoder_outputs = encoder_outputs
    # encoder_outputs = torch.transpose(encoder_outputs, 0, 1)
    # encoder_end = time()
    # print("encoder takes: {} seconds".format(encoder_end - encoder_start))
    # print("encoder_outputs shape: {}".format(encoder_outputs.size()))
    # print("encoder_outputs: {}".format(encoder_outputs))
    #
    # print("decoder_hidden shape: {}".format(decoder_hidden.size()))
    # print("decoder_hidden: {}".format(decoder_hidden))

    if use_cuda:
        # encoder_outputs = encoder_outputs.cuda(gpu_idx, async=True)
        # decoder_hidden = decoder_hidden.cuda(gpu_idx, async=True)
        decoder_inputs = decoder_inputs.cuda(gpu_idx, async=True)

    mask_length = encoder_outputs.size()[0]

    decoder_predicted = torch.LongTensor(batch_size, target_length)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # decoder_start = time()
    if True:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_inputs[i, :], decoder_hidden, encoder_outputs,
                                                                        batch_size, mask[0:mask_length, :])
            topv, topi = decoder_output.data.topk(1)
            decoder_predicted[:, i] = topi[:, 0]
            # print("decoder_output: {}".format(decoder_output))
            # print("decoder_inputs[{}, :]".format(i, decoder_inputs[i, :]))
            loss += criterion(decoder_output, decoder_inputs[i+1, :])

    else:
        # Without teacher forcing: use its own predictions as the next input
        # weights = torch.ones(target_variable.size()[0]).type(torch.LongTensor).pin_memory()
        # weights = weights.cuda(gpu_idx, async=True) if use_cuda else weights
        # print("weights: {}".format(weights))
        for i in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        batch_size)
            # print("decoder output: {}", decoder_output)
            topv, topi = decoder_output.data.topk(1)
            # print("topv: {}".format(topv))
            # print("topi: {}".format(topi))

            # ni = topi[:, 0]
            decoder_input = topi[:, 0]
            decoder_predicted[:, i] = topi[:, 0]
            # print(type(weights))
            # weights = Variable(weights)
            # print("After EOS check, weights: {}".format(weights))
            # print(weights)
            # weights = Variable(weights)
            # weights = weights.cuda(gpu_idx) if use_cuda else weights
            # print("After EOS check, weights: {}".format(weights))
            # print("decoder_input: {}".format(decoder_input))
            # print("target_variable[:,i]: {}".format(target_variable[:, i]))
            # print(type(target_variable[:, i]))
            # target_variable.data[:, i] = target_variable.data[:, i] * weights
            # weights[decoder_input == EOS_token] = 0
            decoder_input = Variable(decoder_input).cuda(gpu_idx) if use_cuda else Variable(decoder_input)
            # print("After times weights, target_variable[:,i]: {}".format(target_variable[:, i]))
            loss += criterion(decoder_output, decoder_inputs[:, i])
            # if (topi[:, 0] == EOS_token).all():
            #     break
    # decoder_end = time()
    # print("decoder takes: {} seconds".format(decoder_end - decoder_start))

    # bp_start = time()
    loss.backward()
    # bp_end = time()
    # print("bp takes: {} seconds".format(bp_end - bp_start))
    # print("******************************************************************************************")
    # print()

    # para_start = time()
    encoder_optimizer.step()
    decoder_optimizer.step()
    # para_end = time()
    # print("adjust weight takes: {} seconds".format(para_end - para_start))
    # x = input("Check GPU usage")
    return loss.data[0] / target_length, decoder_predicted
