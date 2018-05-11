import torch
from torch.autograd import Variable
from paras import *
import random
from time import time
from math import log
from text import Text
import numpy as np

def evaluate(encoder_inputs, encoder_input_lengths, encoder, decoder, content):
    """
    evaluate the model with testing dataset. provide both greedy and beam search
    :param encoder_inputs:
    :param encoder_input_lengths:
    :param encoder:
    :param decoder:
    :return:
    """
    # print()
    # print("******************************************************************************************")
    # print("input_variable: {}".format(encoder_inputs.size()))

    batch_size = len(encoder_input_lengths)
    mask = encoder_inputs != 0
    mask = mask.unsqueeze(2).type(torch.FloatTensor)

    # print("batch_size: {}".format(batch_size))
    encoder_hidden = encoder.init_hidden(batch_size, bidirectional=bidirectional)
    # print("encoder_hidden shape: {}".format(encoder_hidden.size()))

    if use_cuda:
        encoder_inputs = encoder_inputs.cuda(gpu_idx, async=True)
        mask = mask.cuda(gpu_idx, async=True)

    # bi-direction Encoder
    # forward direction
    # encoder_start = time()
    encoder_outputs, decoder_hidden = encoder(encoder_inputs, encoder_hidden, encoder_input_lengths)
    mask_length = encoder_outputs.size()[0]
    # encoder_outputs = torch.transpose(encoder_outputs, 0, 1)
    # encoder_end = time()
    # print("encoder takes: {} seconds".format(encoder_end - encoder_start))
    # print("encoder_outputs shape: {}".format(encoder_outputs.size()))
    # print("encoder_outputs: {}".format(encoder_outputs))
    #
    # print("decoder_hidden shape: {}".format(decoder_hidden.size()))
    # print("decoder_hidden: {}".format(decoder_hidden))

    # decoder_predicted = decode_greedy(decoder, decoder_hidden, encoder_outputs, mask, mask_length, 3, batch_size)
    #
    # return decoder_predicted

    results = []
    scores = []

    for i in range(batch_size):
        start = time()
        result, score = decode_with_beam_search(encoder_outputs[:, i:i+1, :], decoder_hidden[:, i:i+1, :], beam_size, decoder, mask[0:mask_length, i:i+1])
        print("Finish the {} sample, took {} seconds".format(i, time()-start))
        # print(result)
        # print(scores)
        results.append(result)
        scores.append(score)

    return np.asarray(results), np.asarray(scores)


def decode_with_beam_search(encoder_outputs, decoder_hidden, beam_size, decoder, mask, max_phrase_length=5):
    # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("encoder_outputs shape: {}".format(encoder_outputs.size()))
    # print("decoder_hidden shape: {}".format(decoder_hidden.size()))
    decoder_input = Variable(torch.LongTensor([SOS_token]))
    result = torch.LongTensor([[SOS_token]])
    score = torch.FloatTensor([[0]])
    batch_size = decoder_input.size()[0]

    phrase_scores = []
    phrase_indices = []

    if use_cuda:
        decoder_input = decoder_input.cuda(gpu_idx, async=True)
        result = result.cuda(gpu_idx, async=True)
        score = score.cuda(gpu_idx, async=True)

    count = 0
    while count < max_phrase_length:
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("decoder_input shape: {}".format(decoder_input.size()))
        # print("decoder_input: {}".format(decoder_input))
        #
        # print("result shape: {}".format(result.size()))
        # print("score shape: {}".format(score.size()))
        # print("mask shape: {}".format(mask.size()))

        # (batch_size * beam_size) * count
        result = result.repeat(beam_size, 1)
        # print("After repeat, result shape: {}".format(result.size()))
        # print("After repeat, result: {}".format(result))

        score = score.repeat(beam_size, 1)
        # print("After repeat, score shape: {}".format(score.size()))
        # print("After repeat, score: {}".format(score))

        decoder_outputs, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, batch_size, mask)
        # print("decoder_outputs shape: {}".format(decoder_outputs.size()))
        # print("decoder_hidden shape: {}".format(decoder_hidden.size()))

        # batch_size * k
        top_value, top_index = decoder_outputs.data.topk(beam_size)
        # print("top_value shape: {}".format(top_value.size()))
        # print("top_index shape: {}".format(top_index.size()))
        # print("top_index: {}".format(top_index))

        # convert to (batch_size * k) - 1D vector
        top_value = torch.transpose(top_value, 0, 1).contiguous().view(-1)
        top_index = torch.transpose(top_index, 0, 1).contiguous().view(-1)
        # print("After transpose, top_value shape: {}".format(top_value.size()))
        # print("After transpose, top_index shape: {}".format(top_index.size()))
        # print("After transpose, top_index: {}".format(top_index))

        non_eos_index = (top_index != EOS_token).nonzero().view(-1)
        # print("non_eos_index shape: {}".format(non_eos_index.size()))
        # print("non_eos_index: {}".format(non_eos_index))

        # remove eos_indices from both value and indexes
        non_eos_top_value = torch.index_select(top_value, 0, non_eos_index).view(-1, 1)
        non_eos_top_index = torch.index_select(top_index, 0, non_eos_index).view(-1, 1)
        # print("non_eos_top_value shape: {}".format(non_eos_top_value.size()))
        # print("non_eos_top_value: {}".format(non_eos_top_value))

        # print("non_eos_top_index shape: {}".format(non_eos_top_index.size()))

        # only execute when there are eos_tokens
        if non_eos_index.size() != top_index.size():
            eos_index = (top_index == EOS_token).nonzero().view(-1)
            # print("eos_index shape: {}".format(eos_index.size()))
            # print("eos_index: {}".format(eos_index))
            # remove phrases that reach the eos_token at this step
            # add score and phrases to ret list
            phrase_indices.extend(torch.index_select(result, dim=0, index=eos_index))
            phrase_scores.extend(list(torch.index_select(score, dim=0, index=eos_index)))

        result = torch.index_select(result, dim=0, index=non_eos_index)
        score = torch.index_select(score, dim=0, index=non_eos_index)
        # print("After remove eos_token, result shape: {}".format(result.size()))
        # print("After remove eos_token, result: {}".format(result))
        # print("After remove eos_token, score shape: {}".format(score.size()))
        # print("After remove eos_token, score: {}".format(score))

        result = torch.cat((result, non_eos_top_index), dim=1)
        # print("After cat, result shape: {}".format(result.size()))
        # print("After cat, result: {}".format(result))
        score = torch.cat((score, non_eos_top_value), dim=1)
        # print("After cat, score shape: {}".format(score.size()))
        count += 1
        decoder_input = Variable(result[:, count])
        batch_size = decoder_input.size()[0]
        # print("batch_size: {}".format(batch_size))
        # print("count: {}".format(count))

        decoder_hidden = decoder_hidden.repeat(1, beam_size, 1)
        # print("After repeat, decoder_hidden shape: {}".format(decoder_hidden.size()))
        encoder_outputs = encoder_outputs.repeat(1, beam_size, 1)
        # print("After repeat, encoder_outputs shape: {}".format(encoder_outputs.size()))
        mask = mask.repeat(1, beam_size, 1)

        non_eos_index = Variable(non_eos_index)
        # print("decoder_hidden: {}".format(decoder_hidden))
        # print("non_eos_index: {}".format(non_eos_index))

        decoder_hidden = torch.index_select(decoder_hidden, 1, non_eos_index)
        # print("After filtering eos, decoder_hidden shape: {}".format(decoder_hidden.shape()))
        encoder_outputs = torch.index_select(encoder_outputs, 1, non_eos_index)
        # print("After filtering eos, encoder_outputs shape: {}".format(encoder_outputs.shape()))
        mask = torch.index_select(mask, 1, non_eos_index)

    phrase_indices = np.asarray(list(map(np.asarray, phrase_indices)))
    phrase_scores = np.asarray(list(map(np.asarray, phrase_scores)))

    return phrase_indices, phrase_scores


def decode_greedy(decoder, decoder_hidden, encoder_outputs, mask, mask_length, target_length, batch_size):
    decoder_inputs = torch.ones(batch_size).type(torch.LongTensor)

    if use_cuda:
        decoder_inputs = Variable(decoder_inputs.pin_memory()).cuda(gpu_idx, async=True)
    else:
        decoder_inputs = Variable(decoder_inputs)

    decoder_predicted = torch.LongTensor(batch_size, target_length)
    for i in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_inputs, decoder_hidden, encoder_outputs,
                                                 batch_size, mask[0:mask_length, :])
        topv, topi = decoder_output.data.topk(1)
        decoder_predicted[:, i] = topi[:, 0]
        decoder_inputs = Variable(topi[:, 0])
        # print("decoder_output: {}".format(decoder_output))
        # print("decoder_inputs[{}, :]".format(i, decoder_inputs[i, :]))
    return decoder_predicted.cpu().numpy()
