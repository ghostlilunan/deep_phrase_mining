
import re
from time import time
import math
import os
import torch
import unicodedata
from datasets import papers_result
from logger import Logger
import subprocess
import shutil
from paras import gpu_idx
from nltk.corpus import stopwords
import pickle


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicodeToAscii(s.lower())
    s = re.sub('[^a-zA-Z0-9 ]+', '', s)
    s = re.sub(' +', ' ', s)
    return s.strip()


def remove_stop_words(sentences, delimeter=' ', stop_words=stopwords.words()):
    ret = []

    for count, sentence in enumerate(sentences):
        sentence = normalize_string(sentence)
        words = sentence.split(delimeter)
        clean_sentence = []
        for word in words:
            if word not in stop_words:
                clean_sentence.append(word)
        ret.append(' '.join(clean_sentence))

        if count % 10000 == 0:
            print("{} sentences processed".format(count))
    return ret


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%d min %d sec' % (m, s)


def batch_loss_report(epoch, batch_idx, n_gram, loss_avg):
    return "In epoch {}; {} grams; the {} batch, average loss: {:.5f}".format(epoch, n_gram, batch_idx, loss_avg)


def progress_summary(start_time, iter_percent, avg_loss):
    curr_time = time()
    elapsed_time = curr_time - start_time
    estimated_time = elapsed_time/iter_percent
    rest_time = estimated_time - elapsed_time

    return 'Elapsed time: {}. Estimated remaining Time:{}\n  Progress: {:.2f}%  epochs finished.\n' \
           'Average loss for the last epoch is: {:.5f}'\
        .format(as_minutes(elapsed_time), as_minutes(rest_time), iter_percent * 100, avg_loss)


def build_result_folder(paras):
    # build model folder if not exists
    create_file(paras['model'])
    print('Model folder created')

    # build fig folder if not exists
    create_file(paras['fig'])
    print('Fig folder created')

    create_file(paras['prediction'])
    print('Prediction folder created')


def save_models(forward_encoder, backward_encoder, decoder, path):
    torch.save(forward_encoder, path + 'forward_encoder.pt')
    torch.save(backward_encoder, path + 'backward_encoder.pt')
    torch.save(decoder, path+'decoder.pt')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename + 'checkpoint.pt')


# src: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L291
def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def optimizer_state_cuda(optimizer):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(gpu_idx)


def get_git_version():
    return subprocess.check_output('git rev-parse --short HEAD', shell=True).decode("utf-8").strip()


def create_file(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


def save_beam_search_predition_result(folder_path, result, score, batch_idx, contents):
    create_file(folder_path)
    stat = {
        'result': result,
        'score': score,
        'content': contents.data.numpy()
    }
    print(folder_path + 'stat_{}.pkl'.format(batch_idx))
    with open(folder_path + 'stat_{}.pkl'.format(batch_idx), 'wb') as f:
        pickle.dump(stat, f)


def save_greedy_predition_result(folder_path, result, batch_idx, contents):
    create_file(folder_path)
    stat = {
        'result': result,
        'content': contents.data.numpy()
    }
    with open(folder_path + 'stat_{}.pkl'.format(batch_idx), 'wb') as f:
        pickle.dump(stat, f)

