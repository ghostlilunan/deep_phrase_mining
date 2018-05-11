import operator
from collections import defaultdict, OrderedDict
from models import RnnEncoder, AttnDecoder
from paras import *
from text import Text
from datasets import *
from time import time
from torch import optim
import torch.nn as nn
from train import train
from util import *
from plot import nll_loss_plot
from logger import Logger
import pickle
from evaluate import evaluate
from analyzation import analyze_greedy_prediction, analyze_beam_search_prediction


def preprocess_training_data(file_name, input_text, output_text):
    # print("Start preparing content-keyphrases pairs for {}".format(file_name))
    pairs = Text.load_text(file_name)
    # print("Finish preparing content-keyphrases pairs. Read {} pairs" .format(len(pairs)))

    # print("Start counting words for each Text")
    for pair in pairs:
        input_text.add_sentence(pair[0])
        output_text.add_sentence(pair[1])
    return pairs


def build_texts(input_folder, n_grams):
    print("Start building Text for content and key-phrases")
    input_pairs = dict()
    content = Text('content')
    key_phrases = Text('key_phrases')
    for n_gram in n_grams:
        file_path = input_folder + "{}_gram.txt".format(n_gram)
        print("file_path: {}".format(file_path))
        pairs = preprocess_training_data(file_path, content, key_phrases)
        print("Generate {} pair of inputs".format(len(pairs)))
        input_pairs[n_gram] = pairs
        # print("content max length sentence: {}".format(content.max_sentence))
        # print("key_phrases max length sentence: {}".format(key_phrases.max_sentence))
    print("Content has {} words".format(content.n_words))
    print("KeyPhrases has {} words".format(key_phrases.n_words))
    print("Content Text sentence length distribution: {}".format(content.sentence_len_summary()))
    return content, key_phrases, input_pairs


def build_training_data(content, key_phrases, input_pairs):
    result = dict()
    for key, value in input_pairs.items():
        result[key] = Text.variables_from_pairs(value, content, key_phrases, key)
    return result


def preprocess_testing_data(file_name, accumulated_pairs, contents_length, content_text, content_indices):
    """
    Accumulate content and key_phrases pairs and measure content length
    :param file_name:
    :param accumulated_pairs: {content: [key_phrase1, key_phrase2, ...]}
    :param contents_length: {content: content_length}
    :return:
    """
    pairs = Text.load_text(file_name)
    for pair in pairs:
        if pair[0] not in content_indices:
            content_indices[pair[0]] = Text.indices_from_sentence(content_text, pair[0])
            contents_length[pair[0]] = len(content_indices[pair[0]])
        accumulated_pairs[pair[0]].append(pair[1])


def build_testing_data(content, input_folder, n_grams):
    print("Building testing data...")
    pairs = defaultdict(list)
    contents_length = dict()
    contents_indices = dict()
    for n_gram in n_grams:
        file_path = input_folder + "{}_gram.txt".format(n_gram)
        print("file_path: {}".format(file_path))
        preprocess_testing_data(file_path, pairs, contents_length, content, contents_indices)

    sorted_pairs = OrderedDict()
    sorted_length = sorted(contents_length.items(), key=operator.itemgetter(1), reverse=True)
    contents_length = [x[0] for x in sorted_length]

    for key in contents_length:
        sorted_pairs[key] = pairs[key]

    # for key, value in pairs.items():
    #     print("{}, {}".format(key, value))
    max_length = sorted_length[0][1]
    print("max_length is {}".format(max_length))
    return Text.variables_from_sentences(list(sorted_pairs.keys()), contents_indices, max_length), sorted_pairs


def train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, pairs, content_text, key_phrases_text, model_folder):
    logger = Logger.get_logger()

    # init parameter
    epochs = 3000

    print('Start training')
    start_time = time()
    epoch_losses = []
    batch_losses = defaultdict(list)


    # prepare criterion
    criterion = nn.NLLLoss(ignore_index=0)

    for epoch in range(1, epochs+1):
        print("{} epoch starts".format(epoch))
        epoch_loss = 0
        total_pairs = 0
        batch_idx = 0
        for key, value in pairs.items():
            contents = value['input_variable']
            contents_lengths = value['input_lengths']

            key_phrases = value['output_variable']
            key_phrases_lengths = value['output_lengths']

            if epoch == 1:
                print("For {}_gram".format(key))
                print("contents shape: {}".format(contents.size()))
                print("key_phrases shape: {}".format(key_phrases.size()))

            num_pairs = contents.size()[1]
            print("Start training on {}_gram key_phrases. {} pairs".format(key, num_pairs))
            idx = 0
            while idx < num_pairs:
                batch_idx += 1
                end_idx = min(idx+batch_size, num_pairs)
                total_pairs += end_idx - idx
                loss, predicted_phrases_idx = train(contents[:, idx:end_idx].clone(), contents_lengths[idx:end_idx],
                                                    key_phrases[:, idx:end_idx].clone(), key_phrases_lengths,
                                                    encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
                # batch_loss = loss
                epoch_loss += loss

                batch_losses[key].append(loss)
                print(batch_loss_report(epoch, batch_idx, key, loss))

                if epoch % 20 == 0:
                    logger.info(Logger.build_log_message("main", "train_iters",
                                                         "In epoch: {}, batch: {}".format(epoch, batch_idx)))
                    logger.info(Logger.build_log_message("main", "train_iters", ""))
                    Text.print_target_and_predicted(content_text, key_phrases_text, contents.data[:, idx:end_idx], key_phrases.data[:, idx:end_idx], predicted_phrases_idx)
                    logger.info(Logger.build_log_message("main", "train_iters",
                                                         "In epoch: {}, batch: {}".format(epoch, batch_idx)))
                    logger.info(Logger.build_log_message("main", "train_iters", ""))

                idx = end_idx

        epoch_avg_loss = epoch_loss/batch_idx
        epoch_losses.append(epoch_avg_loss)
        print(progress_summary(start_time, epoch/ epochs, epoch_avg_loss))

        if epoch % 100 == 0:
            save_checkpoint({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': epoch_avg_loss
            }, model_folder)

            with open(model_folder + "losses.pkl", 'wb') as f:
                pickle.dump({
                    "epoch_losses": epoch_losses,
                    "batch_losses": batch_losses,
                    "epoch": epoch,
                }, f)

    print('Finish training')
    print('Total time: {}'.format(as_minutes(time() - start_time)))
    return epoch_losses, batch_losses


def evaluate_mini_batch(encoder, decoder, inputs, result_folder, content_text):
    """

    :param encoder:
    :param decoder:
    :param inputs: this is a dictionary contains input and phrases information
    :param result_folder:
    :param content_text: text for input corpus
    :return:
    """
    logger = Logger.get_logger()

    contents = inputs['input_variable']
    contents_lengths = inputs['input_lengths']
    idx = 0
    num_pairs = contents.size()[1]
    print("contents size: {}".format(contents.size()))
    print("num_pairs: {}".format(num_pairs))
    batch_idx = 0
    total_pairs = 0

    while idx < num_pairs:
        batch_idx += 1
        end_idx = min(idx + batch_size, num_pairs)
        total_pairs += end_idx - idx

        # results = evaluate(contents[:, idx:end_idx].clone(), contents_lengths[idx:end_idx], encoder, decoder, content_text)
        # save_greedy_predition_result(result_folder, results, batch_idx, contents[:, idx:end_idx])

        results, scores = evaluate(contents[:, idx:end_idx].clone(), contents_lengths[idx:end_idx], encoder, decoder, content_text)
        save_beam_search_predition_result(result_folder, results, scores, batch_idx, contents[:, idx:end_idx])

        idx = end_idx


def prepare_models(content, key_phrases, load_module, load_module_path):
    """

    :param content: text for input
    :param key_phrases: text for phrases
    :param load_module: whether load previous module
    :param load_module_path: path of previous module
    :return:
    """
    # Init encoder and decoder
    # Decoder doubles the hidden size because we use bidirectional rnn for encoder
    # and concatenates the output from forward and backward directions
    print('Preparing encoder and decoder')
    encoder = RnnEncoder(content.n_words, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
    decoder = AttnDecoder(key_phrases.n_words, hidden_size * 2)
    check_point = None

    # prepare models
    if load_module:
        check_point = torch.load(load_module_path+'checkpoint.pt', map_location=lambda storage, loc: storage)
        encoder.load_state_dict(check_point['encoder'])
        decoder.load_state_dict(check_point['decoder'])
    if use_cuda:
        encoder = encoder.cuda(gpu_idx)
        decoder = decoder.cuda(gpu_idx)

    # prepare optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    if load_module:
        encoder_optimizer.load_state_dict(check_point['encoder_optimizer'])
        decoder_optimizer.load_state_dict(check_point['decoder_optimizer'])
        adjust_learning_rate(encoder_optimizer, learning_rate)
        adjust_learning_rate(decoder_optimizer, learning_rate)
        if use_cuda:
            optimizer_state_cuda(encoder_optimizer)
            optimizer_state_cuda(decoder_optimizer)
    return decoder, decoder_optimizer, encoder, encoder_optimizer


def run(paras, load_module=False, training=True, testing=True):
    build_result_folder(paras)

    train_folder = paras['input_train']
    test_folder = paras['input_test']
    n_grams = [2, 3, 4, 5]

    # training phrase
    content, key_phrases, training_input_pairs = build_texts(train_folder, n_grams)
    decoder, decoder_optimizer, encoder, encoder_optimizer = prepare_models(content, key_phrases, load_module, paras['model'])

    if training:
        training_input_pairs = build_training_data(content, key_phrases, training_input_pairs)
        epoch_losses, batch_losses = train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, training_input_pairs, content, key_phrases, paras['model'])

    # testing phrase
    if testing:
        testing_inputs, testing_input_pairs = build_testing_data(content, test_folder, n_grams)
        print("Start testing...")
        evaluate_mini_batch(encoder, decoder, testing_inputs, paras['prediction'], content)
        analyze_beam_search_prediction(paras['prediction'], key_phrases, testing_input_pairs)


if __name__ == '__main__':
    git_version = get_git_version()
    # git_version = "8c3f2b4"
    paras = papers_result(git_version)
    print("logger address: {}".format(paras['accuracy_log']))
    create_file(paras['result_folder'])
    logger = Logger(paras['accuracy_log'])
    run(paras, load_module=False, training=True, testing=True)
