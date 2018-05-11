# this file is follow the tutorial from https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from collections import OrderedDict, defaultdict
from io import open
import torch
from torch.autograd import Variable
from paras import EOS_token, SOS_token
from logger import Logger


class Text:
    def __init__(self, category):
        self.category = category
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "", 1: "SOS", 2: "EOS"}
        self.n_words = 3
        self.delimeter = ' '
        # number of maximum words in a sentence
        self.max_length = 2
        self.max_sentence = None
        self.len = []
        self.histogram = OrderedDict()
        self.logger = Logger.get_logger()

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 0
            self.index2word[self.n_words] = word
            self.n_words += 1

        self.word2count[word] += 1

    def add_sentence(self, tweet):
        words = tweet.strip().split(self.delimeter)
        self.len.append(len(words))
        if (len(words)+1) > self.max_length:
            self.max_length = len(words) + 1
            self.max_sentence = words

        for word in words:
            self.add_word(word)

    def sentence_len_summary(self):
        self.len = sorted(self.len, key=int)
        curr_thresh = 5
        step_size = 5
        self.histogram[curr_thresh] = 0
        for len in self.len:
            if len > curr_thresh:
                curr_thresh = min(curr_thresh+step_size, self.max_length)
                self.histogram[curr_thresh] = 0
            self.histogram[curr_thresh] += 1

        print(self.histogram)

    def build_phrase_from_indexes(self, indexes):
        result = ""
        for i in range(indexes.size()[0]):
            if indexes[i] == 0:
                continue
            result += self.index2word[indexes[i]] + " "
        return result.strip()

    @staticmethod
    def print_target_and_predicted(content, key_phrases, input, target, predicted):
        logger = Logger.get_logger()
        for i in range(target.size()[1]):
            logger.info(Logger.build_log_message("Text", "print_target_and_predicted",
                                                      "input is: {}".format(content.build_phrase_from_indexes(input[:, i]))))
            logger.info(Logger.build_log_message("Text", "print_target_and_predicted",
                                                 "target phrase: {}".format(key_phrases.build_phrase_from_indexes(target[:, i]))))
            logger.info(Logger.build_log_message("Text", "print_target_and_predicted",
                                                 "predicted phrase: {}".format(key_phrases.build_phrase_from_indexes(predicted[i]))))
            logger.info(Logger.build_log_message("Text", "print_target_and_predicted",
                                             "------------------------------------------------------------------------------------"))

    @staticmethod
    def load_text(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.readlines()
        pairs = []
        for line in data:
            line = line.strip()
            # print("line is: {}".format(line))
            pair = line.split('\t')
            # print("pair is: {}".format(pair))
            pairs.append([pair[0].strip(), pair[1].strip()])
        return pairs

    # =============================================================================================
    # Thanks to PyTorch seq2seq tutorial
    # http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model
    @staticmethod
    def indices_from_sentence(lang, sentence):
        ret = []
        for word in sentence.split(' '):
            if word in lang.word2index:
                ret.append(lang.word2index[word])
        return ret

    @staticmethod
    def sentence_from_indexes(lang, indices):
        result = []
        for i in range(len(indices)):
            if indices[i] == 0 or indices[i] == 1:
                continue
            result.append(lang.index2word[indices[i]])
        return " ".join(result)

    @staticmethod
    def variable_from_sentence(lang, sentence, output=False):
        indexes = Text.indices_from_sentence(lang, sentence)
        indexes.append(EOS_token)

        if output:
            indexes.insert(0, SOS_token)
        # print("Indexs: {}".format(indexes))
        return indexes

    @staticmethod
    def variables_from_pair(pair, input_text, output_text, input_tensor, output_tensor):
        # print("pair: {}".format(pair))
        input_indexes = Text.variable_from_sentence(input_text, pair[0])
        output_indexes = Text.variable_from_sentence(output_text, pair[1], output=True)
        # print("input_indexes: {}".format(input_indexes))
        # print("output_indexes: {}".format(output_indexes))
        # input_idx = Text.sentence_pad_idx(len(input_indexes), len(input_tensor))
        # output_idx = Text.sentence_pad_idx(len(output_indexes), len(output_tensor))
        # print("input_tensor size: {}".format(input_tensor.size()))
        # print("input_indexes length: {}".format(len(input_indexes)))
        input_tensor[0] = len(input_indexes)
        input_tensor[1:len(input_indexes)+1] = torch.LongTensor(input_indexes)
        # print("input_tensor: {}".format(input_tensor[input_idx:input_idx+len(input_indexes)]))
        # output_tensor[output_idx:output_idx+len(output_indexes)] = torch.LongTensor(output_indexes)
        output_tensor[:] = torch.LongTensor(output_indexes)
        # print("output_tensor: {}".format(output_tensor[output_idx:output_idx+len(output_indexes)]))
    # =============================================================================================

    @staticmethod
    def variables_from_pairs(pairs, input_text, output_text, n_gram):
        """
        Creates feature matrix for both inputs and outputs
        Since input are fed into encoder by batch, all tensors are padded with zeros
            at the beginning and at the end
        :param pairs:
        :param input_text:
        :param output_text:
        :return: input tensor(wrapped by Variable), output tensor(wrapped by Variable)
        """
        print("n_gram is {}".format(n_gram))
        input_tensor = torch.LongTensor(len(pairs), input_text.max_length+1).zero_()
        output_tensor = torch.LongTensor(len(pairs), n_gram+2).zero_()
        output_lengths = []
        for i in range(len(pairs)):
            Text.variables_from_pair(pairs[i], input_text, output_text, input_tensor[i], output_tensor[i])

        # input_tensor, _ = torch.sort(input_tensor, dim=0, descending=True)
        # print("input_tensor: {}".format(input_tensor))
        # print(input_tensor.size()[0])
        input_lengths = list(input_tensor[:, 0])
        print("input_lengths: {}".format(input_lengths))

        # transfer to seq_len * batch_size
        input_tensor = torch.transpose(input_tensor[:, 1:], 0, 1)
        output_tensor = torch.transpose(output_tensor, 0, 1)

        input_variable = Variable(input_tensor) #.cuda(gpu_idx) if use_cuda else Variable(input_tensor)
        output_variable = Variable(output_tensor) #.cuda(gpu_idx) if use_cuda else Variable(output_tensor)
        # print("input_variable shape: {}".format(input_variable.size()))
        # print("output_variable shape: {}".format(output_variable.size()))
        return {"input_variable": input_variable, "input_lengths": input_lengths, "output_variable": output_variable, "output_lengths": output_lengths}

    @staticmethod
    def variables_from_sentences(sentences, sentence_indices, max_length):
        input_tensor = torch.LongTensor(len(sentences), max_length + 1).zero_()
        for i, sentence in enumerate(sentences):
            input_tensor[i, 0] = len(sentence_indices[sentence])
            input_tensor[i, 1:len(sentence_indices[sentence]) + 1] = torch.LongTensor(sentence_indices[sentence])
        # print("input_tensor: {}".format(input_tensor))
        input_lengths = list(input_tensor[:, 0])
        # print("input_lengths: {}".format(input_lengths))
        # transform to seq_len * batch_size
        input_tensor = torch.transpose(input_tensor[:, 1:], 0, 1)
        input_variable = Variable(input_tensor)
        return {"input_variable": input_variable, "input_lengths": input_lengths}

    @staticmethod
    def sentence_pad_idx(sentence_length, tensor_length):
        """
        Calculates the start idx for the first word of the sentence in the given tensor
        The tensor pad zeros at both the beginning and the end
        :param sentence_length:
        :param tensor_length:
        :return: the start idx for the first word of the sentence in the given tensor
        """
        gap = tensor_length - sentence_length
        return gap//2

    @staticmethod
    def group_key_phrases_by_content(pairs):
        result = defaultdict(list)
        for pair in pairs:
            result[pair[0]].append(pair[1])
        return result


if __name__ == '__main__':
    tweets = Text()

