import csv
import operator
from datasets import *
from util import *
import sys
import json
from collections import defaultdict, OrderedDict
from sklearn.model_selection import train_test_split
import random

len_thresh = 25
keywords_thresh = 4
n_gram_thresh = 5
phrase_frequency_thresh = 50


def preprocess_text(file_name, args='title'):
    csv.field_size_limit(sys.maxsize)
    if file_name == 'all_the_news':
        preprocess_all_the_news()

    if file_name == 'papers':
        preprocess_papers(args)

    if file_name == 'tweets':
        preprocess_tweets()


def preprocess_all_the_news():
    publication_lists = ['- The New York Times', '- Breitbart', '(VIDEO)']
    files = load_all_the_news()
    result = []
    sentence_omitted = 0
    for file in files.values():

        print("Preprocess {}".format(file))

        with open(file, 'r') as f:
            csv_reader = csv.reader(f, delimiter=',')
            count = 0
            for line in csv_reader:
                if count != 0:

                    for publication in publication_lists:
                        line[2] = line[2].replace(publication, '')

                    # if line[3] == 'New York Times':
                    #     print(line[2])

                    keyphrase = normalize_string(line[2])
                    content = normalize_string(line[-1])

                    # filter out sentences that has more than len_thresh words
                    if len(content.split(' ')) > len_thresh:
                        sentence_omitted += 1
                        continue

                    # print "keyphrase: {}".format(keyphrase)
                    # print "content: {}".format(content)
                    result.append('{}\t{}'.format(content, keyphrase))

                count += 1
                if count %10000 == 0:
                    print('{} lines processed'.format(count))

    print("{} sentences omitted.".format(sentence_omitted))

    with open(all_the_news_training, 'w') as f:
        f.write('\n'.join(result))


def preprocess_papers(content_key):
    files = load_papers()

    title2idx = dict()
    phrase2count = defaultdict(int)
    phrase2title = defaultdict(list)

    content_keyphrases = defaultdict(list)
    with open(files['raw'], 'r') as f:
        data = json.load(f)
    sentence_omitted = 0
    count = 0

    for line in data:
        content = normalize_string(line[content_key])

        if content == '':
            sentence_omitted += 1
            continue

        title2idx[content] = count

        # remove sentences that are too long
        if len(content.split(' ')) > len_thresh:
            sentence_omitted += 1
            continue

        key_phrases = line['keyword'].split(';')

        if len(key_phrases) < keywords_thresh:
            print(key_phrases)
            sentence_omitted += 1
            continue

        for key_phrase in key_phrases:
            key_phrase = normalize_string(key_phrase)

            if key_phrase != '':
                key_phrase_len = len(key_phrase.split(' '))
                if key_phrase_len > 1:
                    content_keyphrases[content].append(key_phrase)
                    phrase2title[key_phrase].append(content)
                    phrase2count[key_phrase] += 1

        count += 1
        if count % 1000 == 0:
            print("{} lines processed".format(count))

        if count == 30000:
            break

    phrase2count = OrderedDict(sorted(phrase2count.items(), key=operator.itemgetter(1), reverse=True))

    content_keyphrases = select_quality_key_phrases(content_keyphrases, phrase2count, phrase2title)
    training_keys, testing_keys = train_test_split([*content_keyphrases.keys()], test_size=.3)
    training_data = dict(filter(lambda i:i[0] in training_keys, content_keyphrases.items()))
    testing_data = dict(filter(lambda i:i[0] in testing_keys, content_keyphrases.items()))

    # group ground truth key_phrases by length

    paras = papers_result()
    train_file_path = paras['input_train']
    test_file_path = paras['input_test']

    build_data_file(training_data, train_file_path)
    build_data_file(testing_data, test_file_path)


def generate_popular_hashtags():
    paras = load_tweets()

    with open(paras['pure_tweets'], 'r') as f:
        data = f.readlines()

    hashtags = defaultdict(int)
    for line in data:
        words = line.strip().split(' ')

        for word in words:
            if word.startswith('#'):
                hashtags[word] += 1

    hashtags = sorted(hashtags.items(), key=lambda t: t[1])

    prints = []
    for item in hashtags:
        hashtag = item[0]
        occurrence = item[1]

        if occurrence > 1000:
            prints.append(hashtag)

    with open("hashtags.txt", 'w') as f:
        f.write('\n'.join(prints))

    print(hashtags)


def parse_popular_hashtags(popular_hastags):
    ret = {}
    for line in popular_hastags:
        line = line.strip().split(',')
        ret[line[0]] = line[1].split(';')
    return ret


def word_frequencies_list(sentences):
    freq = defaultdict(int)

    for sentence in sentences:
        words = sentence.strip().split(' ')
        for word in words:
            freq[word] += 1
    freq = sorted(freq.items(), key=lambda t: t[1], reverse=True)
    print(freq)
    freq = [words[0] for words in freq if words[1] > 10]

    with open('stop_words.txt', 'w') as f:
        f.write('\n'.join(freq))


def preprocess_tweets():
    paras = tweets_result()

    # uncomment if need to remove stop words from tweets
    # with open(paras['pure_tweets'], 'r') as f:
    #     tweets = f.readlines()
    # tweets = remove_stop_words(tweets)
    # with open(paras['clean_tweets'], 'w') as f:
    #     f.write('\n'.join(tweets))

    with open(paras['popular_hashtags'], 'r') as f:
        hashtag2phrase = f.readlines()

    hashtag2phrases = parse_popular_hashtags(hashtag2phrase)

    with open(paras['clean_tweets'], 'r') as f:
        tweets = f.readlines()

    hashtag2phrase = set(hashtag2phrases.keys())
    tweets2hashtags = defaultdict(list)
    filtered_tweets = remove_stop_words(tweets, stop_words=hashtag2phrase)
    with open(paras['filtered_tweets'], 'w') as f:
        f.write('\n'.join(filtered_tweets))

    res = []
    for i, tweet in enumerate(tweets):
        words = tweet.split(' ')
        for word in words:
            if word in hashtag2phrase:
                res.append(filtered_tweets[i])
                phrases = hashtag2phrases[word]
                for phrase in phrases:
                    tweets2hashtags[filtered_tweets[i]].append(phrase.replace('_', ' ').strip())

        if i % 10000 == 0:
            print("{} tweets processed".format(i))

    res = list(set(res))
    non_empty_tweets = []
    for tweet in res:
        if tweet != '':
            non_empty_tweets.append(tweet)
    # for key in tweets2hashtags:
    #     print(key)

    training_tweets, testing_tweets = train_test_split(non_empty_tweets, test_size=.3)
    training_data = dict(filter(lambda i: i[0] in training_tweets, tweets2hashtags.items()))
    testing_data = dict(filter(lambda i: i[0] in testing_tweets, tweets2hashtags.items()))

    build_data_file(training_data, paras['input_train'])
    build_data_file(testing_data, paras['input_test'])


def build_data_file(content_keyphrases, input_file):
    result = group_one_to_many_key_phrase_by_length(content_keyphrases)
    for key, value in result.items():
        file_path = "{}{}_xgram.txt".format(input_file, key)

        if key > n_gram_thresh:
            continue

        with open(file_path, 'w') as f:
            f.write('\n'.join(value))


def group_one_to_many_key_phrase_by_length(content_keyphrases):
    """
    Split data by phrase_length. Each sub-data is then grouped by content length(Decreasing)
    Designed for PyTorch pack_padded_sequence
    :param content_keyphrases: {content: [key_phrase1, key_phrase2, ...]}
    :return:
    """
    result = dict()
    sorted_result = defaultdict(list)

    for content, key_phrases in content_keyphrases.items():
        content_length = len(content.split(' '))

        for key_phrase in key_phrases:
            key_phrase_length = len(key_phrase.split(' '))
            if key_phrase_length not in result:
                result[key_phrase_length] = defaultdict(list)

            result[key_phrase_length][content_length].append('{}\t{}'.format(content, key_phrase))

    for key_phrase_length in result:
        content_length_dic = result[key_phrase_length]
        content_length_dic = OrderedDict(sorted(content_length_dic.items(), key=operator.itemgetter(0), reverse=True))
        for key, value in content_length_dic.items():
            print("key_phrase_length: {}, content_length: {}".format(key_phrase_length, key))
            sorted_result[key_phrase_length].extend(value)

    return sorted_result


def select_quality_key_phrases(content_keyphrases, phrase2count, phrase2title):
    ret = dict()
    for key, value in phrase2count.items():
        if value >= phrase_frequency_thresh:
            for title in phrase2title[key]:
                if title not in ret:
                    ret[title] = content_keyphrases[title]
                    # print(ret[title])
    return ret


if __name__ == '__main__':
    text = 'tweets'
    preprocess_text(text)

    # file = 'tweets/pure_tweets.txt'
    # with open(file,'r') as f:
    #     data = f.readlines()
    #
    # word_frequencies_list(data)

