import os
import pickle
from collections import defaultdict
from text import Text
import numpy as np
import operator


def analyze_greedy_prediction(result_folder, key_phrases_text, pairs):
    """

    :param result_folder:
    :param key_phrases_text:
    :param pairs: the input phrase pair
    :return:
    """
    sentences = list(pairs.keys())
    with open(result_folder + 'analyzation.txt', 'w') as fout:
        for filename in os.listdir(result_folder):
            if filename.endswith("6.pkl"):
                phrases_summary = []
                with open(result_folder + filename, 'rb') as f:
                    data = pickle.load(f)

                contents = np.transpose(data['content'])
                results = data['result']
                for i in range(len(contents)):
                    content = sentences[i]
                    # print("content: {}".format(content))
                    phrase = build_key_phrase(key_phrases_text, results[i])
                    ground_truth_phrases = pairs[content]
                    # print("ground_truth_phrases")
                    phrases_summary.append(
                        build_greedy_sample_analyzation(content, phrase, ground_truth_phrases))
                fout.write('\n'.join(phrases_summary))


def recall_statistics(recall_summary):
    stat = np.zeros(11)

    for val in recall_summary:
        bin_index = int(val * 10)
        stat[bin_index] += 1
    return stat


def analyze_beam_search_prediction(result_folder, key_phrases_text, pairs):
    """

    :param result_folder:
    :param key_phrases_text:
    :param pairs: input and phrase pair
    :return:
    """
    # k is the bin size of beam search
    k = 5
    sentences = list(pairs.keys())
    recall_summary = []
    precdicted_pharse_summary = defaultdict(list)
    with open(result_folder+'analyzation.txt', 'w') as fout:
        for filename in os.listdir(result_folder):
            if filename.endswith(".pkl"):
                phrases_summary = []
                with open(result_folder + filename, 'rb') as f:
                    data = pickle.load(f)

                contents = np.transpose(data['content'])
                print("contents shape: {}".format(contents.shape))
                phrases = extract_predicted_phrases(data['result'], key_phrases_text)
                results = data['result']
                phrase_scores = calculate_score(data['score'])
                scores = data['score']
                # print(len(phrases))
                # print(len(phrase_scores))
                # for i in range(3):
                for i in range(len(contents)):
                    content = sentences[i]
                    # print("content: {}".format(content))
                    top_k_phrases, top_k_scores = select_top_k_phrases(k, phrase_scores[i], phrases[i], results[i])
                    ground_truth_phrases = pairs[content]
                    # print("ground_truth_phrases")
                    phrases_summary.append(build_beam_search_sample_analyzation(content, top_k_phrases, top_k_scores, ground_truth_phrases))
                    recall_score = calculate_beam_search_recall(top_k_phrases, ground_truth_phrases)

                    recall_summary.append(recall_score)

                    if recall_score >= 0.0:
                        group_tweets_by_predicted_phrase(precdicted_pharse_summary, top_k_phrases, top_k_scores,
                                                     content)
                fout.write('\n'.join(phrases_summary))

    stat = recall_statistics(recall_summary)
    print("recall: {}".format(stat))
    # print(precdicted_pharse_summary)
    for key, val in precdicted_pharse_summary.items():
        precdicted_pharse_summary[key] = sorted(val, key=operator.itemgetter(1), reverse=True)
        # print(precdicted_pharse_summary[key][:10])
        phrases_top_10 = [x[0] for x in precdicted_pharse_summary[key][:50]]
        with open('paper_phrases/{}.txt'.format(key), 'w') as f:
            f.write('\n'.join(phrases_top_10))


def group_tweets_by_predicted_phrase(precdicted_pharse_summary, top_k_phrases, top_k_scores, tweet):
    for i, phrase in enumerate(top_k_phrases):
        precdicted_pharse_summary[phrase].append((tweet, top_k_scores[i]))


def build_key_phrase(text, phrase_indices):
    return Text.sentence_from_indexes(text, phrase_indices)


def calculate_score(score):
    phrase_score = []
    for i in range(len(score)):
        phrase_score.append([])
        for j in range(len(score[i])):
            phrase_score[i].append(sum(score[i][j]))
    return phrase_score


def extract_predicted_phrases(result, content):
    phrases = []
    for i in range(len(result)):
        phrases.append([])
        for j in range(len(result[i])):
            phrases[i].append(Text.sentence_from_indexes(content, result[i][j]))
    return phrases


def select_top_k_phrases(k, phrase_scores, phrases, results):
    score_rank = np.argsort(phrase_scores)[::-1]
    top_k_scores = []
    ret_phrases = []

    if len(phrases) != len(phrase_scores):
        print("phrases/score length mismatch")

    # print("top k of score_rank: {}".format(score_rank[:k]))
    count = 0
    for i in range(len(score_rank)):

        # skip prediction with only eos/sos token
        if phrases[score_rank[i]] == '':
            continue

        ret_phrases.append(phrases[score_rank[i]])
        top_k_scores.append(phrase_scores[score_rank[i]])

        # print("phrase: {}".format(phrases[score_rank[count]]))
        # print("phrase scores: {}".format(phrase_scores[score_rank[count]]))
        # print("result: {}".format(results[score_rank[count]]))
        count += 1

        if count == k:
            break

    return ret_phrases, top_k_scores


def build_greedy_sample_analyzation(content, predicted_phrase, ground_truth_phrases):

    ret = "Content:\n{}\n".format(content)
    ret += "=======================================================================\n"
    ret += "Ground truths:\n"
    for phrase in ground_truth_phrases:
        ret += '{}\n'.format(phrase)
    ret += "=======================================================================\n"
    ret += "Predictions:\n"
    ret += '{}\n'.format(predicted_phrase)
    return ret


def build_beam_search_sample_analyzation(content, top_k_phrases, top_k_scores, ground_truth_phrases):

    ret = "Content:\n{}\n".format(content)
    ret += "=======================================================================\n"
    ret += "Ground truths:\n"
    for phrase in ground_truth_phrases:
        ret += '{}\n'.format(phrase)
    ret += "=======================================================================\n"
    ret += "Predictions:\n"
    for phrase, score in zip(top_k_phrases, top_k_scores):
        ret += '{}, {}\n'.format(phrase, score)

    return ret


def calculate_beam_search_recall(predicted_phrases, ground_truth_phrases):
    ground_truth = set(ground_truth_phrases)
    ground_truth_len = len(ground_truth)
    # print(ground_truth)
    # print(predicted_phrases)
    relevant_phrase = 0
    for phrase in predicted_phrases:
        if phrase in ground_truth:
            relevant_phrase += 1

    return relevant_phrase/ground_truth_len

