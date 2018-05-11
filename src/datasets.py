def load_papers():

    # set this path to the original dataset
    folder_path = ''
    return {'raw': folder_path + 'all_title_abstract_keyword_clean.json'}


def papers_result(git_cv='1'):
    paras = dict()

    paras['train'] = '/Users/Lunan/deep_phrase_mining/train/papers/'
    paras['test'] = '/Users/Lunan/deep_phrase_mining/test/papers/'
    paras['result_folder'] = '/Users/Lunan/deep_phrase_mining/result/papers/' + git_cv + '/'

    paras['input_train'] = paras['train'] + 'input/'
    paras['input_test'] = paras['test'] + 'input/'
    paras['model'] = paras['result_folder'] + 'model/'
    paras['fig'] = paras['result_folder'] + 'fig/'
    paras['prediction'] = paras['result_folder'] + 'prediction/'
    paras['accuracy_log'] = paras['result_folder'] + 'accuracy.log'
    return paras
