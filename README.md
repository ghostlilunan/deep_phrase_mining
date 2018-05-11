# deep_phrase_mining

# Overview
This is a deep learning project that mines non-exist phrase from text corpus. 
The project is inspired by [Deep Keyphrase Generation](https://github.com/memray/seq2seq-keyphrase)
The project uses a simple encoder-decoder model to do phrase mining.

# Implementation

The codebase is composed of 8 components.  
Detailed documentation and comments for main functions are provided in the code.


src/main: the main entrance of the codebase. call training/testing phase.  
src/models: all the RNN networks for the encoder-decoder model
src/paras: parameters for all RNN networks  
src/train: the training phase  
src/test: the testing phase  
src/evaluate: call in the testing phrase. provide greedy and beam search evaluation  
src/text: summary of text corpus  
src/datasets: file path for input datasets  

train/papers/input: the training data  
test/papers/input:  the testing data  
The original [dataset](https://github.com/memray/seq2seq-keyphrase). The data in this repo is processed.

Due to the size limit of github, the pre-trained model cannot be uploaded.
Let me know if you need it. 

# Results
result/papers/8c3f2b4/prediction/analyzation.txt: contains the prediction of the testing dataset  
The result is not satisfactory. It seems that the title of scientific publication doesn't contain enough information. Thus, it fails to correctly predict most phrases. 

# Usage

To run the code, simply run "python3 main.py"
It takes 24+ hours to run 3000 epochs for the dataset on a TI1080 GPU

# Reference
This project follows the tutorial to build simple RNN networks.  
[Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)