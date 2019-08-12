import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    """
    Train HMM based on training data

    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags

    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    default = 1e-06
    words = []
    word_index_map = {}
    
    S = len(tags)
    L = len(train_data)
    state_dict = {tags[i]:i for i in range(S)}
    first_tag_counts = {tags[i]:0 for i in range(S)}
    tag_counts = {tags[i]:0 for i in range(S)}
    tag_tag_counts = {tags[i]:{tags[j]:0 for j in range(S)} for i in range(S)}
    tag_word_counts = {tags[i]:{} for i in range(S)}
    
    for line in train_data:
        first_tag_counts[line.tags[0]] += 1

        for index in range(line.length):
            tag, word = line.tags[index], line.words[index]
            
            if word not in word_index_map:
                word_index_map[word] = len(words)
                words.append(word)
            
            tag_counts[tag] += 1
            tag_word_counts[tag].setdefault(word, 0)
            tag_word_counts[tag][word] += 1

            if index < line.length-1:
                nexttag = line.tags[index+1]
                tag_tag_counts[tag][nexttag] += 1
                
    pi = np.array([first_tag_counts[t] for t in tags]) / L
    tag_counts_array = np.array([[tag_counts[t]] for t in tags])
    A = np.array([[tag_tag_counts[s].get(ss, default) for ss in tags] for s in tags]) / tag_counts_array
    B = np.array([[tag_word_counts[s].get(w, default) for w in words] for s in tags]) / tag_counts_array
    
    model = HMM(pi, A, B, word_index_map, state_dict)
    ###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class

    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    ###################################################
    default = 1e-06
    for i in range(len(test_data)):
        for word in test_data[i].words:
            if word not in model.obs_dict:
                model.obs_dict[word] = len(model.obs_dict)
                new_col = default * np.ones([model.B.shape[0], 1])
                model.B = np.concatenate((model.B, new_col), axis=1)
        tagging.append(model.viterbi(test_data[i].words))
    ###################################################
    return tagging
