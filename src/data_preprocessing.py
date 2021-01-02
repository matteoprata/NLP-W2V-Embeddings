from collections import Counter
import math
import numpy as np
import operator
import sys
import pickle
import re
import string 

#UNUSED
def generate_batch_static(window_size, data):
    """
    It generates the whole train data and label batch from the dataset. The train data and label data are given as output.
    Since they tend to be very big depending on data size and window size, it may be wise to prefer the dynamic version of
    the algorithm.
    :param window_size: Integer - the size of the window
    :param data: [String] - the list of words in the dataset
    :return: ([String], [String]) - the train data and the labeled data of the whole dataset
    """

    train_data = []
    labels = []
    
    for iwd in range(len(data)):
        # The list represents the window of item data[iwd] of size at most 2*window_size, item data[iwd] is excluded
        wind = data[max(iwd - window_size, 0): iwd] + data[iwd+1: min(iwd + window_size, len(data))+1]
        
        # Put each element of the window in labels, put the corresponding data[iwd] in train_data
        for el in wind:
            train_data.append(data[iwd])
            labels.append(el)

    return train_data, labels


def generate_batch_dynamic(batch_size, data_start, window_start, window_size, data):
    """
    It generates training data and relative labels from the dataset. The train data and label data have size
    batch_size and they are dynamically generated from data[data_start] and from the index window_start of its window.
    :param batch_size: Integer - the size of this batch
    :param data_start: Integer - the index in data where to start generating the batch
    :param window_start: Integer - the index in window of data[data_start] where to start generating the batch
    :param window_size: Integer - the size of the window
    :param data: [String] - the list of words in the dataset

    :return: ([String], [String], (Integer, Integer)) - the train data and the labeled data, a couple of integer
    representing the point where to start generating the batches next time
    """

    train_data = []
    labels = []

    for iwd in range(data_start, len(data)):
        # The list represents the window of item data[iwd] of size at most 2*window_size, item data[iwd] is excluded
        wind = data[max(iwd - window_size, 0): iwd] + data[iwd + 1: min(iwd + window_size, len(data)) + 1]
        
        # Put each element of the window in labels, put the corresponding data[iwd] in train_data.
        # Start from window_start avoid element generator of the window
        for el in wind[window_start:]:

            # If train_data and labels have space fill them, else return them, further indices of data and window
            if len(train_data) < batch_size:
                train_data.append(data[iwd])
                labels.append(el)
                window_start += 1
            else:
                return train_data, labels, (data_start, window_start)
        data_start += 1
        window_start = 0
    return train_data, labels, (data_start, window_start)


def build_dataset(words, vocab_size):
    """
    This function is responsible of generating the dataset and dictionaries (=vocabularies).
    Dictionaries contain the association between a word and its encoding. The dataset
    is the list of encoded words.
    :param words: [String] - the list of words of the domains
    :param vocab_size: Integer - the desired size of the vocabularies
    :return: ([Integer],{String:Integer},{Integer:String}) - the dataset of encodings and the dictionaries
    """

    dictionary = Counter()
    vocabulary_occ = Counter()
    reversed_dictionary = dict()
    data = []
    unk = '<UNK>'
    
    # It builds a dictionary of shape word:number_occurrences 
    for wd in words:
        vocabulary_occ[wd] += 1

    # 1) Create dictionary 
    # A special word <UNK> is encoded 0
    ind = 0
    dictionary[unk] = ind
    # For all the words in vocabulary_occ, sorted by value: associate word to its encoding
    for wd, _ in vocabulary_occ.most_common(vocab_size-1):
            ind += 1
            dictionary[wd] += ind

    # 2) Create dictionary 
    # Each word in dictionary is the value of its encoding in the vocabulary
    for wd in dictionary:
        reversed_dictionary[dictionary[wd]] = wd

    # 3) Create data
    # Each word in words is translated into its encoding that's then put into a list
    for wd in words:
        if wd not in dictionary:
            wd = unk
        data.append(dictionary[wd])

    return data, dictionary, reversed_dictionary


def serialize(file_name, data):
    """
    It serializes the data passed as input in the specified location.
    :param file_name: String - the path of the file where to store the data
    :param data: T - wether sort of data to store
    """

    with open(file_name, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize(file_name):
    """
    It deserializes the data stored in the specified location.
    :param file_name: String - the path of the file from where to retrieve the data
    :return: T - wether sort of data serialized in file_name
    """

    with open(file_name, 'rb') as handle:
        data = pickle.load(handle)
    return data


def save_vectors(file_name, reversedic, vectors, configuration):
    """
    Save embedding vectors in a suitable representation for the domain identification task.
    I also save the reverse dictionary and a descriptive string.
    :param file_name: String - the path of the file from where to save the data
    :param reversedic: the reversed dictionary contaiining id:word pairs
    :param vectors: the matrix of the embeddings
    :param configuration: String - a scring describing the test configuration
    """

    serialize(file_name, (reversedic, vectors, configuration))


def reverse_dictionary(dictionary):
    """
    It inverts a dictionary, meaning that it swaps key and value.
    :param dictionary: the dictionary to reverse
    :return: the reversed dictionay
    """
    
    out_dict = dict()
    for ent in dictionary:
        out_dict[dictionary[ent]] = ent
    return out_dict


def read_analogies(file, dictionary):
    """
    Reads through the analogy question file. It returns a numpy array representing the questions.
    Each question is a list of encodings representing words that are very close to each other.
    It prints the total number of possible questions in the file, the skipped questions due to
    small dictionary, the used questions.
    :param file: String - the path of the file with the questions
    :param dictionary:  {String:Integer} - the dictionary containing the encoding of each word
    :return:  numpy.array of lists - the array of questions
    """

    questions = []
    questions_skipped = 0
    with open(file, "r") as analogy_f:
        for line in analogy_f:
            if line.startswith(":"):  # Skip comments.
                continue
            words = line.strip().lower().split(" ")
            ids = [dictionary.get(str(w.strip())) for w in words]
            
            if None in ids or len(ids) != 4:
                questions_skipped += 1
            else:
                questions.append(np.array(ids))
                
    print("Eval analogy file: ", file)
    print("Questions: ", len(questions))
    print("Skipped: ", questions_skipped)
    return np.array(questions, dtype=np.int32)


