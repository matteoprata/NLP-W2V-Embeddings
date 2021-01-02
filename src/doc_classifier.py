from collections import Counter
from data_preprocessing import reverse_dictionary, serialize, deserialize
from word2vec import remove_punctuation
import tensorflow as tf
import numpy as np
import re
import os
from sklearn import svm
from utilities import serialize, deserialize
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


# Where to store the embeddings of the documents in the datasets
SERIALIZED_TRAINING = "Partitions/training_set.data"
SERIALIZED_TEST = "Partitions/test_set.data"
SERIALIZED_TEST_COMP = "Partitions/comp_set.data"

# Where to store the answers of the competition
ANSWERS_TEST_COMP = "test_answers.tsv"

# Where to store the model (neural network weights and bias)
NN_MODEL = "nn_model.model"

# Paths to retrieve the datasets and the embeddings
PATH_TRAIN = "dataset/DATA/TRAIN"
PATH_TEST = "dataset/DATA/DEV"
PATH_TEST_COMP = "dataset/DATA/TEST"
SERIALIZED_W2V_OUT = "final_embeddings.em"

WD_THRESH = 20  # Threshold of the number of most frequent words to consider for each file
NUM_CLASSES = 34  # Number of classes
EMB_SIZE = 150  # Size of the embeddings

# The classes and their partial encodings
CLASS_ENC_DICT = {'HERALDRY_HONORS_AND_VEXILLOLOGY': 0, 'GEOGRAPHY_AND_PLACES': 1, 'BIOLOGY': 2, 'EDUCATION': 3,
                  'ENGINEERING_AND_TECHNOLOGY': 4, 'SPORT_AND_RECREATION': 5,
                  'MEDIA': 6, 'ROYALTY_AND_NOBILITY': 7, 'CHEMISTRY_AND_MINERALOGY': 8,
                  'ART_ARCHITECTURE_AND_ARCHAEOLOGY': 9, 'POLITICS_AND_GOVERNMENT': 10, 'PHYSICS_AND_ASTRONOMY': 11,
                  'LAW_AND_CRIME': 12, 'LANGUAGE_AND_LINGUISTICS': 13, 'FARMING': 14, 'METEOROLOGY': 15,
                  'COMPUTING': 16, 'BUSINESS_ECONOMICS_AND_FINANCE': 17, 'WARFARE_AND_DEFENSE': 18,
                  'LITERATURE_AND_THEATRE': 19, 'TEXTILE_AND_CLOTHING': 20, 'FOOD_AND_DRINK': 21, 'MATHEMATICS': 22,
                  'GEOLOGY_AND_GEOPHYSICS': 23, 'GAMES_AND_VIDEO_GAMES': 24, 'HISTORY': 25,
                  'ANIMALS': 26, 'CULTURE_AND_SOCIETY': 27, 'HEALTH_AND_MEDICINE': 28, 'MUSIC': 29,
                  'NUMISMATICS_AND_CURRENCIES': 30, 'PHILOSOPHY_AND_PSYCHOLOGY': 31,
                  'RELIGION_MYSTICISM_AND_MYTHOLOGY': 32, 'TRANSPORT_AND_TRAVEL': 33}


def sort_nicely(list_to_sort):
    """
    This function sorts the file in a human-readable way. (Found on the web)
    :param list_to_sort: the list to sort
    :return: the sorted list
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    list_to_sort.sort(key=alphanum_key)
    return list_to_sort


def generate_document_vector(doc_words, word_ind_dict, embeddings):
    """
    For each word in 'doc_words' the functon gets its embedding then it computes the mean embedding that's the embedding of the document.
    :param doc_words: the list of words to generate the embedding of the document
    :param word_ind_dict: the voucabulary with the association word:encoding
    :param embeddings: the embedding matrix
    :return: the embedding of the document.
    """
    doc_embeddings = []
    for word in doc_words:
        ind = word_ind_dict[word]
        doc_embeddings.append(embeddings[ind])

    # It computes the average of the words embeddings stored in doc_embeddings
    average = np.mean(np.array(doc_embeddings), axis=0)
    return average


# This is an overloading of generate_datasets
def generate_training_data(ind_wd_dict, embeddings, files, fromDisk):
    return __generate_datasets(PATH_TRAIN, SERIALIZED_TRAINING, ind_wd_dict, embeddings, fromDisk, files)


# This is an overloading of generate_datasets
def generate_test_data(ind_wd_dict, embeddings, files, fromDisk):
    return __generate_datasets(PATH_TEST, SERIALIZED_TEST, ind_wd_dict, embeddings, fromDisk, files)


def __generate_datasets(directory, directory_ser, ind_wd_dict, embeddings, fromDisk, limit_docs):
    """
    It is reposnible for creating the pairs (document_embedding, domain_label) for the classification task, 
    given the path of the dataset, the vocabulary and the embeddings matrix. It is used both for the training dataset
    and the dev dataset. It can store the genated instances on disk, or read them from disk if specified.
    It can read only part of the dataset, if needed.

    :param directory: the directory where the dataset is stored
    :param directory_ser: direcorty in which to serialize the resulting documents embeddings
    :param ind_wd_dict: the voucabulary with the association encoding:word
    :param embeddings: the embedding matrix
    :param fromDisk: if true we require deserialization, otherwise we store on the disk
    :param limit_docs: limits the number of documents to read for each directroy
    :return: the lists containing the document embeddings and the list containing the relative labels (the domains)
    """

    # If it is requested deserialization
    if fromDisk:
        try:
            return deserialize(directory_ser)
        except:
            print('No serialized dataset yet. Set fromDisk option to True.')

    # It stores the document embeddings
    x_instances = []
    # It stores the labels of the embeddings (the domains)
    y_classes = []

    wd_ind_dict = reverse_dictionary(ind_wd_dict)

    # FOR EACH of the domains
    for domain in sorted(os.listdir(directory)):
        # Ignore hidden directories
        if domain.startswith("."):
            continue

        # Limits the number of files to read according to 'limit_docs'
        list_files = sorted(os.listdir(os.path.join(directory, domain))) if limit_docs == -1 else sorted(os.listdir(os.path.join(directory, domain)))[:limit_docs]

        # FOR EACH of the documents
        for f in list_files:

            # This keeps traks of the occurrences of each word in the document
            words = Counter()
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:

                    # FOR EACH of the lines of the document
                    for lines in file.readlines():
                        content = lines.lower().strip().split()
                        for wd in content:
                            wd = remove_punctuation(wd).strip()
                            try:
                                wd_ind_dict[wd]  # Check if the word is in the dictionary
                                words[wd] += 1
                            except:
                                None

            doc_words = [wd for wd, _ in words.most_common(WD_THRESH)]
            # doc_words = [wd for wd in words]

            # With doc_words go to generate the embedding of this document
            document_embedding = generate_document_vector(doc_words[:WD_THRESH], wd_ind_dict, embeddings)

            x_instances.append(document_embedding)
            y_classes.append(domain)

    serialize(directory_ser, (x_instances, y_classes))

    return x_instances, y_classes


# -------------------------------------------- SK-LEARNER: SVM --------------------------------------------------
# Not used for the task, but it works
def svm_classifier_train(X, Y, XVER):
    """
    This learner takes X instances and learns to predict Y labels. Then from new unseen instances XVER, it returns a prediction.
    :param X: the instances to perform the training with
    :param Y: the labels to perform the training with
    :param XVER: the new unseen instances to make predictions
    :return: the list of predicted labels for XVER instances
    """
    clf = svm.SVC(decision_function_shape='ovo')

    # Removes traces of the empy file
    for i, x in enumerate(X):
        if not x.shape == (150,):
            X.pop(i)
            Y.pop(i)

    clf.fit(X, Y)
    return classifier_predict(clf, XVER)


# ---------------------------------------- SK-LEARNER: LOG-REGRESSION --------------------------------------------
# Not used for the task, but it works
def lg_classifier_train(X, Y, XVER):
    """
    This learner takes X instances and learns to predict Y labels. Then from new unseen instances XVER, it returns a prediction.
    :param X: the instances to perform the training with
    :param Y: the labels to perform the training with
    :param XVER: the new unseen instances to make predictions
    :return: the list of predicted labels for XVER instances
    """
    clf = LogisticRegression(solver='sag', max_iter=len(X) * 100, random_state=42, multi_class='multinomial')

    # Removes traces of the empy file
    for i, x in enumerate(X):
        if not x.shape == (150,):
            X.pop(i)
            Y.pop(i)

    clf.fit(X, Y)
    return classifier_predict(clf, XVER)


def classifier_predict(model, X):
    """
    Given a model and list of instances it guesses their labels.
    :param model: the model learned by a learner
    :param X: a list of instances
    :return: the list of predicted labels for X
    """
    y_pred = []
    for e, x in enumerate(X):
        list_y = model.predict(x.reshape(1, -1))
        y_pred.append(list_y[0])

    return y_pred


# --------------------------------- TENSOR FLOW LEARNER: LOG-REGRESSION NN ---------------------------------------

def encode_nn_out(Ydata):
    """
    This creates the encodings for the labels contained in Ydata. The encodings are needed for allowing the neural network
    to make predicitions. The encoding of the a class depend on its partial encoding. The partial encoding is an integer stored in
    CLASS_ENC_DICT for all of the classes that, it goes from 0 to NUM_CLASSES-1. The encoding is a list of size NUM_CLASSES
    of all zeros except in the position represented by the encoding of that class.
    
    For example label ANIMALS which has partial encoding 0 whould have and encoding: [1,0,0,0,0,0,...,0]

    :param Ydata: the list of labels to encode
    :return: the list of encoded labels
    """
    encoded_out = []
    for val in Ydata:
        # A list of NUM_CLASSES zeros
        enc = [0] * NUM_CLASSES
        # Put a 1 corrisponding to the partial encoding defined in CLASS_ENC_DICT
        enc[CLASS_ENC_DICT[val]] = 1
        encoded_out.append(enc)

    return encoded_out


def shuffle_datasets(X, Y):
    """
    Given the lists X and Y they get shuffled according to the same permutation.

    :param X: the first list
    :param Y: the second list
    :return: the two shuffled lists
    """
    perm = np.random.permutation(len(X))
    Xn = []
    Yn = []
    for ind in perm:
        Xn.append(X[ind])
        Yn.append(Y[ind])
    return Xn, Yn


def nn_tf_classifier(X, Y, XVER):
    """
    This learner takes X instances and learns to predict Y labels. Then from new unseen instances XVER, it returns a prediction.
    :param X: the instances to perform the training with
    :param Y: the labels to perform the training with
    :param XVER: the new unseen instances to make predictions
    :return: the list of predicted labels for XVER instances
    """
    X_SIZE = EMB_SIZE
    Y_SIZE = NUM_CLASSES

    print("Shuffling:")
    X, Y = shuffle_datasets(X, Y)

    # Y now is the set of the encoded labels
    Y = encode_nn_out(Y)
    enc_to_class = reverse_dictionary(CLASS_ENC_DICT)

    sess = tf.Session()
    tf.set_random_seed(2)
    
    # The input layer - documents imbeddings will flow here
    x = tf.placeholder(tf.float32, [1, X_SIZE])
    # Layer used to make comparisons between prediciton and ground truth 
    y = tf.placeholder(tf.float32, [1, Y_SIZE])
    
    # Weights connecting EMB_SIZE neurons to NUM_CLASSES neurons 
    W = tf.Variable(tf.zeros([X_SIZE, Y_SIZE]))
    # The bias connected to NUM_CLASSES neurons
    b = tf.Variable(tf.zeros([Y_SIZE]))
    
    # The output tensor 
    out = tf.nn.softmax(tf.matmul(x, W) + b)
    # This will be equal to the index of the output tensor with the highest value (=the partial encoding of a label)
    prediction = tf.argmax(out, 1)
    
    # Loss function and optimizers 
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # The training phase starts
    epochs = 10
    for curr in range(epochs):
        print('Epoch:', str(curr + 1) + "/" + str(epochs))
        for i in range(len(X)):
            ex = np.expand_dims(X[i], axis=0)
            ey = np.expand_dims(Y[i], axis=0)
            # Avoids empty embeddings of empty files
            if ex.shape == (1, 150) and ey.shape == (1, 34):
                sess.run(train_step, feed_dict={x: ex, y: ey})

    # Save the model
    serialize(NN_MODEL, (W.eval(), b.eval()))

    # Stores the predicitons for the instances in XVER
    y_pred = []
    for i in range(len(XVER)):
        ex = np.expand_dims(XVER[i], axis=0)
        yp = sess.run(prediction, feed_dict={x: ex})
        label = enc_to_class[yp[0]]
        y_pred.append(label)

    sess.close()

    return y_pred


def nn_tf_trained_classifier(XVER):
    """
     Then from new unseen instances XVER, it returns a prediction. Note this learner is already trained with a previously stored model.
    :param XVER: the new unseen instances to make predictions
    :return: the list of predicted labels for XVER instances
    """
    X_SIZE = EMB_SIZE
    enc_to_class = reverse_dictionary(CLASS_ENC_DICT)
   
    # Reads from memory the stored model (= weights and biases)
    weights, bias = deserialize(NN_MODEL)
    
    sess = tf.Session()
    tf.set_random_seed(2)

    x = tf.placeholder(tf.float32, [1, X_SIZE])

    W = tf.constant(weights)
    b = tf.constant(bias)

    out = tf.nn.softmax(tf.matmul(x, W) + b)
    prediction = tf.argmax(out, 1)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    y_pred = []
    for i in range(len(XVER)):
        ex = np.expand_dims(XVER[i], axis=0)
        yp = sess.run(prediction, feed_dict={x: ex})
        label = enc_to_class[yp[0]]
        y_pred.append(label)

    sess.close()

    return y_pred


# --------------------------------- EVALUATION PHASE ---------------------------------------

def evaluate_model(y_pred, y_true):
    """
    It prints some statistics and interestin metrics about the evaluation.
    :param y_pred: a list containing the predicted labels
    :param y_true: a alist containing the ground truth labels 
    """
    print('Test set size:', len(y_pred))

    correct = 0
    errors = len(y_pred)
    for x in range(len(y_pred)):
        # Correctness condition
        if y_true[x] == y_pred[x]:
            correct += 1
            errors -= 1

    print('True positives:', correct)
    print('Number of errors:', errors)
    print('Precision:', correct / len(y_pred))

    labels = sorted([*set(y_true)])
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Preints the precision, recall and f1measure for all of the classes
    stats = []
    for ind, lab in enumerate(labels):
        stat = []
        lab = lab.replace("_", "\_")
        print('Stats of class:', lab)
        
        # cm[ind][ind] are the values on the diagonal
        precision = cm[ind][ind] / np.sum(cm[ind])
        print('\t precision:', precision)
        stat.append((lab, precision))

        recall = cm[ind][ind] / np.sum([cm[i][0] for i in range(len(cm))])
        print('\t recall:', recall)
        stat.append((lab, recall))

        f2measure = 2 * (precision * recall) / (precision + recall)
        print('\t f1measure:', f2measure)
        stat.append((lab, f2measure))
        print()
        stats.append(stat)

# ------------------------------------------------------- RUN ONE OF THESE TWO --------------------------------------------------------------------------

def run_classification_comp(toserialize):
    """
    This function is used for executing the the classification on a learned model, for the TEST class. So this function produces
    the .tsv file used as the answer for the domain classification competition.
    :param toserialize: wether to serialize the embeddings of TEST dataset or not.
    """
    print("Deserializing world embeddings.")

    # We read the serialized vocabulary, that maps word-encodings to words, and word embeddings
    ind_wd_dict, embeddings, _ = deserialize(SERIALIZED_W2V_OUT)

    # Store the reversed version of ind_wd_dict
    wd_ind_dict = reverse_dictionary(ind_wd_dict)

    print("Reading files.")
    if (toserialize):

        x_instances = []
        pos = 0

        for f in sort_nicely(os.listdir(PATH_TEST_COMP)):
            pos += 1
            print(pos, "di", len(os.listdir(PATH_TEST_COMP)))
            words = Counter()
            with open(os.path.join(PATH_TEST_COMP, f)) as file:

                for lines in file.readlines():

                    # Manipulate the line so to make its words ready for the look up in the dictionary
                    content = lines.lower().strip().split()
                    for wd in content:
                        wd = remove_punctuation(wd).strip()
                        try:
                            # Global dictionary (wd_ind_dict) lookup if the word is in the global dictionary increase it's counter in the local dictionary (words), otherwise add it
                            wd_ind_dict[wd]
                            words[wd] += 1
                        except:
                            # If the word is not in the global dictionary it means it is a stopword, don't do anything.
                            None

                # The list of the most frequent (WD_THRESH times) words of this document
                doc_words = [wd for wd, _ in words.most_common(WD_THRESH)]
                # print('Generated embedding for:', f)

                # For all of the most frequent words in doc_words generate their embeddings and the final document embedding, store it in document_embedding
                document_embedding = generate_document_vector(doc_words, wd_ind_dict, embeddings)

                # It appends parirs (name of the document, document embedding)
                x_instances.append((f, document_embedding))

        print('Serializing embeddings of the documents.')
        serialize(SERIALIZED_TEST_COMP, x_instances)
        print('Serialized embeddings of the documents.')

    else:
        x_instances = deserialize(SERIALIZED_TEST_COMP)

    print("Prediction.")

    # Isolates all documents embeddings in the X list, ready to be classified
    X = [doc_emb for _, doc_emb in x_instances]
    F = [fname for fname, _ in x_instances]

    # Predict the labels for the emebddings stored in X. The model should already be trained.
    predictions = nn_tf_trained_classifier(X)

    # Write the answer file with format filename \t prediction \n
    with open(ANSWERS_TEST_COMP, 'w') as f:
        for i, fname in enumerate(F):
            idd = fname.split(".")[0]
            f.write(idd + "\t" + predictions[i] + "\n")


def run_classification():
    """
    This function is used for training a learner and executing the the classification on a learned model and its evaluation.
    """
    print("Deserializing word embeddings.")
    ind_wd_dict, embeddings, _ = deserialize(SERIALIZED_W2V_OUT)

    print("Generating training data.")
    X, Y = generate_training_data(ind_wd_dict, embeddings, -1, True) # Generate the training data.

    print("Generating test data.")
    Xv, Yv = generate_test_data(ind_wd_dict, embeddings, -1, True) # Generate the tasting data.

    print("Prediction.")
    # predictions = lg_classifier_train(X, Y, Xv)
    # predictions = svm_classifier_train(X, Y, Xv)
    predictions = nn_tf_classifier(X, Y, Xv   # After the training with X, Y lists, predict labels for Xv  
    # predictions = nn_tf_trained_classifier(Xv)

    print("Evaluation.")
    evaluate_model(predictions, Yv) # Evaluate the predictions 


# run_classification()
run_classification_comp(True)



