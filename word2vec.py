import os
import re
import string
import tensorflow as tf
import numpy as np
import tqdm
from tensorboard.plugins import projector
from data_preprocessing import generate_batch_dynamic, build_dataset, save_vectors, read_analogies, get_vectors
from evaluation import evaluation
from random import shuffle
import datetime as dt
import matplotlib.pyplot as plt
from utilities import serialize, deserialize

# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

### PARAMETERS ###
BATCH_SIZE = 100  # Number of samples per batch
EMBEDDING_SIZE = 150  # Dimension of the embedding vector.
WINDOW_SIZE = 2  # How many words to consider left and right.
NEG_SAMPLES = 20  # Number of negative examples to sample.
VOCABULARY_SIZE = 100000  # The most N word to consider in the dictionary
EPOCHS = 3  # Number of epochs to perform training

TRAIN_DIR = "dataset/DATA/TRAIN"
STOP_WORDS = "englishST.txt"
TMP_DIR = "/tmp/"
ANALOGIES_FILE = "dataset/eval/questions-words.txt"

TO_SERIALIZE = True  # Wether to serialize the dictionary and the dataset or not
DATA_DICT_SERIALIZATION = "dataset_dict.data"

# Generate raw data
c_nfiles = 18000
c_stopwd = True
c_shuffle_docs = True
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.


# --------------------------------------------- UTILITY  FUNCTIONS ----------------------------------------------------

def plot_acc_loss(accuracy, lossp):
    """
    Given two lists as input this function plots the values contained in them on the same graph. It is used for plotting
    the accuracy and loss.
    :param accuracy: a list of floats representing the accuracy at each step.
    :param lossp: a list of floats representing the loss at each step.
    """
    plt.plot(accuracy, label='accuracy')
    plt.plot(lossp, label='loss')
    plt.legend(loc='upper right')
    plt.xlabel("steps")
    plt.show()


def date_of_today():
    """
    It returns a string representing the information about the time and date in the moment it gets called.
    :return: String - time and date of this moment.
    """
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def stop_words_list():
    """
    It returns the list of stop-words of the english language srored in the file 'STOP_WORDS'.
    :return: a set ot strings, the stopwords contained in 'STOP_WORDS' file.
    """
    return set([w.rstrip('\r\n') for w in open(STOP_WORDS)])


def remove_punctuation(word):
    """
    Preprocess words of the dataset. Remove punctuation.
    :param word: String - the word to preprocess
    :return: String - the pre processed word
    """
    out = re.sub('[%s]' % re.escape(string.punctuation), '', word)
    return out


def read_data(directory, nfiles=-1):
    """
    Read nfiles (all of the files if  nfiles=-1) from each folder of directory. It return the whole dataset.
    :param directory: String - the path containing the domains
    :param nfiles: Integer - the number of files to read for each directory, optional, default -1
    @return [String] - the whole dataset of words
    """

    # Get the list of stopwords from the file
    stopwords = stop_words_list()

    # A list of lists representing the words in each of the documents
    documents = []

    # count the processed files up to now
    ac_dom = 0

    # FOR EACH of the domains
    for domain in os.listdir(directory):
        ac_dom += 1
        print('DOMAIN: ', ac_dom, 'of', len(os.listdir(directory)))

        # Ingores hidden folders
        if domain.startswith("."):
            continue

        # Counts the number of files read per directory
        limit_files = 0

        # FOR EACH of the docuents in this domain
        for f in os.listdir(os.path.join(directory, domain)):

            limit_files += 1

            # Checks if the limit of file read has been reached
            if limit_files > nfiles > 0:
                break

            # A list fo the words in this document
            document = []
            if f.endswith(".txt"):
                with open(os.path.join(directory, domain, f)) as file:

                    # FOR EACH of the lines in this document
                    for line in file.readlines():
                        split = line.lower().strip().split()

                        # FOR EACH of the words in this line
                        for wd in split:
                            wd = remove_punctuation(wd)
                            if wd == '' or wd in stopwords:
                                continue
                            document.append(wd)

            documents.append(document)

    shuffle(documents)

    # Get all words in the now randomized documents
    dataset = [wd for doc in documents for wd in doc]

    return dataset


# -------------------------------------------------- START -------------------------------------------------------------

print("Start test " + date_of_today())

# Get 16 random values from 0 to 100 without replacement 
valid_examples = np.random.choice(valid_window, valid_size, replace=False) 

if TO_SERIALIZE:

    raw_data = read_data(TRAIN_DIR, nfiles=c_nfiles)

    # Build the dataset and the dictionaries form the raw data
    data, dictionary, reverse_dictionary = build_dataset(raw_data, VOCABULARY_SIZE)

    del raw_data  # To reduce memory.

    # To avoid reading the whole dataset again an again
    print("Serializing the data.")
    serialize(DATA_DICT_SERIALIZATION, (data, dictionary, reverse_dictionary))
else: 
    print("Reading serialization: ")
    data, dictionary, reverse_dictionary = deserialize(DATA_DICT_SERIALIZATION)

# Stores some informations about the actual test configuration
configuration = 'BATCH_SIZE: ' + str(BATCH_SIZE) + ' EMBEDDING_SIZE: ' + str(EMBEDDING_SIZE) + ' WINDOW_SIZE: ' \
                + str(WINDOW_SIZE) + ' VOCABULARY_SIZE ' + str(VOCABULARY_SIZE) + ' nfiles: ' + str(c_nfiles) + \
                ' stopwd: ' + str(c_stopwd) + ' shuffle_docs: ' + str(c_shuffle_docs) + ' dataset_size ' + str(len(data))

print("CONFIG: " + configuration)

# Read the question file for the Analogical Reasoning evaluation
questions = read_analogies(ANALOGIES_FILE, dictionary)


# ------------------------------------------ MODEL DEFINITION --------------------------------------------------------

graph = tf.Graph()
evall = None

with graph.as_default():
    # Define input data tensors.
    with tf.name_scope('inputs'):

        # Input layer for the batches
        train_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        # Output layer for the predictions
        train_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Embeddings matrix
    embeddings = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1, 1))

    # Weights and bias for the noise-contrastive estimation loss
    nce_weights = tf.Variable(tf.truncated_normal([VOCABULARY_SIZE, EMBEDDING_SIZE],
                                                  stddev=1.0 / np.sqrt(EMBEDDING_SIZE)))
    nce_biases = tf.Variable(tf.zeros([VOCABULARY_SIZE]))

    # Look up function to look up the vectors for each of the words passed as input
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels, inputs=embed,
                                             num_sampled=NEG_SAMPLES, num_classes=VOCABULARY_SIZE))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdagradOptimizer(learning_rate=1)
        train = optimizer.minimize(loss)

    # Embeddings evaluation
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

    # evaluation graph
    evall = evaluation(normalized_embeddings, dictionary, questions)

# ------------------------------------------ START TRAINING --------------------------------------------------------

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(TMP_DIR + 'logging', session.graph)
    # We must initialize all variables before we use them.
    init.run()

    print('Initialized')

    # Some metrics to plot
    losses = []
    accuracies = []
    average_loss = 0

    # The bar showing the progress of the training
    bar = tqdm.tqdm(total=EPOCHS*len(data))
    avg = ""
    e = ""

    # Run the training
    for step in range(EPOCHS):
        # Counts the iterations
        iteration = -1

        # The starting pair pairs indicates in position 0 where in the dataset you need to start generating the batch,
        # and in position 1 where in the window (of the element in position 1) you must start reading.
        starting_pair = (0, 0)

        # Index of the last read batch (used for updating the bar
        last_batch_index = 0

        # While you didn't reach the end of the dataset, keep reading
        while starting_pair[0] < len(data):
            iteration += 1

            # Dynamically generate the imput batch, the labels and keep track of the starting pair
            batch_inputs, batch_labels, starting_pair = \
                generate_batch_dynamic(BATCH_SIZE, starting_pair[0], starting_pair[1], WINDOW_SIZE, data)

            # Update the bar
            if last_batch_index != starting_pair[0]:
                bar.update(starting_pair[0] - last_batch_index)
            last_batch_index = starting_pair[0]

            # If a batch has a smaller size than expected, then avoid it
            if len(batch_inputs) != BATCH_SIZE:
                continue
        
            # Define metadata variable.
            run_metadata = tf.RunMetadata()

            # Feed the data to the input placeholder and execute the operations 'train' 'merged' and 'loss'
            batch_labels = np.expand_dims(batch_labels, axis=1)
            feeder = {train_inputs: batch_inputs, train_labels: batch_labels}

            _, summary, loss_val = session.run([train, merged, loss], feed_dict=feeder, run_metadata=run_metadata)

            average_loss += loss_val
            
            # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)

            # Test the goodness of the trained model
            if iteration % 500000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

                writer.add_run_metadata(run_metadata, 'step%d' % step)

                desc, acc = evall.eval(session)
                losses.append(loss_val)
                accuracies.append(acc)

                print("\n", desc)
                avg = "avg loss: " + str(average_loss / iteration)
                print(avg)

    plot_acc_loss(accuracies, losses)

    desc, _ = evall.eval(session)
    print("FINALS: ", desc)

    # Add some information to the configuration string in order to store it on the disk
    configuration += "\n" + desc + "\n" + avg

    final_embeddings = normalized_embeddings.eval()

    # Serialize final embeddings
    save_vectors("test" + date_of_today() + ".data", reverse_dictionary, final_embeddings, configuration)

    # Save the model for checkpoints
    saver.save(session, os.path.join(TMP_DIR, 'model.ckpt'))

    # Write corresponding labels for the embeddings.
    with open(TMP_DIR + 'metadata.tsv', 'w') as f:
        for i in range(VOCABULARY_SIZE):
            f.write(reverse_dictionary[i] + '\n')

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(TMP_DIR, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

writer.close()
