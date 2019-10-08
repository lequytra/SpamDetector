import numpy as np
from AttentionLayer import AttLayer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import Embedding, GRU, Bidirectional, TimeDistributed
from keras.models import Model

MAX_SENT_LENGTH = 100
MAX_SENTS = 30
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2


def text_processing(emails, labels, weights_path=None):
    """
        Turn a list of email contents to (num_email, sentences, num_words)
    :param weights_path: Path to pretrained word embeddings
    :param emails: list of email contents
    :return: list of lists
    """
    # Split emails' contents to list of sentences

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(emails)
    # Initialize 3D array of 0s
    emails = [item.split('.') for item in emails]
    data = np.zeros((len(emails), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(emails):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for word in wordTokens:
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))

    labels = np.asarray(labels)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print("Number of spams in the training set is {}/{}".format(y_train.sum(axis=0), len(y_train)))
    print("Number of spams in the training set is {}/{}".format(y_val.sum(axis=0), len(y_val)))

    embeddings_index = {}
    with open(weights_path, 'r+') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                continue

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros(shape=(len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return (x_train, y_train), (x_val, y_val), word_index, embedding_matrix


def HATN(num_words, trainable=True, embedding_matrix=None):
    if not trainable:
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=False,
                                    mask_zero=False)

    else:
        embedding_layer = Embedding(num_words,
                                    EMBEDDING_DIM,
                                    embeddings_initializer='uniform',
                                    input_length=MAX_SENT_LENGTH,
                                    trainable=True,
                                    mask_zero=True)

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(100)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(100)(l_lstm_sent)
    preds = Dense(1, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)

    return model


# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])
#
# print("model fitting - Hierachical attention network")
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=10, batch_size=50)
