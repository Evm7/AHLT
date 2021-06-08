import nltk
from nltk.corpus import stopwords
from xml.dom.minidom import parse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import keras as k

import pandas as pd
from keras.preprocessing.sequence import pad_sequences

from nltk.tokenize import word_tokenize
from os import listdir
import string, sys
import numpy as np
import pickle
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
stopwords_ = set(stopwords.words('english'))

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, concatenate, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Layer
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy

sys.path.append("../")
import evaluator

class Learner():
    def __init__(self):
        print("[WELCOME]... Init learning progress")

    def tokenize(self, sentence):
        '''
        Task :
        Given a sentence , calls nltk . tokenize to split it in
        tokens , and adds to each token its start / end offset
        in the original sentence .
        '''
        tokens = []
        offset = 0

        words = word_tokenize(sentence)

        for w in words:
            if (w in stopwords_) or (w in string.punctuation):
                continue
            offset = sentence.find(w, offset)
            tokens.append((w, offset, offset + len(w) - 1))
            offset += len(w) +1

        return tokens

    def get_tag(self,token, gold):
        '''
        Task :
        Given a token and a list of ground truth entites in a sentence , decide
        which is the B-I-O tag for the token
        '''
        (form, start, end) = token
        for (gold_start, gold_end, gold_type) in gold:
            if start == gold_start and end <= gold_end:
                return "B-" + gold_type
            elif start >= gold_start and end <= gold_end:
                return "I-" + gold_type
        return "O"

    def load_data(self, datadir):
        '''
        Load XML files in given directory , tokenize each sentence , and extract ground truth BIO labels for each token .
        '''
        result = {}
        # process each file in directory
        for f in listdir(datadir):
            # parse XML file , obtaining a DOM tree
            tree = parse(datadir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text
                # load ground truth entities .
                gold = []
                entities = s.getElementsByTagName("entity")
                for e in entities:
                    # for discontinuous entities , we only get the first span
                    offset = e.attributes["charOffset"].value
                    (start, end) = offset.split(";")[0].split("-")
                    gold.append((int(start), int(end), e.attributes["type"].value))
                # tokenize text
                tokens = self.tokenize(stext)
                info_ = []
                for tok_ in tokens:
                    tag_ = self.get_tag(tok_, gold)
                    n, i1, i2 = tok_
                    info_.append((n, i1, i2, tag_))
                result[sid] = info_
        return result

    def create_indexs(self, dataset, max_length):
        '''
        Create index dictionaries both for input ( words ) and output ( labels ) from given dataset .
        '''
        words = ['<PAD>', '<UNK>']
        prefixes = ['<PAD>', '<UNK>']
        suffixes = ['<PAD>', '<UNK>']
        labels = ['<PAD>']
        positions = ['<PAD>','<UNK>']
        for data in list(dataset.values()):
            pos = 0
            for w_pack in data:
                if w_pack[0] not in words:
                    words.append(w_pack[0])
                if w_pack[3] not in labels:
                    labels.append(w_pack[3])
                if w_pack[0][:3] not in prefixes:
                    prefixes.append(w_pack[0][:3])
                if w_pack[0][-3:] not in suffixes:
                    suffixes.append(w_pack[0][-3:])
                if pos not in positions:
                    positions.append(pos)
                pos+=1
        words = {k: v for v, k in enumerate(words)}
        labels = {k: v for v, k in enumerate(labels)}
        prefixes = {k: v for v, k in enumerate(prefixes)}
        suffixes = {k: v for v, k in enumerate(suffixes)}
        positions = {k: v for v, k in enumerate(positions)}
        result = {}
        result['words'] = words
        result['labels'] = labels
        result['maxlen'] = max_length
        result["pref"] = prefixes
        result["suff"] = suffixes
        result["position"] = positions
        return result

    def encode_words(self, dataset, idx):
        '''
        Encode the words in a sentence dataset formed by lists of tokens
        into lists of indexes suitable for NN input .

        The dataset encoded as a list of sentence , each of them is a list of
        word indices . If the word is not in the index , <UNK > code is used . If
        the sentence is shorter than max_len it is padded with <PAD > code .
        '''
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                if word[0] in idx["words"]:
                    index = idx["words"][word[0]]
                else:
                    index = idx["words"]['<UNK>']
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["words"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)

    def encode_positions(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            pos = 0
            for word in sentence:
                if pos in idx["position"]:
                    index = idx["position"][pos]
                else:
                    index = idx["position"]['<UNK>']
                encoded_sentence.append(index)
                pos+=1
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["position"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)
        
    def encode_prefixes(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                if word[0][:3] in idx["pref"]:
                    index = idx["pref"][word[0][:3]]
                else:
                    index = idx["pref"]['<UNK>']
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["pref"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)

    def encode_suffixes(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                if word[0][-3:] in idx["suff"]:
                    index = idx["suff"][word[0][-3:]]
                else:
                    index = idx["suff"]['<UNK>']
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["suff"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)

    def encode_labels(self, dataset, idx):
        '''
        Encode the ground truth labels in a sentence dataset formed by lists of
        tokens into lists of indexes suitable for NN output .
        '''
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                index = idx["labels"][word[3]]
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                index = idx["labels"]['<PAD>']
                encoded_sentence.append(index)
            results.append(np.array(encoded_sentence))
        n_tags = len(idx["labels"])
        results = [to_categorical(i, num_classes=n_tags) for i in results]
        results = np.array(results)
        print(results.shape)
        return results

    def save_model_and_indexs(self, model, idx, filename):
        '''
        Save given model and indexs to disk
        '''
        model.save_weights(filename + '.h5')
        with open(filename + '.idx', 'wb') as fp:
            pickle.dump(idx, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load_model_and_indexs(self, filename):
        '''
        Save given model and indexs to disk
        '''
        with open(filename + '.idx', 'rb') as fp:
            data = pickle.load(fp)
        n_words = len(data['words'])
        n_labels = len(data['labels'])
        max_len = data['maxlen']
        model = self.defineModel(n_words, n_labels, max_len)
        model.load_weights(filename + '.h5')
        return model, data

    def output_entities(self, dataset, preds, outfile):
        '''
        Output detected entities in the format expected by the evaluator
        '''
        # if it's not waiting will print the BI elements without the marks
        # in order to not print the O's or print together the BI
        wait = False  # while it's waiting will not print the elements
        name = ''
        off_start = '0'
        element = {'name': '', 'offset': '', 'type': ''}
        f = open(outfile, "w+")
        for i, (sid, sentence) in enumerate(dataset.items()):
            for ind, token in enumerate(sentence):
                curr = preds[i][ind]
                if curr == 'O' or curr=='<PAD>':  # if it's a O or <PAD> element, we do nothing
                    wait = True
                elif ind == (len(sentence) - 1):  # if it's the last element of the sentence
                    if curr.startswith('B'):
                        element = {'name': token[0],
                                   'offset': str(token[1]) + '-' + str(token[2]),
                                   'type': curr.split('-')[1]  # without B or I
                                   }
                    elif curr.startswith('I'):
                        name = token[0] if name is '' else name + ' ' + token[0]
                        element = {'name': name,
                                   'offset': off_start + '-' + str(token[2]),
                                   'type': curr.split('-')[1]
                                   }
                    else:  # only to check
                        print('There\'s something wrong')
                    wait = False

                else:
                    next = preds[i][ind+1]
                    if curr.startswith('B'):
                        if next.startswith('O') or next.startswith('B') or next.startswith('<'):
                            element = {'name': token[0],
                                       'offset': str(token[1]) + '-' + str(token[2]),
                                       'type': curr.split('-')[1]  # without B or I
                                       }
                            wait = False
                        elif next.startswith('I'):
                            name = token[0]
                            off_start = str(token[1])
                            wait = True
                    elif curr.startswith('I'):
                        if next.startswith('O') or next.startswith('B') or next.startswith('<'):
                            element = {'name': name + ' ' + token[0],
                                       'offset': off_start + '-' + str(token[2]),
                                       'type': curr.split('-')[1]
                                       }
                            if name == '':
                                element["name"] = token[0]
                            wait = False
                        elif next.startswith('I'):
                            name = token[0] if name is '' else name + ' ' + token[0]
                            wait = True
                    else:  # only to check
                        print('There\'s something wrong2')

                if not wait:
                    f.write(sid + '|' + element['offset'] + '|' + element['name'] + '|' + element['type'] + '\n')
        f.close()

    def predict(self, modelname, datadir, outfile):
        '''
        Loads a NN model from file ’modelname ’ and uses it to extract drugs
        in datadir . Saves results to ’outfile ’ in the appropriate format .
        '''
        print("[INFO]... Model in inference process")
        # load model and associated encoding data
        model, idx = self.load_model_and_indexs(modelname)
        # load data to annotate
        testdata = self.load_data(datadir)
        # encode dataset
        X = self.encode_words(testdata, idx)
        X_suff = self.encode_prefixes(testdata, idx)
        X_pref = self.encode_suffixes(testdata, idx)
        X_pos = self.encode_positions(testdata, idx)

        # tag sentences in dataset
        Y = model.predict([X, X_suff, X_pref, X_pos])
        reverse_labels= {y: x for x, y in idx['labels'].items()}
        Y = [[reverse_labels[np.argmax(y)] for y in s] for s in Y]
        # extract entities and dump them to output file
        self.output_entities(testdata, Y, outfile)

        # evaluate using official evaluator
        self.evaluation(datadir, outfile)

    def checkOutputs(self, modelname, datadir, outfile):
        print("[INFO]... Model in checking process")
        # load model and associated encoding data
        model, idx = self.load_model_and_indexs(modelname)
        # load data to annotate
        testdata = self.load_data(datadir)
        # encode dataset
        Y = self.encode_labels(testdata, idx)
        print(idx["labels"])
        reverse_labels = {y: x for x, y in idx['labels'].items()}
        Y = [[reverse_labels[np.argmax(y)] for y in s] for s in Y]
        # extract entities and dump them to output file
        self.output_entities(testdata, Y, outfile)

        # evaluate using official evaluator
        self.evaluation(datadir, outfile)

    def evaluation(self, datadir, outfile):
        evaluator.evaluate("NER", datadir, outfile)

    def learn(self, traindir, validationdir, modelname):
        '''
        Learns a NN model using traindir as training data , and validationdir
        as validation data . Saves learnt model in a file named modelname
        '''
        print("[INFO]... Model architecture in training process")

        # load train and validation data in a suitable form
        traindata = self.load_data(traindir)
        valdata = self.load_data(validationdir)

        # create indexes from training data
        max_len = 100
        idx = self.create_indexs(traindata, max_len)
        # build network
        model = self.build_network(idx)
        # encode datasets
        Xtrain = self.encode_words(traindata, idx)
        Xtrain_suff = self.encode_prefixes(traindata, idx)
        Xtrain_pref = self.encode_suffixes(traindata, idx)
        Xtrain_pos = self.encode_positions(traindata, idx)
        Ytrain = self.encode_labels(traindata, idx)
        
        Xval = self.encode_words(valdata, idx)
        Xval_suff = self.encode_prefixes(valdata, idx)
        Xval_pref = self.encode_suffixes(valdata, idx)
        Xval_pos = self.encode_positions(valdata, idx)
        Yval = self.encode_labels(valdata, idx)

        # train model

        # Saving the best model only
        filepath = modelname+"-{val_crf_viterbi_accuracy:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Fit the best model
        history = model.fit([Xtrain, Xtrain_suff, Xtrain_pref, Xtrain_pos], Ytrain, 
        validation_data=([Xval, Xval_suff, Xval_pref, Xval_pos], Yval), 
        batch_size=256, epochs=20, verbose=1, callbacks=callbacks_list)
        '''
        model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=256)
        '''
        # save model and indexs , for later use in prediction
        self.save_model_and_indexs(model, idx, modelname)

        self.plot(history)

    def plot(self, history):
        # Plot the graph
        plt.style.use('ggplot')

        accuracy = history.history['crf_viterbi_accuracy']
        val_accuracy = history.history['val_crf_viterbi_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(accuracy) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, accuracy, 'b', label='Training acc')
        plt.plot(x, val_accuracy, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig("History_model.jpg")



    def defineModel(self, n_words, n_labels, max_len):
        word_in = Input(shape=(max_len,))
        word_emb = Embedding(input_dim=n_words, output_dim=100, input_length=max_len, trainable=True)(word_in)  # 20-dim embedding
        
        suf_in = Input(shape=(max_len,))
        suf_emb = Embedding(input_dim=n_words, output_dim=100,
                        input_length=max_len)(suf_in)

        pref_in = Input(shape=(max_len,))
        pref_emb = Embedding(input_dim=n_words, output_dim=100,
                        input_length=max_len)(pref_in)

        pos_in = Input(shape=(max_len,))
        pos_emb = Embedding(input_dim=n_words, output_dim=100,
                        input_length=max_len)(pos_in)

        concat = concatenate([word_emb, suf_emb, pref_emb, pos_emb])
        model = Dropout(0.2)(concat)

        '''
        model = LSTM(units=max_len * 2,
                     return_sequences=True,
                     dropout=0.5,
                     recurrent_dropout=0.5,
                     kernel_initializer=k.initializers.he_normal())(model)
        '''

        model = Bidirectional(LSTM(units=32,return_sequences=True,recurrent_dropout=0.5,))(model)  # variational biLSTM
        model = TimeDistributed(Dense(n_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer

        crf = CRF(units=n_labels)  # CRF layer
        out = crf(model)  # output

        # create and compile model
        model = Model([word_in, suf_in, pref_in, pos_in], out)
        return model

    def build_network(self,idx):
        '''
        Create network for the learner
        '''
        # sizes
        n_words = len(idx['words'])
        n_labels = len(idx['labels'])
        max_len = idx['maxlen']
        # create network layers
        model = self.defineModel(n_words, n_labels, max_len)
        # set appropriate parameters (optimizer, loss, etc)
        model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy])
        model.summary()
        return model


if __name__ == '__main__':
    learner = Learner()
    learner.learn("../data/train", "../data/devel", "firstmodel")
    learner.predict("firstmodel", "../data/test", "results.txt")
