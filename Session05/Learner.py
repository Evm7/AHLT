from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import nltk
from nltk.corpus import stopwords
from xml.dom.minidom import parse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import keras as k

from numpy.random import seed

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
        prevword = ['<PAD>','<UNK>']
        nextword = ['<PAD>','<UNK>']
        class_suffixes = ['<PAD>', 'brand', 'drug', 'drug_n', 'group', 'none']
        for data in list(dataset.values()):
            pos = 0
            w_pack_prev = '<START>'
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
                if w_pack_prev not in prevword:
                    prevword.append(w_pack_prev)
                if w_pack[0] not in nextword:
                    nextword.append(w_pack[0])
                w_pack_prev = w_pack[0]
                pos+=1
            if '<END>' not in nextword:
                nextword.append('<END>')
        words = {k: v for v, k in enumerate(words)}
        labels = {k: v for v, k in enumerate(labels)}
        prefixes = {k: v for v, k in enumerate(prefixes)}
        suffixes = {k: v for v, k in enumerate(suffixes)}
        positions = {k: v for v, k in enumerate(positions)}
        prevword = {k: v for v, k in enumerate(prevword)}
        nextword = {k: v for v, k in enumerate(nextword)}
        class_suffixes = {k: v for v, k in enumerate(class_suffixes)}

        result = {}
        result['words'] = words
        result['labels'] = labels        
        result['maxlen'] = max_length
        result['prev'] = prevword
        result['next'] = nextword
        result["pref"] = prefixes
        result["suff"] = suffixes
        result["position"] = positions
        result["class_suffixes"] = class_suffixes
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
        
    def encode_words_lower(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                if word[0].lower() in idx["words_lower"]:
                    index = idx["words_lower"][word[0].lower()]
                else:
                    index = idx["words_lower"]['<UNK>']
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["words_lower"]['<PAD>'])
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

    def encode_prevwords(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            prevword = '<START>'
            for word in sentence:
                if prevword in idx["prev"]:
                    index = idx["prev"][prevword]
                else:
                    index = idx["prev"]['<UNK>']
                encoded_sentence.append(index)
                prevword=word[0]
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["prev"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)

    def encode_nextwords(self, dataset, idx):
        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for i in range(len(sentence)-1):
                if sentence[i+1][0] in idx["next"]:
                    index = idx["next"][sentence[i+1][0]]
                else:
                    index = idx["next"]['<UNK>']
                encoded_sentence.append(index)
            index = idx["next"]['<END>']
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["next"]['<PAD>'])
            results.append(np.array(encoded_sentence))
        return np.array(results)


    def check_Prefixes(self, tok, pref):
        for p in pref:
            if str(tok).lower().startswith(p):
                return True
        return False

    def check_Suffixes(self, tok, pref):
        for p in pref:
            if str(tok).endswith(p):
                return True
        return False

    def check_contains(self, tok, cont):
        for p in cont:
            if p in str(tok):
                return True
        return False

    def encode_class_suffixes(self, dataset, idx):
        
        suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
        suffixes_drug = ["ine", "cin", "ium", "vir","ide", "lam", "il", "ril", "cin", "tin"]
        #suffixes_brand = ["gen"]
        suffixes_brand = []
        suffixes_group = ["ines", "ides", "cins", "oles"]

        prefixes_drug_n = ['ibog', 'endo', "bombe", "contor", "dmp", "egf", "ginse", "heo", "ibo", "jac", "phen"]
        #prefixes_brand = ["SPR", "Acc", "equ", "EQU"]
        prefixes_brand = []
        prefixes_group = ["beta-adre", "hmg", "monoamine", "calcium", "drugs", "sali", "quino", "ssri", "cepha", "sulfo", "TCA", "thiaz", "benzo", "barb", "contracept", "cortico", "digitalis", "diu", "central", "nervous", "system", "beta", "psycho", "cepha", "macro", "prot", "ace", "mao", "cardiac"]
        prefixes_drug = ['digox', 'warfa', 'meth', 'theophy', 'lith', 'keto', 'cime', 'insu', 'fluox', 'alcoh', 'cyclos', 'eryth', 'carba', 'rifa', 'caffe']

        contains_drug_n = ["MHD", "NaC", "MC", "gaine", "PTX", "PCP"]
        contains_group = ["ids", "urea" ]
        contains_brand = ["PEGA", "aspirin", "Aspirin", "XX", "IVA"]
        '''
        suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
        suffixes_drug = ["ine", "cin", "ium"]
        suffixes_brand = ["gen"]
        suffixes_group = ["ines", "ides", "cins", "oles"]
        '''

        results = []
        for sentence in dataset.values():
            encoded_sentence = []
            for word in sentence:
                token = word[0]
                if token.isupper() or self.check_contains(token, contains_brand):
                    index = idx["class_suffixes"]['brand']
                elif self.check_Suffixes(token, suffixes_drug) or self.check_Suffixes(token, suffixes) or self.check_Prefixes(token, prefixes_drug):
                    index = idx["class_suffixes"]['drug']
                elif self.check_Suffixes(token, suffixes_group) or "agent" in token or self.check_Prefixes(token, prefixes_group) or self.check_contains(token, contains_group):
                    index = idx["class_suffixes"]['group']
                elif self.check_Prefixes(token, prefixes_drug_n) or self.check_contains(token, contains_drug_n):
                    index = idx["class_suffixes"]['drug_n']
                else:
                    index = idx["class_suffixes"]['none']
                encoded_sentence.append(index)
            while len(encoded_sentence) < idx["maxlen"]:
                encoded_sentence.append(idx["class_suffixes"]['<PAD>'])
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

    def load_model_and_indexs(self, filename, embedding_matrix):
        '''
        Save given model and indexs to disk
        '''
        with open(filename + '.idx', 'rb') as fp:
            data = pickle.load(fp)
        n_words = len(data['words'])
        n_labels = len(data['labels'])
        max_len = data['maxlen']
        
        n_prev = len(data['prev'])
        n_next = len(data['next'])
        n_pref = len(data["pref"])
        n_suff = len(data["suff"])
        n_pos = len(data["position"])
        n_class = len(data["class_suffixes"])

        numbers=[n_words, n_suff, n_pref,n_pos,n_prev, n_next, n_class]

        model = self.defineModel(numbers, n_labels, max_len, embedding_matrix)
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

    def predict(self, modelname, datadir, outfile, embedding_matrix):
        '''
        Loads a NN model from file ’modelname ’ and uses it to extract drugs
        in datadir . Saves results to ’outfile ’ in the appropriate format .
        '''
        print("[INFO]... Model in inference process")
        # load model and associated encoding data
        model, idx = self.load_model_and_indexs(modelname, embedding_matrix)
        # load data to annotate
        testdata = self.load_data(datadir)
        # encode dataset
        X = self.encode_words(testdata, idx)
        X_pref = self.encode_prefixes(testdata, idx)
        X_suff = self.encode_suffixes(testdata, idx)
        X_pos = self.encode_positions(testdata, idx)
        X_prev = self.encode_prevwords(testdata, idx)
        X_next = self.encode_nextwords(testdata, idx)
        X_class_suff = self.encode_class_suffixes(testdata, idx)

        # tag sentences in dataset
        Y = model.predict([X, X_suff, X_pref, X_pos, X_prev, X_next, X_class_suff])
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

        # encode datasets
        Xtrain = self.encode_words(traindata, idx)
        Xtrain_pref = self.encode_prefixes(traindata, idx)
        Xtrain_suff = self.encode_suffixes(traindata, idx)
        Xtrain_pos = self.encode_positions(traindata, idx)
        Xtrain_prev = self.encode_prevwords(traindata, idx)
        Xtrain_next = self.encode_nextwords(traindata, idx)
        Xtrain_class_suff = self.encode_class_suffixes(traindata, idx)
        Ytrain = self.encode_labels(traindata, idx)

        Xval = self.encode_words(valdata, idx)
        Xval_pref = self.encode_prefixes(valdata, idx)
        Xval_suff = self.encode_suffixes(valdata, idx)
        Xval_pos = self.encode_positions(valdata, idx)
        Xval_prev = self.encode_prevwords(valdata, idx)
        Xval_next = self.encode_nextwords(valdata, idx)
        Xval_class_suff = self.encode_class_suffixes(valdata, idx)
        Yval = self.encode_labels(valdata, idx)
        
        n_words=len(idx['words'])

        # load the whole embedding into memory
        embeddings_index = dict()
        f = open('../data/glove.6B/glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((n_words, 100))
        h=0
        for word in idx['words']:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[h] = embedding_vector
            h+=1
        # train model
        # build network
        model = self.build_network(idx, embedding_matrix)

        # Saving the best model only
        filepath = modelname+"-{val_crf_viterbi_accuracy:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_crf_viterbi_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Fit the best model
        history = model.fit([Xtrain, Xtrain_suff, Xtrain_pref, Xtrain_pos, Xtrain_prev, Xtrain_next, Xtrain_class_suff], Ytrain, validation_data=([Xval, Xval_suff, Xval_pref, Xval_pos, Xval_prev, Xval_next, Xval_class_suff], Yval), batch_size=256, epochs=20, verbose=1, callbacks=callbacks_list)
        '''
        model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=256)
        '''
        # save model and indexs , for later use in prediction
        self.save_model_and_indexs(model, idx, modelname)

        self.plot(history)
        return embedding_matrix

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

    def defineModel(self, numbers, n_labels, max_len, embedding_matrix):
        word_in = Input(shape=(max_len,))
        word_emb = Embedding(input_dim=numbers[0], output_dim=100, input_length=max_len, trainable=False, weights = [embedding_matrix])(word_in)  # 20-dim embedding
        
        suf_in = Input(shape=(max_len,))
        suf_emb = Embedding(input_dim=numbers[1], output_dim=100,
                        input_length=max_len)(suf_in)

        pref_in = Input(shape=(max_len,))
        pref_emb = Embedding(input_dim=numbers[2], output_dim=100,
                        input_length=max_len)(pref_in)

        pos_in = Input(shape=(max_len,))
        pos_emb = Embedding(input_dim=numbers[3], output_dim=100,
                        input_length=max_len)(pos_in)

        prev_in = Input(shape=(max_len,))
        prev_emb = Embedding(input_dim=numbers[4], output_dim=100,
                        input_length=max_len)(prev_in)

        next_in = Input(shape=(max_len,))
        next_emb = Embedding(input_dim=numbers[5], output_dim=100,
                        input_length=max_len)(next_in)

        class_suff_in = Input(shape=(max_len,))
        class_suff_emb = Embedding(input_dim=numbers[6], output_dim=100,
                        input_length=max_len)(class_suff_in)

        concat = concatenate([word_emb, suf_emb, pref_emb, pos_emb, prev_emb, next_emb, class_suff_emb])
        model = Dropout(0.2)(concat)

        '''
        model = LSTM(units=max_len * 2,
                     return_sequences=True,
                     dropout=0.5,
                     recurrent_dropout=0.5,
                     kernel_initializer=k.initializers.he_normal())(model)
        '''

        model = Bidirectional(LSTM(units=32,return_sequences=True,recurrent_dropout=0.5,))(model)  # variational biLSTM
        #model = Bidirectional(LSTM(units=32,return_sequences=True,recurrent_dropout=0.5,))(model)  # variational biLSTM
        #model = Bidirectional(LSTM(units=32,return_sequences=True,recurrent_dropout=0.5,))(model)  # variational biLSTM
        model = TimeDistributed(Dense(n_labels, activation="relu"))(model)  # a dense layer as suggested by neuralNer

        crf = CRF(units=n_labels)  # CRF layer
        out = crf(model)  # output

        # create and compile model
        model = Model([word_in, suf_in, pref_in, pos_in, prev_in, next_in, class_suff_in], out)
        return model

    def build_network(self,idx, embedding_matrix):
        from keras.optimizers import RMSprop
        '''
        Create network for the learner
        '''
        # sizes
        n_words = len(idx['words'])
        n_prev = len(idx['prev'])
        n_next = len(idx['next'])
        n_pref = len(idx["pref"])
        n_suff = len(idx["suff"])
        n_pos = len(idx["position"])
        n_labels = len(idx['labels'])
        n_class = len(idx["class_suffixes"])

        numbers=[n_words, n_suff, n_pref,n_pos,n_prev, n_next, n_class]

        max_len = idx['maxlen']
        # create network layers
        model = self.defineModel(numbers, n_labels, max_len, embedding_matrix)
        # set appropriate parameters (optimizer, loss, etc)
        optimizer = RMSprop(lr=0.001, epsilon=None, decay=0.0)

        crf = CRF(n_labels, activation='linear')  # CRF layer
        model.compile(optimizer=optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
        model.summary()
        return model


if __name__ == '__main__':
    learner = Learner()
    emb_matrix = learner.learn("../data/train", "../data/devel", "firstmodel")
    print("TRAIN")
    learner.predict("firstmodel", "../data/train", "results.txt", emb_matrix)
    print("\nDEVEL")
    learner.predict("firstmodel", "../data/devel", "results.txt", emb_matrix)
    print("\nTEST")
    learner.predict("firstmodel", "../data/test", "results.txt", emb_matrix)


