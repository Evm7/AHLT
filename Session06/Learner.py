import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
from numpy.random import seed
seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import string, sys
import numpy as np
import pickle

import matplotlib.pyplot as plt
import json, pathlib

import keras as k
from keras.callbacks import ModelCheckpoint
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed,concatenate,  Dropout, Bidirectional, Lambda, Layer, Conv1D, MaxPooling1D, Flatten
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

sys.path.append("../")
import evaluator

class Learner():
    def __init__(self):
        print("[WELCOME NEURAL NETWORKS DDI]... Init learning progress")
        # import nltk CoreNLP module (just once)


    def load_data(self, path):
        with open(path) as outfile:
            data = json.load(outfile)
        return data

    def create_indexs(self, dataset, max_length):
        '''
        Create index dictionaries both for input ( words ) and output ( labels ) from given dataset .
        '''
        words = ['<PAD>', '<UNK>']
        lemmas = ['<PAD>', '<UNK>']
        tags = ['<PAD>', '<UNK>']
        rel_pos1 = ['<PAD>', '<UNK>']
        rel_pos2 = ['<PAD>', '<UNK>']

        labels = []
        # create reverse index for words, lemmas and tags
        for data in dataset:
            dditype = data["dditype"]
            if dditype not in labels:
                labels.append(dditype)
            for word in data["feats"]:
                feat_word, feat_lemma, feat_tags, feat_pos1, feat_pos2 = word
                if feat_word not in words:
                    words.append(feat_word)
                if feat_lemma not in lemmas:
                    lemmas.append(feat_lemma)
                if feat_tags not in tags:
                    tags.append(feat_tags)
                if feat_pos1 not in rel_pos1:
                    rel_pos1.append(feat_pos1)
                if feat_pos2 not in rel_pos2:
                    rel_pos2.append(feat_pos2)
        words = {k: v for v, k in enumerate(words)}
        lemmas = {k: v for v, k in enumerate(lemmas)}
        tags = {k: v for v, k in enumerate(tags)}
        rel_pos1 = {k: v for v, k in enumerate(rel_pos1)}
        rel_pos2 = {k: v for v, k in enumerate(rel_pos2)}
        labels = {k: v for v, k in enumerate(labels)}

        result = {}
        result['words'] = words
        result['lemmas'] = lemmas
        result['tags'] = tags
        result['rel_pos1'] = rel_pos1
        result['rel_pos2'] = rel_pos2
        result['labels'] = labels
        result['maxlen'] = max_length
        return result

    def encode_words(self, dataset, idx):
        '''
        Encode the words in a sentence dataset formed by lists of tokens
        into lists of indexes suitable for NN input .

        The dataset encoded as a list of sentence , each of them is a list of
        word indices . If the word is not in the index , <UNK > code is used . If
        the sentence is shorter than max_len it is padded with <PAD > code .
        '''

        def getIndex(feat, index_specific):
            if feat in index_specific:
                index = index_specific[feat]
            else:
                index = index_specific['<UNK>']
            return index

        def pad(array, max_len, tag):
            while len(array) < max_len:
                array.append(tag)
            return array

        results_words = []
        results_lemmas = []
        results_tags = []
        results_pos1 = []
        results_pos2 = []

        for data in dataset:
            encoded_sentence_words = []
            encoded_sentence_lemmas = []
            encoded_sentence_tags = []
            encoded_sentence_pos_1 = []
            encoded_sentence_pos_2 = []
            for word in data['feats']:
                if len(word) != 5:
                    print(word)
                feat_word, feat_lemma, feat_tags, feat_pos_1, feat_pos_2 = word
                encoded_sentence_words.append(getIndex(feat_word, idx["words"]))
                encoded_sentence_lemmas.append(getIndex(feat_lemma, idx["lemmas"]))
                encoded_sentence_tags.append(getIndex(feat_tags, idx["tags"]))
                encoded_sentence_pos_1.append(getIndex(feat_pos_1, idx["rel_pos1"]))
                encoded_sentence_pos_2.append(getIndex(feat_pos_2, idx["rel_pos2"]))

            encoded_sentence_words = pad(encoded_sentence_words, idx["maxlen"], idx["words"]['<PAD>'])
            encoded_sentence_lemmas = pad(encoded_sentence_lemmas, idx["maxlen"], idx["lemmas"]['<PAD>'])
            encoded_sentence_tags = pad(encoded_sentence_tags, idx["maxlen"], idx["tags"]['<PAD>'])
            encoded_sentence_pos_1 = pad(encoded_sentence_pos_1, idx["maxlen"], idx["rel_pos1"]['<PAD>'])
            encoded_sentence_pos_2 = pad(encoded_sentence_pos_2, idx["maxlen"], idx["rel_pos2"]['<PAD>'])

            results_words.append(np.array(encoded_sentence_words))
            results_lemmas.append(np.array(encoded_sentence_lemmas))
            results_tags.append(np.array(encoded_sentence_tags))
            results_pos1.append(np.array(encoded_sentence_pos_1))
            results_pos2.append(np.array(encoded_sentence_pos_2))

        return [np.array(results_words), np.array(results_lemmas), np.array(results_tags), np.array(results_pos1), np.array(results_pos2)]

    def encode_labels(self, dataset, idx):
        '''
        Encode the ground truth labels in a sentence dataset formed by lists of
        tokens into lists of indexes suitable for NN output .
        '''
        results = []
        for data in dataset:
            index = idx["labels"][data['dditype']]
            results.append(np.array(index))
        n_tags = len(idx["labels"])

        def to_categorical(y, num_classes):
            return np.eye(num_classes)[y]

        results = [to_categorical(i, num_classes=n_tags) for i in results]
        results = np.array(results)
        return results

    def save_model_and_indexs(self, model, idx, filename):
        '''
        Save given model and indexs to disk
        '''
        model.save_weights(filename + '.hdf5')
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
        model.load_weights(filename + '-0.886.hdf5')
        return model, data

    def output_interactions(self, dataset, preds, outfile):
        '''
        Output detected entities in the format expected by the evaluator
        '''
        # if it's not waiting will print the BI elements without the marks
        # in order to not print the O's or print together the BI
        f = open(outfile, "w+")

        for i, pair in enumerate(dataset):
            sid= pair['sid']
            e1 = pair['id_e1']
            e2 = pair['id_e2']
            type_pred = preds[i]
            if type_pred != "null":
                f.write(sid + '|' + e1 + '|' +e2 + '|' + type_pred + '\n')
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
        filename = "features_"+ pathlib.Path(datadir).stem+".txt"
        testdata = self.load_data(filename)
        # encode dataset
        X = self.encode_words(testdata, idx)

        # tag sentences in dataset
        Y = model.predict(X)
        #[s.numpy().argmax() for s in y_val_pred]
        reverse_labels= {y: x for x, y in idx['labels'].items()}
        Y = [reverse_labels[np.argmax(s)] for s in Y]
        # extract entities and dump them to output file
        self.output_interactions(testdata, Y, outfile)

        # evaluate using official evaluator
        self.evaluation(datadir, outfile)

    def checkOutputs(self, modelname, datadir, outfile):
        print("[INFO]... Model in checking process")
        # load model and associated encoding data
        model, idx = self.load_model_and_indexs(modelname)
        # load data to annotate
        filename = "features_"+ pathlib.Path(datadir).stem+".txt"
        testdata = self.load_data(filename)
        # encode dataset
        Y = self.encode_labels(testdata, idx)
        reverse_labels = {y: x for x, y in idx['labels'].items()}
        Y = [reverse_labels[np.argmax(s)] for s in Y]

        # extract entities and dump them to output file
        self.output_interactions(testdata, Y, outfile)

        # evaluate using official evaluator
        self.evaluation(datadir, outfile)

    def evaluation(self, datadir, outfile):
        evaluator.evaluate("DDI", datadir, outfile)

    def learn(self, traindir, validationdir, modelname, finetune=""):
        '''
        Learns a NN model using traindir as training data , and validationdir
        as validation data . Saves learnt model in a file named modelname
        '''
        print("[INFO]... Model architecture in training process")

        # load train and validation data in a suitable form
        traindata = self.load_data(traindir)
        valdata = self.load_data(validationdir)

        if finetune=="":
            # create indexes from training data
            max_len = 200
            idx = self.create_indexs(traindata, max_len)

            # build network
            model = self.build_network(idx)
        else:
            model, idx = self.load_model_and_indexs(finetune)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        # encode datasets
        Xtrain = self.encode_words(traindata, idx)
        Ytrain = self.encode_labels(traindata, idx)
        Xval = self.encode_words(valdata, idx)
        Yval = self.encode_labels(valdata, idx)


        # train model
        # Saving the best model only
        filepath = modelname+"-{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]

        # Fit the best model
        history = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=256, epochs=15, verbose=1, callbacks=callbacks_list)

        # save model and indexs , for later use in prediction
        self.save_model_and_indexs(model, idx, modelname)

        self.plot(history)

    def plot(self, history):
        # Plot the graph
        print(history.history)
        plt.style.use('ggplot')
        accuracy = history.history['acc']
        val_accuracy = history.history['val_acc']
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



    def defineModel(self, input_dim, n_labels, max_len):
        word_in = Input(shape=(max_len,))
        word_emb = Embedding(input_dim=input_dim, output_dim=100, input_length=max_len, trainable=True)(
            word_in)  # 20-dim embedding

        lemma_in = Input(shape=(max_len,))
        lemma_emb = Embedding(input_dim=input_dim, output_dim=100, input_length=max_len)(word_in)

        tags_in = Input(shape=(max_len,))
        tags_emb = Embedding(input_dim=input_dim, output_dim=100, input_length=max_len)(lemma_in)

        pos1_in = Input(shape=(max_len,))
        pos1_emb = Embedding(input_dim=input_dim, output_dim=100, input_length=max_len)(tags_in)

        pos2_in = Input(shape=(max_len,))
        pos2_emb = Embedding(input_dim=input_dim, output_dim=100, input_length=max_len)(pos1_in)

        concat = concatenate([word_emb, lemma_emb, tags_emb, pos1_emb, pos2_emb])
        model = Dropout(0.55)(concat)
        model = Conv1D(128, 5, padding='same', activation='relu')(model)
        model = MaxPooling1D(pool_size=2)(model)
        model = Bidirectional(LSTM(units=100,return_sequences=False,recurrent_dropout=0.3,))(model)  # variational biLSTM
        #model = Flatten()(model))
        #model = LSTM(100)(model)
        out = Dense(n_labels, activation='softmax')(model)

        #out = TimeDistributed(Dense(n_labels, activation="softmax"))(model)  # a dense layer as suggested by neuralNer


        # create and compile model
        model = Model([word_in, lemma_in, tags_in, pos1_in, pos2_in], out)
        return model

    def build_network(self,idx):
        '''
        Create network for the learner
        '''
        # sizes
        n_words = len(idx['words'])
        n_lemmas = len(idx['lemmas'])
        n_tags = len(idx['tags'])
        n_pos1 = len(idx['rel_pos1'])
        n_pos2 = len(idx['rel_pos2'])
        input_dim = max(n_words, n_lemmas, n_tags, n_pos1, n_pos2)

        n_labels = len(idx['labels'])
        max_len = idx['maxlen']

        # create network layers
        model = self.defineModel(input_dim, n_labels, max_len)

        # set appropriate parameters (optimizer, loss, etc)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        return model


if __name__ == '__main__':
    learner = Learner()
    learner.learn("features_train.txt", "features_devel.txt", "definitive")
    #learner.learn("features_train.txt", "features_devel.txt", "adecuate_2",  "adecuate")
    #learner.checkOutputs("original", "../data/train", "results.txt")
    #learner.predict("adecuate_2", "../data/test", "results.txt")
