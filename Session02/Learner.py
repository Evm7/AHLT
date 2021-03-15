import sys
import argparse
from os import listdir
import string
import pandas as pd

sys.path.append("../")

import evaluator

from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite



class Learner():
    def __init__(self):
        args = self.parse_arguments()

        self.train_file = args["train"]
        self.val_file = args["val"]
        self.test_file = args["test"]

        self.outfile_name = args["outfile"]
        self.external = args["external"]
        self.evaluator = args["evaluate"]

        self.f = open(self.outfile_name, "w+")

        print("Starting to train a learner saved in " + self.outfile_name)
        self.extract_dict()

    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-train', '--train', type=str, default="train.feat", help='Name for the training feature file')
        parser.add_argument('-val', '--val', type=str, default="devel.feat", help='Name for the validation feature file')
        parser.add_argument('-test', '--test', type=str, default="test.feat", help='Name for the validation feature file')

        parser.add_argument('-outfile', '--outfile', type=str, default="results.out", help='Name for the output file')
        parser.add_argument('--external', action="store_false", default=True, help='Whether to use external resources or not')
        parser.add_argument('-evaluate', '--evaluate', type=str, default="test", help='Evaluating over the testing dataset')


        args = vars(parser.parse_args())
        return args

    def extract_dict(self):
        if self.external:
            # Loading DrugBank.txt
            with open("../resources/DrugBank.txt", 'r', encoding='utf8') as doc:
                document = doc.readlines()

            self.drugbank_dict = {}
            for d in document:
                sep = d.rsplit('|', 1)
                self.drugbank_dict[sep[0]] = sep[-1].rstrip()

            # Loading HSDB.txt
            with open("../resources/HSDB.txt", 'r', encoding='utf8') as doc:
                document = doc.readlines()

            self.HSDB = []
            for d in document:
                self.HSDB.append(d.rstrip().lower())

    def sent2features(self, sent):
        if len(sent)>3:
            return [sent[5:]]
        return []


    def sent2labels(self, sent):
        if len(sent)>3:
            return [sent[4]]
        return []


    def sent2tokens(self, sent):
        if len(sent)>3:
            return [sent[:4]]
        return []

    # EVALUATE THE MODEL
    def bio_classification_report(self, y_true, y_pred):
        """
        Classification report for a list of BIO-encoded sequences.
        It computes token-level metrics and discards "O" labels.

        Note that it requires scikit-learn 0.15+ (or a version from github master)
        to calculate averages properly!
        """
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = set(lb.classes_) - {'O'}
        tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
        print(tagset)
        class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels=[class_indices[cls] for cls in tagset],
            target_names=tagset,
        )

    def print_transitions(self, trans_features):
        for (label_from, label_to), weight in trans_features:
            print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

    def print_state_features(self, state_features):
        for (attr, label), weight in state_features:
            print("%0.6f %-6s %s" % (weight, label, attr))

    def checkKnoweledge(self, tagger):
        # let's check what classifier learned
        from collections import Counter
        info = tagger.info()

        print("Top likely transitions:")
        self.print_transitions(Counter(info.transitions).most_common(15))

        print("\nTop unlikely transitions:")
        self.print_transitions(Counter(info.transitions).most_common()[-15:])

        print("Top positive:")
        self.print_state_features(Counter(info.state_features).most_common(20))

        print("\nTop negative:")
        self.print_state_features(Counter(info.state_features).most_common()[-20:])

    def readFile(self, filename):
        with open(filename) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip().split("\t")for x in content]
        return content

    def createDataset(self):
        self.train_sents = self.readFile(self.train_file)
        self.val_sents = self.readFile(self.val_file)
        self.test_sents = self.readFile(self.test_file)


        self.X_train = [self.sent2features(s) for s in self.train_sents]
        self.y_train = [self.sent2labels(s) for s in self.train_sents]

        self.X_val = [self.sent2features(s) for s in self.val_sents]
        self.y_val = [self.sent2labels(s) for s in self.val_sents]

        self.X_test = [self.sent2features(s) for s in self.test_sents]
        self.y_test = [self.sent2labels(s) for s in self.test_sents]


    def train(self):
        self.createDataset()
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(self.X_train, self.y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 0.2,  # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            # as we have a considerable quantity of features to train from
            # the training should be longer
            'max_iterations': 50,
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        from datetime import date
        today = date.today()
        # dd/mm/YY
        d1 = today.strftime("%d_%m_%Y")
        trainer.train('pycrfsuite_'+str(d1))

    def classify(self):
        # MAKE PREDICTIONS
        tagger = pycrfsuite.Tagger()
        tagger.open('pycrfsuite_15_03_2021')
        self.createDataset()
        if self.evaluator is "val":
            print("Evaluating the classifier over the validation dataset")
            y_pred = [tagger.tag(xseq) for xseq in self.X_val]
            #print(self.bio_classification_report(self.y_val, y_pred))
            #self.checkKnoweledge(tagger)
            self.output_results(y_pred, self.val_sents)
            self.f.close()
            # print performance score
            evaluator.evaluate("NER", "../data/devel", self.outfile_name)
        elif self.evaluator is "train":
            print("Evaluating the classifier over the training dataset")
            y_pred = [tagger.tag(xseq) for xseq in self.X_train]
            #print(self.bio_classification_report(self.y_train, y_pred))
            #self.checkKnoweledge(tagger)
            self.output_results(y_pred, self.train_sents)
            self.f.close()
            # print performance score
            evaluator.evaluate("NER", "../data/train", self.outfile_name)

        else:
            print("Evaluating the classifier over the testing dataset")
            y_pred = [tagger.tag(xseq) for xseq in self.X_test]
            #print(self.bio_classification_report(self.y_train, y_pred))
            #self.checkKnoweledge(tagger)
            self.output_results(y_pred, self.test_sents)
            self.f.close()
            # print performance score
            evaluator.evaluate("NER", "../data/test", self.outfile_name)

    def output_results(self, y_pred, sentences_file):
        self.tokens = [self.sent2tokens(s) for s in sentences_file]
        for index, pred in enumerate(y_pred):
            if len(pred)>0:
                tag = pred[0]
                if 'O' not in tag:
                    tag = tag.split("-")[1]
                    if len(self.tokens[index])>0:
                        e = self.createMap(self.tokens[index][0], tag)
                        print(e["id"] + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"], file=self.f)

    def createMap(self, token, type):
        return {'id': token[0], 'name': token[1], 'offset': str(token[2])+"-" +str(token[3]), 'type': type}

    def output_entities (self, sid , tokens , tags ) :
        '''
        Task :
        Given a list of tokens and the B-I-O tag for each token , produce a list
        of drugs in the format expected by the evaluator .
        Input :
        sid : sentence identifier ( required by the evaluator output format )
        tokens : List of tokens in the sentence , i.e. list of tuples (word ,
        offsetFrom , offsetTo )
        tags : List of B-I-O tags for each token
        Output :
        Prints to stdout the entities in the right format : one line per entity ,
        fields separated by '|', field order : id , offset , name , type .
        Example :
        output_entities (" DDI - DrugBank . d553 .s0",
        [(" Ascorbic " ,0 ,7) , (" acid " ,9 ,12) , (" ," ,13 ,13) ,
        (" aspirin " ,15 ,21) , (" ," ,22 ,22) , (" and " ,24 ,26) ,
        (" the " ,28 ,30) ,(" common " ,32 ,37) , (" cold " ,39 ,42) ],
        ["B- drug ", "I- drug ", "O", "B- brand ", "O", "O", "O",
        "O", "O "])
        DDI - DrugBank . d553 .s0 |0 -12| Ascorbic acid | drug
        DDI - DrugBank . d553 .s0 |15 -21| aspirin | brand
        '''
        # if it's not waiting will print the BI elements without the marks
        # in order to not print the O's or print together the BI
        wait = False  # while it's waiting will not print the elements
        name = ''
        off_start = '0'
        element = {'name': '', 'offset': '', 'type': ''}
        for ind, token in enumerate(tokens):
            if tags[ind] == 'O':  # if it's a O element, we do nothing
                wait = True
            elif ind == len(tokens) - 1:  # if it's the last element of the sentence
                if tags[ind].startswith('B'):
                    element = {'name': token[0],
                               'offset': str(token[1]) + '-' + str(token[2]),
                               'type': tags[ind].split('-')[1]  # without B or I
                               }
                elif tags[ind].startswith('I'):
                    element = {'name': name + ' ' + token[0],
                               'offset': off_start + '-' + str(token[2]),
                               'type': tags[ind].split('-')[1]
                               }
                    if name == '' and ind != len(tokens) - 1:
                        element["name"] = token[0]
                    wait = False
                else:  # only to check
                    print('There\'s something wrong')
                wait = False

            else:
                if ((tags[ind].startswith('B') and tags[ind + 1].startswith('O')) or
                        (tags[ind].startswith('B') and tags[ind + 1].startswith('B'))):
                    element = {'name': token[0],
                               'offset': str(token[1]) + '-' + str(token[2]),
                               'type': tags[ind].split('-')[1]  # without B or I
                               }
                    wait = False
                elif tags[ind].startswith('B') and tags[ind + 1].startswith('I'):
                    name = token[0]
                    off_start = str(token[1])
                    wait = True
                elif ((tags[ind].startswith('I') and tags[ind + 1].startswith('O')) or
                      (tags[ind].startswith('I') and tags[ind + 1].startswith('B'))):
                    element = {'name': name + ' ' + token[0],
                               'offset': off_start + '-' + str(token[2]),
                               'type': tags[ind].split('-')[1]
                               }
                    if name == '':
                        element["name"] = token[0]
                    wait = False
                elif tags[ind].startswith('I') and tags[ind + 1].startswith('I'):
                    if name == '':
                        name = token[0]
                    else:
                        name = name + ' ' + token[0]
                    wait = True
                else:  # only to check
                    print('There\'s something wrong2')

            if not wait:
                self.f.write(sid + '|' + element['offset'] + '|' + element['name'] + '|' + element['type'] + '\n')
        self.f.close()

if __name__ == '__main__':
    learner = Learner()
    #learner.train()
    learner.classify()
