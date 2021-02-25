#! /usr/bin/python3

import sys
import argparse
from os import listdir
import string
import pandas as pd


#import evaluator

from xml.dom.minidom import parse

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class BaselineNer():
    def __init__(self):
        args = self.parse_arguments()

        self.datadir = args["datadir"]
        self.outfile_name = args["outfile"]
        self.external = args["external"]

        self.f = open(self.outfile_name, "w+")

        self.stopwords = set(stopwords.words('english'))

        self.suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
        self.suffixes_drug = ["ine", "cin", "ium", "vir","ide", "lam", "il", "ril", "cin", "tin"]
        self.suffixes_brand = ["gen"]
        self.suffixes_group = ["ines", "ides", "cins", "oles"]

        self.prefixes_drug = ["bombe", "contor", "dmp", "egf", "ginse", "heo", "ibo", "jac", "phen"]
        self.prefixes_brand = ["beta", "psycho", "cepha", "macro", "prot", "ace", "mao", "cardiac", "SPR", "acc", "equ"]
        self.prefixes_group = ["beta-adre", "hmg", "monoamine", "calcium", "drugs", "sali", "quino", "ssri", "cepha", "sulfo", "TCA", "thiaz", "benzo", "barb", "contracept", "cortico", "digitalis", "diu"]

        print("Starting the process of directory " + self.datadir + " saved in " + self.outfile_name)

        self.extract_dict()

    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', type=str, default="data/devel/", help='Directory with XML files to process')
        parser.add_argument('-outfile', '--outfile', type=str, default="result.out", help='Name for the output file')
        parser.add_argument('--external', action="store_false", default=True, help='Whether to use external resources or not')

        args = vars(parser.parse_args())
        return args

    def extract_dict(self):
        if self.external:
            # Loading DrugBank.txt
            with open("resources/DrugBank.txt", 'r', encoding='utf8') as doc:
                document = doc.readlines()

            self.drugbank_dict = {}
            for d in document:
                sep = d.rsplit('|', 1)
                self.drugbank_dict[sep[0]] = sep[-1].rstrip()

            # Loading HSDB.txt
            with open("resources/HSDB.txt", 'r', encoding='utf8') as doc:
                document = doc.readlines()

            self.HSDB = []
            for d in document:
                self.HSDB.append(d.rstrip().lower())


    def tokenize(self, sentence):
        '''
        Task :
        Given a sentence , calls nltk . tokenize to split it in
        tokens , and adds to each token its start / end offset
        in the original sentence .
        Input :
        s: string containing the text for one sentence
        Output :
        Returns a list of tuples (word , offsetFrom , offsetTo )
        Example :
        tokenize (" Ascorbic acid , aspirin , and the commoncold .")
        [(" Ascorbic " ,0 ,7) , (" acid " ,9 ,12) , (" ," ,13 ,13) , ("
        aspirin " ,15 ,21) , (" ," ,22 ,22) , (" and " ,24 ,26) , (" the
        " ,28 ,30) , (" common " ,32 ,37) , (" cold " ,39 ,42) ,
        ("." ,43 ,43) ]
        '''
        tokens = []
        offset = 0

        words = word_tokenize(sentence)

        for w in words:
            if (w in self.stopwords) or (w in string.punctuation):
                continue
            offset = sentence.find(w, offset)
            tokens.append((w, offset, offset+len(w)-1))
            offset+=len(w)+1

        print(tokens, flush=True)
        return tokens

    def extract_entities(self, tokens):
        '''
        Task :
        Given a tokenized sentence , identify which tokens (or groups of
        consecutive tokens ) are drugs
        Input :
            s: A tokenized sentence ( list of triples (word , offsetFrom , offsetTo ) )
        Output :
            A list of entities . Each entity is a dictionary with the keys 'name ', ' offset ', and 'type '.
        Example :
            extract_entities ([(" Ascorbic " ,0 ,7) , (" acid " ,9 ,12) , (" ," ,13 ,13) , ("
            aspirin " ,15 ,21) , (" ," ,22 ,22) , (" and " ,24 ,26) , (" the " ,28 ,30) , (" common
            " ,32 ,37) , (" cold " ,39 ,42) , ("." ,43 ,43) ])
            [{" name ":" Ascorbic acid ", " offset ":"0 -12" , " type ":" drug "},
            {" name ":" aspirin ", " offset ":"15 -21" , " type ":" brand "}]
        '''
        entities = []

        for tok in tokens:
            isdrug, type = self.apply_rules(tok[0])
            if isdrug:
                entities.append(self.createMap(tok, type))

        return entities

    def createMap(self, token, type):
        return  {'name': token[0], 'offset': str(token[1])+" -" +str(token[2]), 'type': type}

    def check_Prefixes(self, tok, pref):
        for p in pref:
            if str(tok).lower().startswith(p):
                return True
        return False

    def apply_rules(self, token):
        if token[-5:] in self.suffixes or token[-3:] in self.suffixes_drug or self.check_Prefixes(token, self.prefixes_drug) or token.lower() in self.drugbank_dict or token.lower() in self.HSDB:
            return True, "drug"
        elif token.isupper() or token[-3:] in self.suffixes_brand or self.check_Prefixes(token, self.prefixes_brand):
            return True, "brand"
        elif token[-4:] in self.suffixes_group or "agent" in token or self.check_Prefixes(token, self.prefixes_group):
            return True, "group"
        else:
            return False, ""

    def processFile(self):
        # process each file in directory
        for f in listdir(self.datadir):
            # parse XML file , obtaining a DOM tree
            tree = parse(self.datadir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text
                print(stext)
                # tokenize text
                tokens = self.tokenize(stext)
                # extract entities from tokenized sentence text
                entities = self.extract_entities(tokens)

                # print sentence entities in format requested for evaluation
                for e in entities:
                    print(sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"], file = self.f)
        # print performance score
        #evaluator.evaluate("NER", self.datadir, self.outfile_name)

if __name__ == '__main__':
    baseline = BaselineNer()
    baseline.processFile()
