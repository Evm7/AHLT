#! /usr/bin/python3

import sys
import argparse
from os import listdir
import string

sys.path.append("../")

import evaluator

from xml.dom.minidom import parse

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

class FeatureExtractor():
    def __init__(self):
        args = self.parse_arguments()

        self.datadir = "../"+ args["datadir"]
        self.outfile_name = args["outfile"]
        self.external = args["external"]

        self.f = open(self.outfile_name, "w+")

        self.stopwords = set(stopwords.words('english'))

        print("Starting the process of directory " + self.datadir + " saved in " + self.outfile_name)
        self.extract_dict()


    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', type=str, default="data/devel/", help='Directory with XML files to process')
        parser.add_argument('-outfile', '--outfile', type=str, default="devel_ext.feat", help='Name for the output file')
        parser.add_argument('--external', action="store_false", default=True, help='Whether to use external resources or not')

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

        return tokens


    def extract_features(self, s):
        #TODO Add more conditions to extract better features
        '''
        Task :
        Given a tokenized sentence , return a feature vector for each token
        Input :
        s: A tokenized sentence ( list of triples (word , offsetFrom , offsetTo ) )
        Output :
        A list of feature vectors , one per token .
        Features are binary and vectors are in sparse representation (i.e. only active features are listed )
        Example :
        extract_features ([(" Ascorbic " ,0 ,7) , (" acid " ,9 ,12) , (" ," ,13 ,13) ,
        (" aspirin " ,15 ,21) , (" ," ,22 ,22) , (" and " ,24 ,26) , (" the " ,28 ,30) ,
        (" common " ,32 ,37) , (" cold " ,39 ,42) , ("." ,43 ,43) ])
        [ [ " form = Ascorbic " , " suf4 = rbic ", " next = acid ", " prev = _BoS_ " , "capitalized" ],
        [ " form = acid ", " suf4 = acid " , " next = ," , " prev = Ascorbic " ] ,
        [ " form = ," , " suf4 = ," , " next = aspirin ", " prev = acid ", " punct " ],
        [ " form = aspirin " , " suf4 = irin ", " next = ," , " prev = ," ],
        ...
        ]
        '''
        prev = "_BoS_"
        feature_vectors = []
        for i in range(len(s)):
            if i == len(s)-1:
                next = "_EoS_"
            else:
                next = s[i+1][0]
            feature_vectors.append(["form = " + s[i][0], "suf3 = " + s[i][0][-3:], "pref3 = " + s[i][0][:3], "next = " + next, "prev = " + prev])
            prev = s[i][0]
            if s[i][0].isupper():
                feature_vectors[i].append("All_Caps")
            if s[i][0][0].isupper():
                feature_vectors[i].append("capitalized")
            if (s[i][0] in set(stopwords.words('english'))) or (s[i][0] in string.punctuation):
                feature_vectors[i].append("punct")
            if len(s[i][0]) > 10:
                feature_vectors[i].append("Over10Chars")
            if len(s[i][0]) < 5:
                feature_vectors[i].append("Under5Chars")
            for char in s[i][0]:
                if char.isdigit():
                    feature_vectors[i].append("ContainsNumber")
                    break
            for char in s[i][0]:
                if char=="-":
                    feature_vectors[i].append("ContainsDash")
                    break
            for char in s[i][0]:
                if char=="/":
                    feature_vectors[i].append("ContainsSlash")
                    break
            if self.external:
                   if s[i][0].lower() in self.HSDB:
                        feature_vectors[i].append('hsdb_drug')
                   if s[i][0].lower() in self.drugbank_dict:
                            feature_vectors[i].append('drug_bank_' + self.drugbank_dict[s[i][0].lower()])
                   elif s[i][0].upper() in self.drugbank_dict:
                            feature_vectors[i].append('drug_bank_' + self.drugbank_dict[s[i][0].upper()])
                   elif s[i][0] in self.drugbank_dict:
                            feature_vectors[i].append('drug_bank_' + self.drugbank_dict[s[i][0]])
            

        return feature_vectors

    def seak_External(self, tok, map):
        if tok in map:
            return map[tok]
        return None

    def get_tag(self, token ,gold):
        '''
        Task :
        Given a token and a list of ground truth entites in a sentence , decide
        which is the B-I-O tag for the token
        Input :
        token : A token , i.e. one triple (word , offsetFrom , offsetTo )
        gold : A list of ground truth entities , i.e. a list of triples (
        offsetFrom , offsetTo , type )
        Output :
        The B-I-O ground truth tag for the given token ("B- drug ", "I- drug ", "Bgroup ", "I- group ", "O", ...)
        Example :
        get_tag ((" Ascorbic " ,0 ,7) , [(0 , 12 , " drug ") , (15 , 21 , " brand ") ])
        B- drug
        get_tag ((" acid " ,9 ,12) , [(0 , 12 , " drug ") , (15 , 21 , " brand ") ])
        I- drug
        get_tag ((" common " ,32 ,37) , [(0 , 12 , " drug ") , (15 , 21 , " brand ") ])
        O
        get_tag ((" aspirin " ,15 ,21) , [(0 , 12 , " drug ") , (15 , 21 , " brand ") ])
        B- brand
        '''

        range1 = range(token[1], token[2])
        for gold_word in gold:
            range2 = range(gold_word[0], gold_word[1])
            if range1.start in range2 and range1.stop-1 in range2:
                if token[1] == gold_word[0]:
                    return "B-" + gold_word[2]
                else:
                    return "I-" + gold_word[2]
        return "O"

    def test(self):
        print(self.extract_features([("Ascorbic" ,0 ,7) , ("acid" ,9 ,12) , ("," ,13 ,13) ,
        ("aspirin" ,15 ,21) , ("," ,22 ,22) , ("and" ,24 ,26) , ("the" ,28 ,30) ,
        ("common" ,32 ,37) , ("cold" ,39 ,42) , ("." ,43 ,43) ]))
        print("\n")
        print(self.get_tag((" Ascorbic " ,0 ,7) , [(0 , 12 , "drug"), (15 , 21 , "brand")]))
        print(self.get_tag((" acid " ,9 ,12) , [(0 , 12 , "drug"), (15 , 21 , "brand")]))
        print(self.get_tag((" common " ,32 ,37) , [(0 , 12 , "drug"), (15 , 21 , "brand")]))
        print(self.get_tag((" aspirin " ,15 ,21) , [(0 , 12 , "drug"), (15 , 21 , "brand")]))


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
                # extract features for each word in the sentence
                features = self.extract_features(tokens)

                # print features in format suitable for the learner / classifier
                for i in range(0, len(tokens)):
                    # see if the token is part of an entity , and which part(B/I)
                    tag = self.get_tag(tokens[i], gold)
                    print(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t', file = self.f)
                # blank line to separate sentences
                print("", file = self.f)

        self.f.close()


if __name__ == '__main__':
    featureExtractor = FeatureExtractor()
    featureExtractor.processFile()
