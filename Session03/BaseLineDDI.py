#! /usr/bin/python3

import sys
from os import listdir

import nltk
nltk.download('stopwords')

sys.path.append("../")

import evaluator

from xml.dom.minidom import parse

import argparse

class BaselineDDI():
    def __init__(self):
        args = self.parse_arguments()

        self.datadir = args["datadir"]
        self.outfile_name = args["outfile"]
        self.external = args["external"]

        self.f = open(self.outfile_name, "w+")

        # Launch a CoreNLP server
        # cd stanford-corenlp-full-2018-10-05
        # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer

        # import nltk CoreNLP module (just once)
        from nltk.parse.corenlp import CoreNLPDependencyParser

        # connect to your CoreNLP server (just once)
        self.my_parser = CoreNLPDependencyParser(url="http://localhost:9000")


        print("[INFO] Starting...", flush=True)

    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', type=str, default="../data/train/", help='Directory with XML files to process')
        parser.add_argument('-outfile', '--outfile', type=str, default="result.out", help='Name for the output file')
        parser.add_argument('--external', action="store_false", default=True, help='Whether to use external resources or not')

        args = vars(parser.parse_args())
        return args


    def analyze(self, s):
        '''
        Task :
        Given one sentence , sends it to CoreNLP to obtain the tokens , tags , and
        dependency tree . It also adds the start /end offsets to each token .
        Input :
        s: string containing the text for one sentence
        Output :
        Returns the nltk DependencyGraph ( https :// www . nltk . org / _modules / nltk /
        parse / dependencygraph . html ) object produced by CoreNLP , enriched with
        token offsets.
        Example : analyze (" Caution should be exercised when combining resorcinol or
        salicylic acid with DIFFERIN Gel ")
        {0:{ ' head ':None ,' lemma ':None ,' rel ':None ,' tag ':' TOP ','word ': None },
        1:{ ' word ':' Caution ','head ':4,' lemma ':' caution ','rel ':' nsubjpass ','tag ':'NN
        ',' start ':0,' end ':6} ,
        2:{ ' word ':' should ','head ':4,' lemma ':' should ','rel ':' aux ','tag ':'MD ',' start
        ':8,' end ':13} ,
        3:{ ' word ':'be ','head ':4,' lemma ':'be ','rel ':' auxpass ','tag ':'VB ',' start
        ':15,' end ':16} ,
        4:{ ' word ':' exercised ','head ':0,' lemma ':' exercise ','rel ':' ROOT ','tag ':' VBN
        ',' start ':18,' end ':26} ,
        5:{ ' word ':' when ','head ':6,' lemma ':' when ','rel ':' advmod ','tag ':'WRB ',' start
        ':28,' end ':31} ,
        6:{ ' word ':' combining ','head ':4,' lemma ':' combine ','rel ':' advcl ','tag ':' VBG
        ',' start ':33,' end ':41} ,
        7:{ ' word ':' resorcinol ','head ':6,' lemma ':' resorcinol ','rel ':' dobj ','tag ':'NN
        ',' start ':43,' end ':52} ,
        '''
        # parse text (as many times as needed). Watch the comma!
        mytree, = self.my_parser.raw_parse(s)
        i = 0
        offset = 0
        for w in mytree.nodes:
            word = str(mytree.get_by_address(i)["word"])
            offset = s.find(word, offset - 1)
            if offset == -1:
                offset += 1
            mytree.nodes[i]['start'] = offset
            mytree.nodes[i]['end'] = offset + len(word) - 1
            offset += len(word) + 1
            i += 1
        return mytree


    def check_interaction(self, analysis, entities, e1, e2):
        '''
        Task :
        Decide whether a sentence is expressing a DDI between two drugs .
        Input :
        analysis : a DependencyGraph object with all sentence information
        entites : A list of all entities in the sentence (id and offsets )
        e1 ,e2: ids of the two entities to be checked .
        Output : Returns the type of interaction ( ’ effect ’, ’mechanism ’, ’advice
        ’, ’int ’) between e1 and e2 expressed by the sentence , or ’None ’ if no
        interaction is described .
        '''
        word = "..."
        if (int(e2[-1:]) - int(e1[-1:])) > 1:
            for w in analysis.nodes:
                if not ";" in entities.get(e1)[1] and not ";" in entities.get(e2)[0]:
                    if int(analysis.get_by_address(w)["start"]) > int(entities.get(e1)[1]) and int(
                            analysis.get_by_address(w)["start"]) < int(entities.get(e2)[0]):
                        word = analysis.get_by_address(w)["word"]
                    if word == "administer" or word == "potentiate" or word == "prevent":
                        return "effect"
                    elif word == "reduce" or word == "increase" or word == "decrease":
                        return "mechanism"
                    elif word == "interact" or word == "interaction":
                        return "int"
        return "null"

    def process_directory(self):
        # process each file in directory
        length = len(listdir(self.datadir))
        for i, f in enumerate(listdir(self.datadir)):
            # parse XML file , obtaining a DOM tree
            tree = parse(self.datadir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text
                # load sentence entities into a dictionary
                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    eid = e.attributes["id"].value
                    entities[eid] = e.attributes["charOffset"].value.split("-")

                # Tokenize , tag , and parse sentence
                if "%" not in stext and len(stext) != 0:
                    analysis = self.analyze(stext)

                # for each pair in the sentence , decide whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    ddi_type = self.check_interaction(analysis, entities, id_e1, id_e2)
                    if ddi_type != None and ddi_type != "null":
                        print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file = self.f)
            if(i % (length/10) == 0):
                print("[INFO] "+ (i/length)*100+" complete.")

        self.f.close()
        # get performance score
        evaluator.evaluate("DDI", self.datadir, self.outfile_name)

if __name__ == '__main__':
    baseline = BaselineDDI()
    baseline.process_directory()
