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

        self.advise_pos = ["MD"]
        self.interest_pos = ["NN", "VBZ", "VBN", "VBD", "VB", "MD", "RB", "VBP"]
        self.advise_clues = ["should", "may", "could", "would"]
        self.effect_clues = ["administered", "concurrently", "concomitantly", "increase", "increases", "increased", "effect",
                        "effects", "prevent", "prevents", "prevented", "potentiate", "potentiates", "potentiated", "administer"]
        self.mechanism_clues = ["inhibit", "reduce", "reduces", "reduced", "decrease", "decreases", "decreased", "change",
                           "changes", "changed", "elevate", "elevates", "elevated", "interfere", "interferes",
                           "interfered"]
        self.int_clues = ["interaction", "intereact"]

        print("[INFO] Starting...", flush=True)

    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', type=str, default="../data/devel/", help='Directory with XML files to process')
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
        for _ in mytree.nodes:
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
        nodes = analysis.nodes
        # Get entities
        entity1, entity2 = self.getNodes(nodes, entities, e1, e2)

        # If entities can not be retrieved, no DDI
        if len(entity1) < 1 or len(entity2)< 1:
            return (0, "null")

        # Get parent Nodes of each entity
        parent1, rel1 = self.getParentNode(nodes, entity1)
        parent2, rel2 = self.getParentNode(nodes, entity2)


        # If entity1 is under entity2, then no DDI
        if self.is_under(entity1, entity2):
            return (0, "null")

        # Entities under same parent
        if self.sameNode(parent1, parent2):
            tag = parent1['tag'].lower()[0]
            if tag != 'v':
                return (0, "null")
            return (1, "advise")

        self.responses1 = {}
        self.responses1["effect"] = ['response', 'diminish', 'enhance', 'effect']
        self.responses1["mechanism"] = ['concentration', 'concentrations', 'absorption', 'metabolism', 'presence', 'administration']
        self.responses1["int"] =['interact', 'interaction']
        self.responses1["advise"] = ['take','administer', 'bind', 'adjustment', 'avoid', 'recommend', 'contraindicate']

        self.responses2 = {}
        self.responses2["effect"] = ['effect', 'steroid', 'response', 'acetaminophen']
        self.responses2["mechanism"] = ['concentration','concentrations', 'absorption', 'metabolism', 'level', 'clearance']
        self.responses2["advise"] = ['take', 'caution']

        for classes_, lemmas in self.responses1.items():
            if parent1["lemma"] in lemmas:
                return (1, classes_)

        for classes_, lemmas in self.responses2.items():
            if parent2["lemma"] in lemmas:
                return (1, classes_)
        return (0, "null")


    def check_inbetweens(self, start_e, end_e, start_k, end_k):
        '''
        Check whether an entity, defined by [start_e, end_e] is inside a given node [start_k, end_k]
        in different ways: being entity in between node, or visceversa.
        This way we are solving the issue with compounds nouns (group of nodes may define one same entity)
        '''
        ranger_tree = range(int(start_k), int(end_k) + 1)
        ranger_entity = range(int(start_e), int(end_e) + 1)
        return (int(start_e) in ranger_tree or int(end_e) in ranger_tree) or (start_k in ranger_entity and end_k in ranger_entity)

    def getNodes(self, tree, entities, e1, e2):
        """
        Allows to return all the nodes which are referenced with the identifiers e1 and e1 given.
        This method may allow to obtain a set of nodes which are owned by a given pair of entities.
        """
        ents1 = []
        ents2 = []

        start1 = entities[e1][0].split(";")
        start2 = entities[e2][0].split(";")
        end1 = entities[e1][1].split(";")
        end2 = entities[e2][1].split(";")

        for k in list(tree.keys()):
            if 'start' in tree[k].keys():
                for i in range(len(start1)):
                    if self.check_inbetweens(start1[i], end1[i], tree[k]['start'], tree[k]['end']):
                        ents1.append(tree[k])
                for i in range(len(start2)):
                    if self.check_inbetweens(start2[i], end2[i], tree[k]['start'], tree[k]['end']):
                        ents2.append(tree[k])
        return ents1, ents2

    def getParentNode(self, tree, entity):
        """
        Function which obtains the parent of a given entity node and the type of relationship
        """
        if len(entity) == 1:
            return tree[entity[0]['head']], entity[0]['rel']
        parent = None
        rel = None
        for e in entity:
            if e['head'] not in [other['address'] for other in entity]:
                parent = tree[e['head']]
                rel = e['rel']
        return parent, rel

    def is_under(self, entity1, entity2):
        """
        Check whether entity1 is under entity2 in the graph
        """
        for i in range(len(entity1)):
            if entity1[i]['head'] in [e['address'] for e in entity2]:
                return True
        return False

    def sameNode(self, node1, node2):
        """
        Check whether both nodes are in the same level/address
        """
        return node1['address'] == node2['address']

    def process_directory(self):
        self.saver = {}
        self.saver["DDI_Drugs"] =[]
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
                if len(stext) < 1:
                    continue
                # load sentence entities into a dictionary
                entities = {}
                ents = s.getElementsByTagName("entity")
                for e in ents:
                    eid = e.attributes["id"].value
                    entities[eid] = e.attributes["charOffset"].value.split("-")

                # Tokenize , tag , and parse sentence

                if "%" not in stext:
                    analysis = self.analyze(stext)
                else:
                    stext = stext.replace("%", "p")
                    analysis = self.analyze(stext)


                # for each pair in the sentence , decide whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    find, ddi_type = self.check_interaction(analysis, entities, id_e1, id_e2)

                    if find:
                        print(sid + "|" + id_e1 + "|" + id_e2 + "|" + ddi_type, file = self.f)


            if not (i % int(length/10)):
                print("[INFO] "+ str(int(i/length*100))+"% complete.", flush=True)

        self.f.close()

        # get performance score
        evaluator.evaluate("DDI", self.datadir, self.outfile_name)

if __name__ == '__main__':
    baseline = BaselineDDI()
    #print(baseline.analyze("However, halothane anesthetic requirement (i.e., MAC) was depressed in a dose-dependent fashion as much as 56% 1-2 hours and as much as 14% 5-6 hours after injection of ketamine, 50 mg/kg, im. "))
    baseline.process_directory()
