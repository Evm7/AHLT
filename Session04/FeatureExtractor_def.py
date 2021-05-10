#! /usr/bin/python3

import sys
from os import listdir

import nltk
nltk.download('stopwords')

sys.path.append("../")

from xml.dom.minidom import parse

import argparse

class FeatureExtractor():
    def __init__(self):
        args = self.parse_arguments()

        self.datadir = args["datadir"]
        self.outfile_name = "outputs/"+ args["outfile"]
        self.external = args["external"]

        self.f = open(self.outfile_name+"_features.txt", "w+")
        self.f_info = open(self.outfile_name+"_info_features.txt", "w+")

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
        parser.add_argument('-outfile', '--outfile', type=str, default="train", help='Name for the output file')
        parser.add_argument('--external', action="store_false", default=True, help='Whether to use external resources or not')
        args = vars(parser.parse_args())
        return args

    def processAll(self):
        splits = ["train", "devel", "test"]
        print("We are proceeding with all the three files")
        for s in splits:
            print("Starting with : "+ s)
            self.datadir = "../data/"+s+"/"
            self.outfile_name= "outputs/"+s
            self.f = open(self.outfile_name + "_features.txt", "w+")
            self.f_info = open(self.outfile_name + "_info_features.txt", "w+")
            self.process_directory()
        print("Bye bye")

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


    def extract_features(self, analysis, entities, e1, e2):
        '''
        Task :
        Given an analyzed sentence and two target entities , compute a feature
        vector for this classification example .
        Input :
        tree : a DependencyGraph object with all sentence information .
        entities : A list of all entities in the sentence (id and offsets ).
        e1 , e2 : ids of the two entities to be checked for an interaction
        Output :
        A vector of binary features .
        Features are binary and vectors are in sparse representation (i.e. only
        active features are listed )
        '''
        feat = {}
        nodes = analysis.nodes

        # Type of Entity as features
        feat["type1"]=entities[e1]["type"]
        feat["type2"]=entities[e2]["type"]

        # Get entities
        entity1, entity2 = self.getNodes(nodes, entities, e1, e2)

        id_others = self.differ([e1, e2], list(entities.keys()))

        other_entities = [self.getNodes_Individual(nodes, entities, id_) for id_ in id_others]

        feat["error"]= False
        # If entities can not be retrieved, no DDI --> null
        if len(entity1) < 1 or len(entity2)< 1:
            return ["error"]

        # Get parent Nodes of each entity
        parent1, rel1 = self.getParentNode(nodes, entity1)
        parent2, rel2 = self.getParentNode(nodes, entity2)


        # information about parent1
        feat["lemma1"] = parent1["lemma"]
        feat["rel1"] = rel1
        feat["word1"] = parent1["word"]
        feat["tag1"] = parent1["tag"]

        # information about parent2
        feat["lemma2"] = parent2["lemma"]
        feat["rel2"] = rel2
        feat["word2"] = parent2["word"]
        feat["tag2"] = parent2["tag"]

        # If entity1 is under entity2, or visceversa
        feat["is_under1"]= self.is_under(entity1, entity2)
        feat["is_under2"]= self.is_under(entity2, entity1)


        feat["sameNode"] = False
        feat["under_verb"] = False

        # Entities under same parent
        if self.sameNode(parent1, parent2):
            tag = parent1['tag'].lower()[0]
            feat["sameNode"] = True

            if tag == 'v':
                feat["under_verb"]=True

        start1 = min([e['start'] for e in entity1])
        end1 = max([e['end'] for e in entity1])
        start2 = min([e['start'] for e in entity2])
        end2 = max([e['end'] for e in entity2])



        words_in_between = 0
        for v in list(analysis.nodes.values()):
            if 'start' in v.keys():
                if v['start'] < start1:
                    feat["lemma_before"] = v["lemma"]
                if end1 < v['start'] < start2:
                    if v['tag'].lower()[0] == "v":
                        feat["verb_inbetween"] = v["lemma"]
                    if v['tag'].lower() == "in":
                        feat["prep_inbetween"] = v["lemma"]

                    words_in_between +=1
                    feat["lemma_inbetween"] = v["lemma"]
                if end2 < v['start']:
                    feat["lemma_after"] = v["lemma"]
        # count the word between both the pair
        feat["words_in_between"] = words_in_between

        ## check number of other entities in between the given pair
        for ent in other_entities:
            if len(ent)>0:
                start_other = min([e['start'] for e in ent])
                if end1 < start_other < start2:
                    feat["other_entities"] = True


        # FEATURES EXTRACTED FROM EDAs
        # information about parent1 being classes
        feat['int1'] = parent1['lemma'] in ['interact', 'interaction', 'other', 'study', 'supplement']
        feat['advise1'] = parent1['lemma'] in ['take', 'administer', 'bind', 'adjustment', 'avoid', 'recommend', 'contraindicate']
        feat['effect1'] = parent1['lemma'] in ['effect', 'enhance', 'receive', 'response', 'action', 'diminish']
        feat['mechanism1'] = parent1['lemma'] in ['increase', 'binding', 'clearance', 'concentration', 'concentrations', 'absorption', 'metabolism', 'presence', 'administration', 'inhibit', 'level']

        # information about parent2 being classes
        feat['int2'] = parent2['lemma'] in ['aminoglycoside', 'aminoglycoside', 'interact', 'interaction']
        feat['advise2'] = parent2['lemma'] in ['dose', 'medication', 'take', 'caution']
        feat['effect2'] = parent2['lemma'] in [None, 'action', 'activity', 'agent', 'effect', 'steroid', 'response', 'acetaminophen']
        feat['mechanism2'] = parent2['lemma'] in ['absorption', 'clearance', 'intake', 'level', 'concentration', 'concentrations','metabolism']

        # information about before nouns
        if "lemma_before" in feat:
            feat['advise_bef'] = feat["lemma_before"] in ['administer', 'take', 'if']
            feat['mechanism_bef'] = feat["lemma_before"] in ['plasma', 'combination']


        effect_between = ["administer", "potentiate", "prevent", "may", "effects", "response", "certain", "include"]
        null_between = ["acid", "drugs"]

        feat['effect_between'] = False
        feat['null_between'] = False

        if self.words_inbetweens(analysis, e1,e2, entities, effect_between):
            feat['effect_between'] = True

        if self.words_inbetweens(analysis, e1,e2, entities, null_between):
            feat['null_between'] = True
            
        features = []

        for k, v in feat.items():
            if type(v) is bool:
                if v:
                    features.append(k)
            else:
                features.append(str(k)+"="+str(v))

        return features

    def differ(self, list_1, list_2):
        a = []
        for l in list_2:
            if l not in list_1:
                a.append(l)
        return a

    def words_inbetweens(self, analysis, e1,e2, entities, words):
        '''
        Check if certain words are placed inbetween our entities. Return True if they are.
        '''
        word = "..."
        nodes = analysis.nodes
        for w in nodes:
            if not ";" in entities.get(e1)["offset"][1] and not ";" in entities.get(e2)["offset"][0] and None != analysis.get_by_address(w)['word']:
                if int(analysis.get_by_address(w)['start']) > int(entities.get(e1)["offset"][1]) and int(
                        analysis.get_by_address(w)['start']) < int(entities.get(e2)["offset"][0]):
                    word = analysis.get_by_address(w)['word']
                if word in words:
                    return True
        return False

    def check_inbetweens(self, start_e, end_e, start_k, end_k):
        '''
        Check whether an entity, defined by [start_e, end_e] is inside a given node [start_k, end_k]
        in different ways: being entity in between node, or visceversa.
        This way we are solving the issue with compounds nouns (group of nodes may define one same entity)
        '''
        ranger_tree = range(int(start_k), int(end_k) + 1)
        ranger_entity = range(int(start_e), int(end_e) + 1)
        return (int(start_e) in ranger_tree or int(end_e) in ranger_tree) or (start_k in ranger_entity and end_k in ranger_entity)

    def getNodes_Individual(self, tree, entities, e1):
        """
        Allows to return all the nodes which are referenced with the identifier e1.
        """
        ents1 = []
        start1 = entities[e1]["offset"][0].split(";")
        end1 = entities[e1]["offset"][1].split(";")

        for k in list(tree.keys()):
            if 'start' in tree[k].keys():
                for i in range(len(start1)):
                    if self.check_inbetweens(start1[i], end1[i], tree[k]['start'], tree[k]['end']):
                        ents1.append(tree[k])
        return ents1

    def getNodes(self, tree, entities, e1, e2):
        """
        Allows to return all the nodes which are referenced with the identifiers e1 and e2 given.
        This method may allow to obtain a set of nodes which are owned by a given pair of entities.
        """
        ents1 = []
        ents2 = []

        start1 = entities[e1]["offset"][0].split(";")
        start2 = entities[e2]["offset"][0].split(";")
        end1 = entities[e1]["offset"][1].split(";")
        end2 = entities[e2]["offset"][1].split(";")

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
        self.features = []
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
                    entities[eid] = {}
                    entities[eid]["type"] = e.attributes["type"].value
                    entities[eid]["offset"] = e.attributes["charOffset"].value.split("-")

                # Tokenize , tag , and parse sentence
                if "%" not in stext:
                    analysis = self.analyze(stext)
                else:
                    stext = stext.replace("%", "p")
                    analysis = self.analyze(stext)


                # for each pair in the sentence , decide whether it is DDI and its type
                pairs = s.getElementsByTagName("pair")
                for p in pairs:
                    #get ground truth
                    ddi = p.attributes["ddi"].value
                    dditype = p.attributes["type"].value if ddi=="true" else "null"

                    # target entities
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    feats = self.extract_features(analysis, entities, id_e1, id_e2)
                    print(dditype, "\t".join(feats), sep="\t", file = self.f)
                    print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t", file = self.f_info)

            if not (i % int(length/10)):
                print("[INFO] "+ str(int(i/length*100))+"% complete.", flush=True)



        self.f.close()
        self.f_info.close()




if __name__ == '__main__':
    feat_extractor = FeatureExtractor()
    feat_extractor.processAll()
