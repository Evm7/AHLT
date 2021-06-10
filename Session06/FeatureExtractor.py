from xml.dom.minidom import parse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from os import listdir


import json, pathlib


class FeaturesExtractor():
    def __init__(self):
        print("[WELCOME NEURAL NETWORKS DDI]... Init Feature Extraction progress")
        # import nltk CoreNLP module (just once)
        from nltk.parse.corenlp import CoreNLPDependencyParser
        # Launch a CoreNLP server
        # cd stanford-corenlp-full-2018-10-05
        # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer
        # connect to your CoreNLP server (just once)
        self.my_parser = CoreNLPDependencyParser(url="http://localhost:9000")

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
        feat = {}
        nodes = analysis.nodes

        sentence = {k:v['word'] for k,v in nodes.items()}
        sentence.pop(0) # remove general root, as do not belong to the sentence
        # Type of Entity as features
        feat["type1"] = entities[e1]["type"]
        feat["type2"] = entities[e2]["type"]

        # Get entities
        entity1, entity2 = self.getNodes(nodes, entities, e1, e2)

        # If entities can not be retrieved, no DDI --> null
        if len(entity1) < 1 or len(entity2) < 1:
            return "error"

        id_others = self.differ([e1, e2], list(entities.keys()))
        other_entities = [self.getNodes_Individual(nodes, entities, id_) for id_ in id_others]
        if len(other_entities)>0:
            sentence = self.mask(sentence, other_entities[0], "<DRUG_OTHERS>")

        # mask sentence
        sentence = self.mask(sentence, entity1, "<DRUG_1>")
        sentence = self.mask(sentence, entity2, "<DRUG_2>")


        pos1, pos2 = self.relPositions(sentence)

        feats = []
        num_feats = 3 # (word, lemma, tag) + (pos_to_1, pos_to_2)
        for index, (k, v) in enumerate(sentence.items()):
            if v in ["<DRUG_1>", "<DRUG_2>", "<DRUG_OTHERS>"]:
                feat_ = (v, )*num_feats
                feat_ + (pos1-index, pos2-index)
                feats.append(feat_)

            else:
                nd = nodes[k]
                if nd["word"] is not None:
                    feat_ = (nd['word'], nd['lemma'], nd['tag'], pos1-index, pos2-index)
                    feats.append(feat_)
        return feats

    def relPositions(self, sentence):
        pos1 = list(sentence.values()).index("<DRUG_1>")
        pos2 = list(sentence.values()).index("<DRUG_2>")
        return pos1, pos2

    def mask(self, sentence, entity, tag):
        for ent in entity:
            sentence[ent['address']] = tag
        return sentence

    def differ(self, list_1, list_2):
        a = []
        for l in list_2:
            if l not in list_1:
                a.append(l)
        return a

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

    def check_inbetweens(self, start_e, end_e, start_k, end_k):
        '''
        Check whether an entity, defined by [start_e, end_e] is inside a given node [start_k, end_k]
        in different ways: being entity in between node, or visceversa.
        This way we are solving the issue with compounds nouns (group of nodes may define one same entity)
        '''
        ranger_tree = range(int(start_k), int(end_k) + 1)
        ranger_entity = range(int(start_e), int(end_e) + 1)
        return (int(start_e) in ranger_tree or int(end_e) in ranger_tree) or (
                    start_k in ranger_entity and end_k in ranger_entity)

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

    def load_data(self, datadir):
        data = []
        # process each file in directory
        length = len(listdir(datadir))
        for i, f in enumerate(listdir(datadir)):
            # parse XML file , obtaining a DOM tree
            tree = parse(datadir + "/" + f)
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
                    # get ground truth
                    ddi = p.attributes["ddi"].value
                    dditype = p.attributes["type"].value if ddi == "true" else "null"

                    # target entities
                    id_e1 = p.attributes["e1"].value
                    id_e2 = p.attributes["e2"].value

                    feats = self.extract_features(analysis, entities, id_e1, id_e2)
                    if feats is not "error":
                        data.append(self.createCase(sid, id_e1, id_e2, dditype, feats))
            if not (i % int(length / 10)):
                print("[INFO] " + str(int(i / length * 100)) + "% complete.", flush=True)
        with open('features_'+pathlib.Path(datadir).stem+'.txt', 'w') as outfile:
            json.dump(data, outfile)
        return data

    def createCase(self, sid, id_e1, id_e2, dditype, feats):
        map_ = {}
        map_["sid"] = sid
        map_["id_e1"] = id_e1
        map_["id_e2"] = id_e2
        map_["dditype"] = dditype
        map_["feats"] = feats
        return map_

    def extractAll(self, traindir, validationdir, test_dir):
        '''
        Learns a NN model using traindir as training data , and validationdir
        as validation data . Saves learnt model in a file named modelname
        '''
        print("[INFO]... Extracting information from the training dataset")
        traindata = self.load_data(traindir)

        print("[INFO]... Extracting information from the training dataset")
        valdata = self.load_data(validationdir)

        print("[INFO]... Extracting information from the training dataset")
        testdata = self.load_data(test_dir)

        return traindata, valdata, testdata


if __name__ == '__main__':
    extractor = FeaturesExtractor()
    extractor.extractAll("../data/train", "../data/devel", "../data/test")

