#! /usr/bin/python3

import sys
import argparse
from os import listdir
import string
import pandas as pd
from xml.dom.minidom import parse



class Scratcher():
    def __init__(self):
        args = self.parse_arguments()

        self.datadir = args["datadir"]
        self.outfile_name = args["outfile"]

        self.f = open(self.outfile_name, "w+")

        print("Starting the process of directory " + self.datadir + " saved in " + self.outfile_name)

    def parse_arguments(self):
        # construct the argument parser
        parser = argparse.ArgumentParser()
        parser.add_argument('-datadir', '--datadir', type=str, default="data/train/", help='Directory with XML files to process')
        parser.add_argument('-outfile', '--outfile', type=str, default="info.out", help='Name for the output file')

        args = vars(parser.parse_args())
        return args


    def createMap(self, token, type):
        return  {'name': token[0], 'offset': str(token[1])+"-" +str(token[2]), 'type': type}


    def scratchInfo(self):
        self.info = []
        # process each file in directory
        for f in listdir(self.datadir):
            # parse XML file , obtaining a DOM tree
            tree = parse(self.datadir + "/" + f)
            # process each sentence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                info = s.getElementsByTagName("entity")
                for i in info:
                    sid = i.attributes["id"].value  # get sentence id
                    stext = i.attributes["text"].value  # get sentence text
                    itype = i.attributes["type"].value
                    self.info.append( {'name': stext, 'type': itype})
                # print sentence entities in format requested for evaluation
                    print(sid + "|" + stext + "|" + itype, file = self.f)
        self.f.close()
        # create statistics
        self.searchSuffixes()

    def searchSuffixes(self):
        self.suffixes = {}
        for info in self.info:
            if info["type"] not in self.suffixes:
                self.suffixes[info["type"]] = {}
            if info["name"][-3:] not in self.suffixes[info["type"]]:
                self.suffixes[info["type"]][info["name"][-3:]] = 0
            self.suffixes[info["type"]][info["name"][-3:]] +=1
        a = pd.DataFrame.from_dict(self.suffixes)
        ordered_suffixes_group = a["group"].sort_values(ascending=False).dropna().to_dict()
        ordered_suffixes_drug = a["drug"].sort_values(ascending=False).dropna().to_dict()
        ordered_suffixes_drug_n = a["drug_n"].sort_values(ascending=False).dropna().to_dict()
        ordered_suffixes_brand= a["brand"].sort_values(ascending=False).dropna().to_dict()
        self.suffixes["group"] = ordered_suffixes_group
        self.suffixes["drug"] = ordered_suffixes_drug
        self.suffixes["drug_n"] = ordered_suffixes_drug_n
        self.suffixes["brand"] = ordered_suffixes_brand

        from matplotlib import pyplot as plt
        #ax = a.plot.bar()
        #plt.show()
        print(self.suffixes)
        self.checkInterctions(self.suffixes)

    def checkInterctions(self, map):
        interactions = {}
        for type, values in map.items():
            for tok, num in values.items():
                for type2 in map.keys():
                    if type is not type2:
                        if tok in map[type2]:
                            if tok not in interactions:
                                interactions[tok] = []
                            interactions[tok].append(str(type)+"-"+str(num))
        print("INTERACTIONS")
        print(interactions)


if __name__ == '__main__':
    scratcher = Scratcher()
    scratcher.scratchInfo()