import sys
sys.path.append("../")

import evaluator

import functools
import operator
split = "test"

datadir = "../data/"+split      # file with the *.xml files for the &split variable set
pred_name = "pred/"+split+".txt"
info_name = "outputs/"+split+"_info_features.txt"
outfile_name = "pred/"+split+"format.txt"

def loadData(path):
  with open(path) as file:
    data = file.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    data = [x.strip().split("\t")for x in data]
  return data

data = loadData(pred_name)
info = loadData(info_name)

y_data = [item[0] for item in data]
info_data = [item[:3] for item in info]

f = open(outfile_name, "w+")

for info, dat in zip(info_data, y_data):
    info.append(dat)
    if dat !="null":
        print("|".join(info), file=f)

f.close()

# get performance score
evaluator.evaluate("DDI", datadir, outfile_name)

