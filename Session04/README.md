RESULTS

[10/05] : Best results obtained with information added of between entities, and verbs between.

|     TRAINING    	|  tp  	|  fp 	|  fn  	| #pred 	| #exp 	|   P   	|   R   	|   F1  	|
|:---------------:	|:----:	|:---:	|:----:	|:-----:	|:----:	|:-----:	|:-----:	|:-----:	|
|        -        	|   -  	|  -  	|   -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      advise     	|  451 	| 106 	|  246 	|  557  	|  697 	| 81.0% 	| 64.7% 	| 71.9% 	|
|      effect     	| 1008 	| 193 	|  442 	|  1201 	| 1450 	| 83.9% 	| 69.5% 	| 76.0% 	|
|       int       	|  188 	|  16 	|  43  	|  204  	|  231 	| 92.2% 	| 81.4% 	| 86.4% 	|
|    mechanism    	|  643 	| 126 	|  377 	|  769  	| 1020 	| 83.6% 	| 63.0% 	| 71.9% 	|
|        -        	|   -  	|  -  	|   -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      M.avg      	|   -  	|  -  	|   -  	|   -   	|   -  	| 85.2% 	| 69.7% 	| 76.6% 	|
|        -        	|   -  	|  -  	|   -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      m.avg      	| 2290 	| 441 	| 1108 	|  2731 	| 3398 	| 83.9% 	| 67.4% 	| 74.7% 	|
| m.avg(no class) 	| 2450 	| 281 	|  948 	|  2731 	| 3398 	| 89.7% 	| 72.1% 	| 79.9% 	|


|      DEVEL      	|  tp 	|  fp 	|  fn 	| #pred 	| #exp 	|   P   	|   R   	|   F1  	|
|:---------------:	|:---:	|:---:	|:---:	|:-----:	|:----:	|:-----:	|:-----:	|:-----:	|
|                 	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      advise     	|  48 	|  52 	|  90 	|  100  	|  138 	| 48.0% 	| 34.8% 	| 40.3% 	|
|      effect     	| 104 	| 105 	| 211 	|  209  	|  315 	| 49.8% 	| 33.0% 	| 39.7% 	|
|       int       	|  16 	|  2  	|  19 	|   18  	|  35  	| 88.9% 	| 45.7% 	| 60.4% 	|
|    mechanism    	|  64 	|  83 	| 200 	|  147  	|  264 	| 43.5% 	| 24.2% 	| 31.1% 	|
|        -        	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      M.avg      	|  -  	|  -  	|  -  	|   -   	|   -  	| 57.5% 	| 34.4% 	| 42.9% 	|
|        -        	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      m.avg      	| 232 	| 242 	| 520 	|  474  	|  752 	| 48.9% 	| 30.9% 	| 37.8% 	|
| m.avg(no class) 	| 311 	| 163 	| 441 	|  474  	|  752 	| 65.6% 	| 41.4% 	| 50.7% 	|

|       TEST      	|  tp 	|  fp 	|  fn 	| #pred 	| #exp 	|   P   	|   R   	|   F1  	|
|:---------------:	|:---:	|:---:	|:---:	|:-----:	|:----:	|:-----:	|:-----:	|:-----:	|
|        -        	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      advise     	|  65 	|  46 	| 147 	|  111  	|  212 	| 58.6% 	| 30.7% 	| 40.2% 	|
|      effect     	| 115 	| 104 	| 168 	|  219  	|  283 	| 52.5% 	| 40.6% 	| 45.8% 	|
|       int       	|  2  	|  2  	|  16 	|   4   	|  18  	| 50.0% 	| 11.1% 	| 18.2% 	|
|    mechanism    	| 106 	|  93 	| 231 	|  199  	|  337 	| 53.3% 	| 31.5% 	| 39.6% 	|
|        -        	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      M.avg      	|  -  	|  -  	|  -  	|   -   	|   -  	| 53.6% 	| 28.5% 	| 35.9% 	|
|        -        	|  -  	|  -  	|  -  	|   -   	|   -  	|   -   	|   -   	|   -   	|
|      m.avg      	| 288 	| 245 	| 562 	|  533  	|  850 	| 54.0% 	| 33.9% 	| 41.6% 	|
| m.avg(no class) 	| 356 	| 177 	| 494 	|  533  	|  850 	| 66.8% 	| 41.9% 	| 51.5% 	|


SCHEMA
In this session we are providing DDI results by the analysis and training with Machine Learning Models
  - Feature Extractor Def is a python Script which allows to extract all the features from the annotated pairs in the .xml files from the DDI dataset.
  - Trainer.py is a file which, given a structured organization that will be mentioned afterwards, train, predicts and evaluate a big number of hyperparameters to allow the optimization of the features.
      - Structure:
        -Trainer.py
        -outputs/
          - Automatically we create a directory for each Model trained, with the model weights, the predictions over the features files and the evaluation of them.
        -features/
          - train_features.txt
          - devel_features.txt
          - test_features.txt
        - models/megam_i686.opt --> file which has the model.opt extracted from [MEGAM](http://users.umiacs.umd.edu/~hal/megam/version0_91/)
