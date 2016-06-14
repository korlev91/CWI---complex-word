import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
from utilities import *
import porter
import numpy as np

def MyClassifier():
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    defaultdata = scriptdir+"/../data/cwi_training/cwi_training.txt.lbl.conll"
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=defaultdata)
    args = parser.parse_args()

    labels = []
    featuredicts = []
    
    print("Collecting features...")
    count=0
    for s in readSentences(args.train):
       print("\r"+str(count), end="")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordInContext(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"])
                featuredicts.append(w.featurize())
                labels.append(w.label)
    print()
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)

    maxent = LogisticRegression(penalty='l1')
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    coeffcounter = Counter(vec.feature_names_)
    negfeats = set(vec.feature_names_)
    posfeats = set(vec.feature_names_)



    scores = defaultdict(list)
    TotalCoeffCounter = Counter()

    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels[TestIndices]

        maxent.fit(TrainX_i,Trainy_i)
        ypred_i = maxent.predict(TestX_i)
        coeffs_i = list(maxent.coef_[0])
        coeffcounter_i = Counter(vec.feature_names_)
        for value,name in zip(coeffs_i,vec.feature_names_):
            coeffcounter_i[name] = value

        scores["Accuracy"].append(accuracy_score(ypred_i,Testy_i))
        scores["F1"].append(f1_score(ypred_i,Testy_i))
        scores["Precision"].append(precision_score(ypred_i,Testy_i))
        scores["Recall"].append(recall_score(ypred_i,Testy_i))

        posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))


    print("Pervasive positive: ", posfeats)
    print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")


    maxent.fit(features,labels) # fit on everything

    coeffs_total = list(maxent.coef_[0])
    for value,name in zip(coeffs_total,vec.feature_names_):
            TotalCoeffCounter[name] = value

    for (key,value) in TotalCoeffCounter.most_common()[:20]:
        print(key,value)
    print("---")
    for (key,value) in TotalCoeffCounter.most_common()[-20:]:
        print(key,value)
    print("lowest coeff:",coeffcounter.most_common()[-1])
    print("highest coeff",coeffcounter.most_common()[0])

    sys.exit(0)