import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn import dummy, tree, ensemble, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from utilities import *
from WordFeatures import WordClass
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import cmudict
from sklearn import preprocessing
from nltk.corpus import brown
from nltk.probability import FreqDist
import numpy as np
import nltk.collocations
import collections
#from sknn.mlp import Classifier, Layer


def MyClassifier():
    traindata = "F:/CWI/cwi_training/cwi_training.txt.lbl.conll"
    testdata = "F:/CWI/cwi_testing/cwi_testing.txt.conll"
    
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=traindata)
    args = parser.parse_args()

    parser.add_argument('--test', help="parsed-and-label input format", default=testdata)
    args2 = parser.parse_args()
    
    labels = []
    featuredicts = []

    featuredicts2 = []
    
####################################################################################################
    import re
    f = 'F:/CWI/cwi_training/ogden.txt'

    with open(f) as temp_file:
        ogden = [line.rstrip(',') for line in temp_file]
        
    ogden[0] = re.sub("[^a-zA-Z]", " ", ogden[0])
    og = nltk.word_tokenize(ogden[0])
    
    cm_dict = cmudict.dict()
    bwords = FreqDist()
    for sentence in brown.sents():
        for w in sentence:
            bwords[w.lower()] += 1
    #bgm    = nltk.collocations.BigramAssocMeasures()
    #finder = nltk.collocations.BigramCollocationFinder.from_words(nltk.corpus.brown.words())
    #scored = finder.score_ngrams(bgm.likelihood_ratio)
    scored = 0
################################################################################################# 
    sylcount = 0
    print("Collecting features...")
    count=0
    for s in readSentences(args.train):
       print("\r"+str(count), "")
       count+=1
       #if count > 150:
        #   break
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,scored,og)
                featuredicts.append(w.baselinefeatures())
                labels.append(w.label)
                                 
                sylcount +=1 
                if sylcount == 2:
                        print featuredicts
                print sylcount
    
    count=0
    for s in readSentences2(args2.test):
       print("\r"+str(count), "")
       count+=1
       #if count > 100:
         #  break
       for i in s["idx"]:
            l=0
            w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,scored,og)
            featuredicts2.append(w.baselinefeatures())
        

                
    print()
    vec = DictVectorizer()

    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    features2 = vec.transform(featuredicts2).toarray()

    '''
    scaler = preprocessing.StandardScaler().fit(features)
    features=scaler.fit_transform(features)
    features2=scaler.transform(features2)
    '''

    #vocab = vec.get_feature_names()
    #print vocab
    #print features[0]
    #maxent = LogisticRegression(penalty='l1')
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    #maxent = tree.DecisionTreeClassifier()
    #maxent = ensemble.RandomForestClassifier()
    maxent1 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5}, gamma = 0.01);
    maxent5 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5});
    maxent10 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5}, gamma = 0.05);
    maxent50 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5}, gamma = 0.1);
    maxent10_1 = svm.SVC(C=10, kernel='rbf', class_weight={1: 1});
    #maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    
    
    for k in range(2):
        TrainX_i = features
        Trainy_i = labels

        TestX_i = features2
        #k = 0
        if k == 0:
            maxent1.fit(TrainX_i,Trainy_i)
            ypred_i = maxent1.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('F:/CWI/cwi_training2/result_C51_5_001.txt', 'w')
            np.savetxt(f,ypred_i)
            
        elif k == 1:
            maxent5.fit(TrainX_i,Trainy_i)
            ypred_i = maxent5.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('F:/CWI/cwi_training2/result_C51_5_auto.txt', 'w')
            np.savetxt(f,ypred_i)
            
        elif k == 2:
            maxent10.fit(TrainX_i,Trainy_i)
            ypred_i = maxent10.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('F:/CWI/cwi_training2/result_C51_5_005.txt', 'w')
            np.savetxt(f,ypred_i)
            
        elif i == 3:
            maxent50.fit(TrainX_i,Trainy_i)
            ypred_i = maxent50.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('F:/CWI/cwi_training2/result_C51_5_01.txt', 'w')
            np.savetxt(f,ypred_i)
            
        elif k == 4:
            maxent10_1.fit(TrainX_i,Trainy_i)
            ypred_i = maxent10_1.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('F:/CWI/cwi_training/result_C10_1main.txt', 'w')
            np.savetxt(f,ypred_i)
    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
        print 'DONE!!!!!!!!!!!!', k

    print("---")
    
    f.close()



######################   RUN ##############################

#BasicClassifier()
MyClassifier()
print "Korlev 1"
   