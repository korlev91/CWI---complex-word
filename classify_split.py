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
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
#from sknn.mlp import Classifier, Layer


def MyClassifier():
    traindata = "F:/CWI/cwi_training/cwi_training2.txt.lbl.conll"
    testdata = "F:/CWI/cwi_testing/cwi_testing.txt.conll"
    
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=traindata)
    args = parser.parse_args()

    parser.add_argument('--test', help="parsed-and-label input format", default=testdata)
    args2 = parser.parse_args()
    
    labels = []
    featuredicts = []
    
    labels2 = []
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
    bgm    = nltk.collocations.BigramAssocMeasures()
    finder = nltk.collocations.BigramCollocationFinder.from_words(nltk.corpus.brown.words())
    scored = finder.score_ngrams(bgm.likelihood_ratio)
    #scored = 0
#################################################################################################   
         
    sylcount = 0
    print("Collecting features...")
    count=0
    onecnt = 0
    zerocnt = 0
    onecnt2 = 0
    zerocnt2 = 0
    for s in readSentences(args.train):
       #print("\r"+str(count), "")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,scored,og)
                if sylcount <1700:
                    featuredicts.append(w.baselinefeatures())
                    labels.append(w.label)
                    if w.label == 1:
                        onecnt += 1
                    else:
                        zerocnt += 1
                    #if sylcount == 2:
                        #print featuredicts
                else:
                    featuredicts2.append(w.baselinefeatures())
                    labels2.append(w.label)
                    if w.label == 1:
                        onecnt2 += 1
                    else:
                        zerocnt2 += 1
                    
                sylcount += 1

    #sys.exit(0)
    print onecnt, zerocnt    
    print onecnt2, zerocnt2         
    print()
    vec = DictVectorizer()

    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    features2 = vec.transform(featuredicts2).toarray()
    labels2 = np.array(labels2)
    
    ####################################  PROCESSING DATA #######################################################
    #selection = SelectKBest(k=1)
    #features = selection.fit(Xfeatures, labels).transform(features)
    
    scaler = preprocessing.Normalizer()
    #scaler = preprocessing.RobustScaler(with_centering = False)
    features=scaler.fit_transform(features)
    features2=scaler.transform(features2)
    '''
    scaler = preprocessing.MaxAbsScaler()
    features=scaler.fit_transform(features)
    features2=scaler.transform(features2)
    #print scaler.mean_
    '''
    
    flen = len(features[0])
    print flen
    print features[0][0], features[0][flen - 9], features[0][flen - 8], features[0][flen - 7], features[0][flen - 6]
    print features[0][flen - 5], features[0][flen - 4], features[0][flen - 3], features[0][flen - 2], features[0][flen - 1]
    #for i in range(len(features[0])):
     #   print features[0][i]
    #print features2
    #vocab = vec.get_feature_names()
    #print vocab
    #print features[0]
    #maxent = LogisticRegression(penalty='l1')
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    #maxent = tree.DecisionTreeClassifier()
    #maxent = ensemble.RandomForestClassifier()
    #maxent = svm.NuSVC(nu=0.3, kernel='rbf', class_weight={1: 10});
    maxent = svm.SVC(C=5, kernel='rbf', class_weight={1: 2});
    #maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    ####################
    #neuralfunc(features,labels)
    #print 'neural func'
    ####################

    scores = defaultdict(list)
    TotalCoeffCounter = Counter()
    cnt = 0;
    for i in range(1):
        TrainX_i = features
        Trainy_i = labels

        TestX_i = features2
        Testy_i =  labels2

        maxent.fit(TrainX_i,Trainy_i)
        plt.scatter(TrainX_i[:, flen-6], TrainX_i[:, flen-5], c=Trainy_i)
        plt.show()

        ypred_i = maxent.predict(TestX_i)
        #coeffs_i = list(maxent.coef_[0])
        #coeffcounter_i = Counter(vec.feature_names_)
        #for value,name in zip(coeffs_i,vec.feature_names_):
        #    coeffcounter_i[name] = value

        a = accuracy_score(Testy_i, ypred_i)
        p = precision_score(Testy_i, ypred_i)
        print p
        r = recall_score(Testy_i, ypred_i)
        scores["Accuracy"].append(accuracy_score(Testy_i, ypred_i))
        scores["F1"].append(f1_score(Testy_i, ypred_i))
        scores["Precision"].append(precision_score(Testy_i, ypred_i))
        scores["Recall"].append(recall_score(Testy_i, ypred_i))
        
        ##########################################
        #print ypred_i
        '''
        tp = tn = fp = fn = 0
        l = len(Testy_i)
        for i in range(l):
            if ypred_i[i] == Testy_i[i]:
                if Testy_i[i] == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if Testy_i[i] == 0:
                    fp += 1
                else:
                    fn += 1
        print 'myprec ', float(tp)/(tp+fp)
        print 'myrecall ', float(tp)/(tp+fn)
        '''
        ########################################
        if a == 0 and r == 0:
            FF = 0
        else:
            FF = (2 * (a * r))/(a + r)
        scores["FF score"].append(FF)
        
        cm = confusion_matrix(Testy_i, ypred_i)
        print(cm)
        cnt += 1
        if cnt == 4:
            break
        #posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        #negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))


    #print("Pervasive positive: ", posfeats)
    #print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")


    maxent.fit(features,labels) # fit on everything

    #coeffs_total = list(maxent.coef_[0])
    #for value,name in zip(coeffs_total,vec.feature_names_):
    #        TotalCoeffCounter[name] = value

    #for (key,value) in TotalCoeffCounter.most_common()[:20]:
        #print(key,value)
    print("---")
    #for (key,value) in TotalCoeffCounter.most_common()[-20:]:
        #print(key,value)
    #print("lowest coeff:",coeffcounter.most_common()[-1])
    #print("highest coeff",coeffcounter.most_common()[0])



######################   RUN ##############################

#BasicClassifier()
MyClassifier()
print "Korlev 1"
   
    
    