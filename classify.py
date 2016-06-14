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
import nltk.collocations
import collections
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from generate_conll import targetWords
#from sknn.mlp import Classifier, Layer


def MainClassifier():
    
    traindata = "D:/SEMEVAL/FROM_HOME_NEW/CWI/cwi_training/cwi_training2.txt.lbl.conll"
    testdata = "D:/SEMEVAL/FROM_HOME_NEW/CWI/cwi_testing/cwi_testing.txt.conll"
    
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
    f = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/cwi_training/ogden.txt'

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
    prefix_keys = collections.defaultdict(list)
    for key, scores in scored:
       prefix_keys[key[0]].append((key[1], scores))
    for key in prefix_keys:
       prefix_keys[key].sort(key = lambda x: -x[1])
    suffix_keys = collections.defaultdict(list)
    for key, scores in scored:
       suffix_keys[key[1]].append((key[0], scores))
    for key in suffix_keys:
       suffix_keys[key].sort(key = lambda x: -x[1])
    #scored = 0
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
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,prefix_keys,
                            suffix_keys, og)
                featuredicts.append(w.baselinefeatures())
                labels.append(w.label)
                                 
                sylcount +=1 
                if sylcount == 2:
                        print featuredicts
                print sylcount
    
    count=0
    incount = 0
    for s in readSentences2(args2.test):
       print("\r"+str(count), "")
       count+=1
       #if count > 100:
         #  break
       print s
       print '---------- ',  targetWords[count]
       for i in s["idx"]:
           if s["form"][i] in targetWords[count]:
                l=0
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,prefix_keys,
                                suffix_keys, og)
                featuredicts2.append(w.baselinefeatures())
                incount += 1
    print 'incount: ', incount

                
    print()
    vec = DictVectorizer()

    features = vec.fit_transform(featuredicts).toarray()
    labels = np.array(labels)
    features2 = vec.transform(featuredicts2).toarray()
    
    ###############################################################################################
    of = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/featSave.txt'
    of2 = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/labelSave.txt'
    f = open(of, 'w')
    np.save(of, features)
    f.close()
    f = open(of2, 'w')
    np.save(of2, labels)
    f.close()
    
    of = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/featSaveTEST.txt'
    f = open(of, 'w')
    np.save(of, features2)
    f.close()
    #################################################################################################
    
    
def FitandPredSVM_MAIN(sclr):
    
    of = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/featSave.txt.npy'
    of2 = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    features = np.load(f) #------
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    labels = np.load(f) #------
    f.close()    
    
    of = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup_MAIN/featSaveTEST.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    features2 = np.load(f) #------
    f.close() 
    
    maxent1 = svm.SVC(C=50, kernel='rbf', class_weight={1: 2}, gamma = 0.001);
    maxent5 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5});
    maxent10 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5}, gamma = 0.05);
    maxent50 = svm.SVC(C=5, kernel='rbf', class_weight={1: 1.5}, gamma = 0.1);
    maxent10_1 = svm.SVC(C=10, kernel='rbf', class_weight={1: 1});
    #maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    
    
    #################################################################################################################
    from sklearn.feature_selection import SelectKBest, chi2
    '''selection = SelectKBest(chi2, k="all")
    features = selection.fit(features, labels).transform(features)
    features2 = selection.transform(features2)'''
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
        
        features = scaler[sclr].fit_transform(features)
        features2 = scaler[sclr].transform(features2)
    #####################################################################################################################
    
    
    for k in range(1):
        TrainX_i = features
        Trainy_i = labels

        TestX_i = features2
        k = 0
        if k == 0:
            maxent1.fit(TrainX_i,Trainy_i)
            ypred_i = maxent1.predict(TestX_i)
            for i in range(len(ypred_i)):
                ypred_i[i] = int(ypred_i[i])
            #print ypred_i
            
            f = open('D:/SEMEVAL/FROM_HOME_NEW/CWI/result_C50_12_001_robust.txt', 'w')
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
    
    
    
############################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #############################################################
def SplitClassifier():

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
    prefix_keys = collections.defaultdict(list)
    for key, scores in scored:
       prefix_keys[key[0]].append((key[1], scores))
    for key in prefix_keys:
       prefix_keys[key].sort(key = lambda x: -x[1])
    suffix_keys = collections.defaultdict(list)
    for key, scores in scored:
       suffix_keys[key[1]].append((key[0], scores))
    for key in suffix_keys:
       suffix_keys[key].sort(key = lambda x: -x[1])
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
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords,prefix_keys,
                            suffix_keys, og)
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
    
    #############################################################
    of = 'F:/CWI/FeatBackup_Split/featSave.txt'
    of2 = 'F:/CWI/FeatBackup_Split/labelSave.txt'
    f = open(of, 'w')
    np.save(of, features)
    f.close()
    f = open(of2, 'w')
    np.save(of2, labels)
    f.close()
    
    of = 'F:/CWI/FeatBackup_Split/featSave2.txt'
    of2 = 'F:/CWI/FeatBackup_Split/labelSave2.txt'
    f = open(of, 'w')
    np.save(of, features2)
    f.close()
    f = open(of2, 'w')
    np.save(of2, labels2)
    f.close()
    #############################################################
    
                
def FitandPred_SPLIT(mxnt, sclr):
    
    of = 'F:/CWI/FeatBackup_Split/featSave.txt'
    of2 = 'F:/CWI/FeatBackup_Split/labelSave.txt'
    f = open(of, 'rb')
    f.seek(0)
    features = np.load(f) #------
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    labels = np.load(f) #------
    f.close()    
    
    of = 'F:/CWI/FeatBackup_Split/featSave2.txt'
    of2 = 'F:/CWI/FeatBackup_Split/labelSave2.txt'
    f = open(of, 'rb')
    f.seek(0)
    features2 = np.load(f) #------
    f.close() 
    f = open(of2, 'rb')
    f.seek(0)
    labels2 = np.load(f) #------
    f.close()    
    
    
    maxent = []
    maxent.append(svm.SVC(C=5, kernel='rbf', class_weight={1: 2}));


    #################################################################################################################
    from sklearn.feature_selection import SelectKBest, chi2
    '''selection = SelectKBest(chi2, k="all")
    features = selection.fit(features, labels).transform(features)
    features2 = selection.transform(features2)'''
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
        
        features = scaler[sclr].fit_transform(features)
        features2 = scaler[sclr].transform(features2)
    #####################################################################################################################
    
    scores = defaultdict(list)

    cnt = 0;
    for i in range(1):
        TrainX_i = features
        Trainy_i = labels

        TestX_i = features2
        Testy_i =  labels2

        maxent[mxnt].fit(TrainX_i,Trainy_i)
        plt.scatter(TrainX_i[:, flen-6], TrainX_i[:, flen-5], c=Trainy_i)
        plt.show()

        ypred_i = maxent[mxnt].predict(TestX_i)

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

    print("--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")

    print("---")

    
############################### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #############################################################    
    
    
def MyClassifier():
    traindata = "F:/CWI/cwi_training/cwi_training2.txt.lbl.conll"
    
    parser = argparse.ArgumentParser(description="Skeleton for features and classifier for CWI-2016")
    parser.add_argument('--train', help="parsed-and-label input format", default=traindata)
    args = parser.parse_args()

    labels = []
    featuredicts = []
    
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
    prefix_keys = collections.defaultdict(list)
    for key, scores in scored:
       prefix_keys[key[0]].append((key[1], scores))
    for key in prefix_keys:
       prefix_keys[key].sort(key = lambda x: -x[1])
    suffix_keys = collections.defaultdict(list)
    for key, scores in scored:
       suffix_keys[key[1]].append((key[0], scores))
    for key in suffix_keys:
       suffix_keys[key].sort(key = lambda x: -x[1])
    #scored = 0
################################################################################################# 

    sylcount = 0
    print("Collecting features...")
    count=0
    for s in readSentences(args.train):
       #print("\r"+str(count), "")
       count+=1
       for l,i in zip(s["label"],s["idx"]):
            if l != "-":
                w = WordClass(s, i, s["form"][i],s["lemma"][i],s["pos"][i],s["ne"][i],l,s["head"],s["deprel"],cm_dict,bwords, prefix_keys,
                            suffix_keys, og)
                featuredicts.append(w.baselinefeatures())
                sylcount +=1 
                if sylcount == 2:
                    print featuredicts
                #print sylcount
                labels.append(w.label)
                
    print(), sylcount
    vec = DictVectorizer()
    features = vec.fit_transform(featuredicts).toarray()
    global VEC
    VEC = vec
    FEATURES.append(features)
    labels = np.array(labels)
    LABELS.append(labels)


    #############################################################
    of = 'F:/CWI/FeatBackup2/featSave.txt'
    of2 = 'F:/CWI/FeatBackup2/labelSave.txt'
    f = open(of, 'w')
    np.save(of, features)
    f.close()
    f = open(of2, 'w')
    np.save(of2, labels)
    f.close()
    #############################################################
    
    #FitandPredictSVM(0, 0, features, labels)
    

    
def FitandPredict(sclr, mxnt):

    of = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup2/featSave.txt.npy'
    of2 = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/FeatBackup2/labelSave.txt.npy'
    f = open(of, 'rb')
    f.seek(0)
    features = np.load(f)
    f.close()
    f = open(of2, 'rb')
    f.seek(0)
    labels = np.load(f)
    f.close()    
    
    #features = FEATURES[0]
    #labels = LABELS[0]
    print len(features)
    
    maxent = []
    maxent.append( svm.SVC(C=70, kernel='rbf', class_weight={1: 1.8}, gamma = 0.001) )
    maxent.append( svm.SVC(C=50, kernel='rbf', class_weight={1: 1.8}) )
    maxent.append( svm.SVC(C=30, kernel='rbf', class_weight={1: 1.8}, gamma = 0.001) )
    maxent.append( svm.SVC(C=70, kernel='rbf', class_weight={1: 1.8}, gamma = 0.1) )
    maxent.append(KNeighborsClassifier(n_neighbors = 10))
    maxent.append(tree.DecisionTreeClassifier(class_weight={1: 2}))
    maxent.append(ensemble.RandomForestClassifier())
    maxent.append(ensemble.ExtraTreesClassifier())
    from sklearn.ensemble import AdaBoostClassifier
    maxent.append(AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(class_weight={1: 2}), n_estimators=100))
    from sklearn.ensemble import GradientBoostingClassifier
    maxent.append(GradientBoostingClassifier(n_estimators=1000))
    from sklearn.ensemble import VotingClassifier
    maxent.append(VotingClassifier(estimators=[('svm', maxent[0]), ('dt', maxent[5])], voting='soft', weights=[5,10]))
    #maxent = svm.NuSVC(nu=0.5, kernel='rbf', class_weight='balanced');
    
    
    flen = len(features[0])
    print flen
    #plt.scatter(features[:, flen-1], features[:, 0], c=labels)
    #plt.show()
    
    '''
    tp = tn = fp = fn = 0
    l = len(labels)
    for i in range(l):

        if labels[i] == 0:
            fp += 1
        else:
            tp += 1

    print 'myprec ', float(tp)/(tp+fp)
    print 'myrecall ', float(tp)/(tp+fn)
    print 'accu ', f  
    '''
    #################################################################################################################
    from sklearn.feature_selection import SelectKBest, chi2
    selection = SelectKBest(chi2, k='all')
    features = selection.fit(features, labels).transform(features)
    
    '''orig_stdout = sys.stdout
    f = file('F:/CWI/rankedfeats.txt', 'w')
    sys.stdout = f
    top_ranked_features = sorted(enumerate(selection.scores_),key=lambda x:x[1], reverse=True)[:1354]
    top_ranked_features_indices = map(list,zip(*top_ranked_features))[0]
    for feature_pvalue in zip(np.asarray(VEC.get_feature_names())[top_ranked_features_indices],selection.pvalues_[top_ranked_features_indices]):
        print feature_pvalue
    sys.stdout = orig_stdout
    f.close()'''
    
    if sclr >= 0:
        scaler = []
        print "\n-------------------------- with Scaler ---------------------------  ", sclr
        scaler.append(preprocessing.RobustScaler(with_centering=False))
        scaler.append(preprocessing.Normalizer())
        scaler.append(preprocessing.StandardScaler())
        
        features = scaler[sclr].fit_transform(features)
    #####################################################################################################################
    
    for k in range(1):
        print "\n\n", maxent[mxnt].get_params()
        
        scores = defaultdict(list)
        cnt = 0;
        for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
    
            TrainX_i = features[TrainIndices]
            Trainy_i = labels[TrainIndices]
    
            TestX_i = features[TestIndices]
            Testy_i =  labels[TestIndices]
    
            maxent[mxnt].fit(TrainX_i,Trainy_i)         
            ypred_i = maxent[mxnt].predict(TestX_i)
    
            a = accuracy_score(Testy_i, ypred_i)
            p = precision_score(Testy_i, ypred_i)
            r = recall_score(Testy_i, ypred_i)
            print r
            scores["Accuracy"].append(accuracy_score(Testy_i, ypred_i))
            scores["F1"].append(f1_score(Testy_i, ypred_i))
            scores["Precision"].append(precision_score(Testy_i, ypred_i))
            scores["Recall"].append(recall_score(Testy_i, ypred_i))
            
    
            if a == 0 and r == 0:
                FF = 0
            else:
                FF = 2 * (a * r)/(a + r)
            scores["FF score"].append(FF)
            
            cm = confusion_matrix(Testy_i, ypred_i)
            print(cm)
            cnt += 1
            #if cnt == 4:
            #   break
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


    
    print("--- Next is grid search-----")
    print("========")

    
    #sys.exit(0)
    
    
def GridSearchOnData(features, labels, flag):
    
    print "============================= GRID SEARCHING NOW ================================"
    
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.cross_validation import train_test_split
    
    '''X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.75, random_state=0)'''


    tuned_parameters = [{'kernel': ['rbf'], 'class_weight': [{1: 1.8}, {1: 2}], 'gamma': [ 1, 0.1, 0.01 , 1e-3, 1e-4],
                    'C': [1, 5, 10, 100, 1000]}]
                    
    if flag == 1:
        tuned_parameters = [{'kernel': ['rbf'], 'class_weight': [None], 'gamma': [ 1, 0.1, 0.01 , 1e-3, 1e-4],
                        'C': [1, 5, 10, 100, 1000]}]
                        
    scaler = []
    scaler.append(preprocessing.Normalizer())
    scaler.append(preprocessing.RobustScaler(with_centering = False))
    
    
    
    for k in range(2):
        features = scaler[k].fit_transform(features)
        scores = ['recall', 'accuracy']
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()
        
            clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, cv=5,
                            scoring='%s' % score)
            #clf.fit(X_train, y_train)

            clf.fit(features, labels)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            for params, mean_score, scores in clf.grid_scores_:
                print("%0.3f (+/-%0.03f) for %r"
                    % (mean_score, scores.std() * 2, params))
            print()
        
            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            #y_true, y_pred = y_test, clf.predict(X_test)
            #print(classification_report(y_true, y_pred))
            print()
  
        
                                             
                                            
######################   RUN ##############################
FEATURES = []
LABELS = []
#MainClassifier()
FitandPredSVM_MAIN(0)
#BasicClassifier()
#MyClassifier()
#SplitClassifier()
#MainClassifier()
#FitandPredict(0, 0)
#FitandPredict(0, 1)
#FitandPredict(0, 2)
#FitandPredict(0, 9)
#FitandPredict(0, 10)
#FitandPredict(0, 5)
sys.exit(0)

orig_stdout = sys.stdout
f = file('F:/CWI/out_0.txt', 'w')
sys.stdout = f
FitandPredict(0, 0)
FitandPredict(0, 1)
FitandPredict(0, 2)
FitandPredict(0, 3)
FitandPredict(0, 4)
FitandPredict(0, 5)
FitandPredict(0, 6)
FitandPredict(0, 7)
sys.stdout = orig_stdout
f.close()
print ' =========================== SHIFTING SCALER ==================================='

orig_stdout = sys.stdout
f = file('F:/CWI/out_1.txt', 'w')
sys.stdout = f
FitandPredict(1, 0)
FitandPredict(1, 1)
FitandPredict(1, 2)
FitandPredict(1, 3)
FitandPredict(1, 4)
FitandPredict(1, 5)
FitandPredict(1, 6)
FitandPredict(1, 7)
sys.stdout = orig_stdout
f.close()
print ' =========================== SHIFTING SCALER ==================================='

orig_stdout = sys.stdout
f = file('F:/CWI/out_2.txt', 'w')
sys.stdout = f
FitandPredict(2, 0)
FitandPredict(2, 1)
FitandPredict(2, 2)
FitandPredict(2, 3)
FitandPredict(2, 4)
FitandPredict(2, 5)
FitandPredict(2, 6)
FitandPredict(2, 7)
sys.stdout = orig_stdout
f.close()


sys.exit(0)
    
    
GridSearchOnData(features, labels, 0)
print ()
print '------==-------==---------==---------'
print ()
GridSearchOnData(features, labels, 1)
    
sys.exit(0)
print "Korlev 1"
   
    
    