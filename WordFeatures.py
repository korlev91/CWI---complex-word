import argparse
import os, sys
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
import nltk as nt
from nltk.corpus import wordnet as wn
import porter
from utilities import *
#import porter
import numpy as np
import collections

#nt.download('cmudict')

class WordClass:

    def __init__(self,sentence,index,word,lemma,pos,namedentity,positive_votes,heads,deprels,cm_dict,bwords,prefix_keys,suffix_keys,ogden):
        self.sentence = sentence #sentence is a list of forms
        self.word = word
        self.index = int(index)
        self.positive_votes = int(positive_votes)
        self.label = int(self.positive_votes > 0)
        self.lemma = lemma
        self.pos = pos
        self.a_namedentity = namedentity
        self._heads = [int(h) for h in heads] #"protected" property
        self._deprels = deprels             #"protected" property
        self.deptree = self._create_deptree() #remember to use self.index+1 to access self.deptree
        self.sylcnt = 0
        self.d = cm_dict
        self.bw = bwords
        #self.scored = scored
        self.ogden = ogden
        self.prefix_keys = prefix_keys
        self.suffix_keys = suffix_keys
    
    def _create_deptree(self):
        deptree = nx.DiGraph()
        for idx_from_zero,head in enumerate(self._heads):
            #deptree.add_node(idx_from_zero+1) # the Id of each node is its Conll index (i.e. from one)
            deptree.add_edge(head, idx_from_zero+1, deprel=self._deprels[idx_from_zero]) #edge from head to dependent with one edge labeling called deprel
        return deptree

    def a_simple_feats(self): 
        D = {}

        D["a_pos"] = self.pos
        D["a_namedentity"] = self.a_namedentity
        D["a_formlength"] = len(self.word)
        D["a_lemma"] = self.lemma
        #print 1
        return D

    def a_simple_feats_lexicalized(self): 
        D = {}
        D["a_form"] = self.word
        #D["a_form"] = 'was'
        D["a_lemma"] = self.lemma
        
        D["a_pos"] = self.pos
        D["a_namedentity"] = self.a_namedentity
        #D["a_formlength"] = len(self.word)
        #print 2
        return D


    def b_wordnet_feats(self): 
        D = {}
        #D["b_nsynsets"] = len(wn.synsets(self.word))
        
        syn = wn.synsets(self.word)
        freq=0
        for s in syn: 
            #print s.lemmas()[0].count()
            for l in s.lemmas():
                f=l.count()
                if f==0:
                    f = 1
                freq += f
        #print freq
        D["b_nsynsets"] = freq
        return D

    def b2_syllabels(self):
        D = {}
         
        def nsyl(w): 
            try:
                for x in self.d[w.lower()]:
                    for y in x:
                        if y[-1].isdigit():
                            self.sylcnt += 1

            except KeyError:
                self.sylcnt = 1
        nsyl(self.word)
        D["b2_sylcnt"] = self.sylcnt
        #print '---------------',self.sylcnt
        return D
    
    def b3_brown_cluster(self):
        D = {}
        #print self.bw[self.word]
        D["brown clust"] = self.bw.freq(self.word)*10000
        return D
            
    def d_frequency_feats(self):
        D = {}
        wProb = (-1)*prob(self.word, corpus="wp")
        wProbSimple = (-1)*prob(self.word, corpus="swp")
        D["d_freq_in_swp"] = wProbSimple
        D["d_freq_in_wp"] = wProb
        D["d_freq_ratio_swp/wp"] = wProbSimple / wProb
        # TODO D["d_freqrank_distance_swp/wp"] = rank(self.word, corpus="swp") - rank(self.word, corpus="wp")  
        # TODO D["d_distributional_distance_swp/wp"] = dist(dist_vector(self.word, "swp"), dist_vector(self.word, "wp"))  # get distributional vector from background corpora, use some dist measure
        #print 4
        return D
        
 ########################################## high level feats ###########################################       
    def c_positional_feats(self):
        D = {}
        #D["c_relativeposition"] =  int(self.index / len(self.sentence))
        before, after = commas_before_after(self.sentence, self.index)
        D["c_preceding_commas"] = before
        D["c_following_commas"] = after
        #before, after = verbs_before_after(self.sentence, self.index)
        #D["c_preceding_verbs"] = before
        #D["c_following_verbs"] = after
        #print 5
        return D
        

    def e_morphological_feats(self):
        D = {}
        #etymology = retrieve_etymology(self.lemma)
        #D["e_latin_root"] = has_ancestor_in_lang("lat", etymology)  # check wiktionary
        D["e_length_dist_lemma_form"] = len(self.word) - len(self.lemma)
        stem, steps = porter.stem(self.word)
        D["e_length_dist_stem_form"] = len(self.word) - len(stem)
        D["e_inflectional_morphemes_count"] = steps 
        
        #if D["e_length_dist_lemma_form"] < 0:
            #print self.word , self.lemma
            #D["e_length_dist_lemma_form"] = 0
        #print D
        return D

    def f_prob_in_context_feats(self):
        D = {}
        # TODO D["f_P(w|w-1)"]    = seq_prob(self.word, [self.sentence[self.index-1]]) # prob() uses freq()
        # TODO D["f_P(w|w-2w-1)"] = seq_prob(self.word, [self.sentence[self.index-2:self.index]]) # prob() uses freq()
        # TODO D["f_P(w|w+1)"]    = seq_prob(self.word, [self.sentence[self.index+1]]) # prob() uses freq()
        # TODO D["f_P(w|w+1w+2)"] = seq_prob(self.word, [self.sentence[self.index+1:self.index+3]]) # prob() uses freq()
        #print 7
        return D
    
    def g_char_complexity_feats(self):
        D = {}
        unigramProb = (-1)*prob(self.word, level="chars", order=1)
        unigramProbSimple = (-1)*prob(self.word, level="chars", corpus="swp", order=1)
        bigramProb = (-1)*prob(self.word, level="chars", order=2)
        bigramProbSimple = (-1)*prob(self.word, level="chars", corpus="swp", order=2)
        #print bigramProb, bigramProbSimple
        D["g_char_unigram_prob"] = unigramProb
        D["g_char_unigram_prob_ratio"] = unigramProbSimple / unigramProb
        D["g_char_bigram_prob"] = bigramProb
        D["g_char_bigram_prob_ratio"] = bigramProbSimple / bigramProb
        #D["g_vowels_ratio"] = float(count_vowels(self.word)) / len(self.word)
        #print 8
        #print D
        return D 
    
    def h_brownpath_feats(self):
        D={}
        #brown cluster path feature
        global brownclusters
        if self.word in brownclusters:
            D["h_cluster"] = brownclusters[self.word]
        #print 9
        return D
        
    def i_browncluster_feats(self):
        D={}
        #brown cluster path feature
        global brownclusters, ave_brown_height, ave_brown_depth
        if self.word in brownclusters:
            bc = brownclusters[self.word]
            for i in range(1,len(bc)):
                D["i_cluster_"+bc[0:i] ]=1
        
            #brown cluster height=general/depth=fringiness
            D["i_cluster_height"]=len(bc)
            D["i_cluster_depth"]=cluster_heights[bc]
        else:
            #taking average
            #D["i_cluster_height"]=ave_brown_height
            #D["i_cluster_depth"]=ave_brown_depth
            #taking extremes
            D["i_cluster_height"]=0
            D["i_cluster_depth"]=max_brown_depth
        #print 10
        return D
        
    def j_embedding_feats(self):
        D={}
        #word embedding
        global embeddings
        if self.word in embeddings.keys():
            emb=embeddings[self.word]
            for d in range(len(emb)):
                D["j_embed_"+str(d)]=float(emb[d])

        #TODO: (1) fringiness of embedding 
        #print 11
        return D

    def k_dependency_feats(self):
        wordindex = self.index + 1
        headindex = dep_head_of(self.deptree,wordindex)
        D = {}
        D["k_dist_to_root"] = len(dep_pathtoroot(self.deptree,wordindex))
        D["k_deprel"] = self.deptree[headindex][wordindex]["deprel"]
        D["k_headdist"] = abs(headindex - wordindex) # maybe do 0 for root?
        D["k_head_degree"] = nx.degree(self.deptree,headindex)
        D["k_child_degree"] = nx.degree(self.deptree,wordindex)
        #print 12
        return D

    def l_context_feats(self):
        wordindex = self.index + 1
        headindex = dep_head_of(self.deptree,wordindex)
        D = {}
        D["l_brown_bag"] = "_"
        D["l_context_embed"] = "_"
        #print 13
        return D

    def m_bigramfreq(self):
       D = {}
       # Group bigrams by first word in bigram.                                        

       #print self.sentence['form']
       sent = self.sentence["form"]
       l = len(sent)
       K = self.index
       '''
       for k in range(l):
           if sent[k] == self.word:
               K = k
               break
       '''
       weight_prefix = 0
       weight_suffix = 0
       
       for i in range(len(self.prefix_keys[self.word])):
           if (K+1) > (l-1):
               break
           if (sent[K+1] == self.prefix_keys[self.word][i][0]):
               weight_prefix += self.prefix_keys[self.word][i][1]
               
       for i in range(len(self.suffix_keys[self.word])):
           if (K-1) < 0:
               break
           if (sent[K-1] == self.suffix_keys[self.word][i][0]):
               weight_suffix += self.suffix_keys[self.word][i][1]
               
       #if weight_prefix > 0 or weight_suffix > 0:
         #  print self.word
       D["gram_weight_pre"] = weight_prefix 
       D["gram_weight_post"] = weight_suffix
       #print sent[K], ' ', D
       print D
       return D
       
    def m_surrounding(self):
       D = {}
       sent = self.sentence["form"]
       l = len(sent)
       #print sent 
       K = self.index
       '''
       for k in range(l):
           if sent[k] == self.word:
               K = k
               break
       '''
       #print K, l
       tagp = tagn = ""
       if (K+1) < l:
           tagn = nt.word_tokenize(sent[K+1])
           tagn = nt.pos_tag(tagn)     
       if (K-1) >=0:
           tagp = nt.word_tokenize(sent[K-1])
           tagp = nt.pos_tag(tagp)        
           
       if tagp != "":
           D["ptag"] = tagp[0][1]
       else: 
           D["ptag"] = ""
       if tagn != "":    
           D["ntag"] = tagn[0][1]
       else:
           D["ntag"] = ""
           
       print D
       return D 
    
    def n_ogden(self):
        D = {}
        D["in_ogden_dict"] = 0
        
        if self.word in self.ogden:
            D["in_ogden_dict"] = 1
            #print self.word, ' in ogden ', D["in_ogden_dict"]
        return D
        
    def featurize(self):
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        D.update(self.c_positional_feats())
        D.update(self.d_frequency_feats())
        D.update(self.e_morphological_feats())
        D.update(self.f_prob_in_context_feats())
        D.update(self.g_char_complexity_feats())
        D.update(self.h_brownpath_feats())
        D.update(self.i_browncluster_feats())
        D.update(self.j_embedding_feats())
        D.update(self.k_dependency_feats())
        D.update(self.l_context_feats())

        return D

    def featurize_lightweight(self): ## smaller set of features used for dev
        D = {}
        D.update(self.a_simple_feats())
        D.update(self.b_wordnet_feats())
        D.update(self.c_positional_feats())
        D.update(self.d_frequency_feats())
        D.update(self.f_prob_in_context_feats())
        D.update(self.g_char_complexity_feats())
        D.update(self.k_dependency_feats())
        D.update(self.l_context_feats())
        return D

    def baselinefeatures(self):
        D = {}
    
        #necessary
        D.update(self.a_simple_feats()) #lemma, POS lenght,ne
        D.update(self.b_wordnet_feats()) #sense count ---- try entropy
        
        #experimental
        #D.update(self.b2_syllabels())    #syllabel count cmu dict
        
        #frequency
        D.update(self.b3_brown_cluster())  #word count
        D.update(self.d_frequency_feats()) #wiki freq
        
        #important
        D.update(self.e_morphological_feats())
        
        ##### contextual ####################
        D.update(self.m_bigramfreq())
        #D.update(self.m_surrounding())
        
        #D.update(self.n_ogden())
        
        D.update(self.g_char_complexity_feats())
        #D.update(self.c_positional_feats()) #--- .27
        #D.update(self.k_dependency_feats())  #----- .32
        return D
        
        
#brownclusters, cluster_heights, ave_brown_depth, ave_brown_height, max_brown_depth=read_brown_clusters('/coastal/brown_clusters/rcv1.64M-c1000-p1.paths', 1000)
#embeddings=read_embeddings('/coastal/mono_embeddings/glove.6B.300d.txt.gz')