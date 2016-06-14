import sys
import nltk
import re

#ogden = [
#f = 'F:/CWI/cwi_training/ogden.txt'
fname = 'D:/SEMEVAL/FROM_HOME_NEW/CWI/result_C50_12_001_robust.txt'

f = open('D:/SEMEVAL/FROM_HOME_NEW/CWI/result_C50_12_001_robust_REDUCED.txt', 'w')
#fname = 'F:/CWI/backup/result_C5main_weight15.txt'
#with open(f) as temp_file:
#  ogden = [line.rstrip(',') for line in temp_file]

#print ogden[0]
#ogden[0] = re.sub("[^a-zA-Z]", " ", ogden[0])
#og = nltk.word_tokenize(ogden[0])
#print '\n\n',og[1]

#if (',' in og) or (' ' in og):
#    print 'what the FUCK!!!!'
    
#if 'this' in og:
#    print 'This can be done brother!!!!!!!!!!!!!!!!!!!!!!!!'
#f.close()



############################################################################################################################
with open(fname) as temp_file:
  preds = [line.rstrip('\n') for line in temp_file]
  
l = len(preds)
print l

x = []

for i in range(l):
    x.append(int(float(preds[i])))
    
#print x

for i in range(len(x)):
    f.write('%d' % x[i])
    f.write('\n')
f.close()
    

labels = x

###################################################################
print '\nAll One\n'

tp = tn = fp = fn = 0
l = len(labels)
for i in range(l):

    if labels[i] == 0:
        
        fp += 1
    else:
        tp += 1

print 'counts ', fp, tp


r = float(tp)/(tp+fn)
a = float(tp + tn) / (tp + tn + fp + fn)
if a == 0 and r == 0:
    FF = 0
else:
    FF = (2 * (a * r))/(a + r)
print 'myrecall ', float(tp)/(tp+fn)
print 'accu ', float(tp + tn) / (tp + tn + fp + fn)
print 'FF', FF
##################################################################


###################################################################
print '\nAll Zero\n'

tp = tn = fp = fn = 0
l = len(labels)
for i in range(l):

    if labels[i] == 0:
        tn += 1
    else:
        fn += 1

print 'counts ', tn, fn

r = float(tp)/(tp+fn)
a = float(tp + tn) / (tp + tn + fp + fn)
if a == 0 and r == 0:
    FF = 0
else:
    FF = (2 * (a * r))/(a + r)
print 'myrecall ', float(tp)/(tp+fn)
print 'accu ', float(tp + tn) / (tp + tn + fp + fn)
print 'FF', FF
##################################################################

'''
###################################################################
print '\nOgden\n'

tp = tn = fp = fn = 0
l = len(labels)
for i in range(l):

    if labels[i] == 0:
        fp += 1
    else:
        tp += 1

r = float(tp)/(tp+fn)
a = float(tp + tn) / (tp + tn + fp + fn)
if a == 0 and r == 0:
    FF = 0
else:
    FF = (2 * (a * r))/(a + r)
print 'myrecall ', float(tp)/(tp+fn)
print 'accu ', float(tp + tn) / (tp + tn + fp + fn)
print 'FF', FF
##################################################################
'''
#fname.close()