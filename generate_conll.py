import sys, os

#infile = open("F:\CWI\cwi_training.zip")
infile = []
procfile = open("D:/SEMEVAL/FROM_HOME_NEW/CWI/cwi_testing/cwi_testing.txt")
outfile = open("D:/SEMEVAL/FROM_HOME_NEW/CWI/cwi_testing/cwi_testing.txt2.conll", "w")
#outfile.write("# idx\tform\tlemma\tpos\tne\thead\tdeprel\n\n")

#cwiBuffer = infile.readline()
cwiBuffer = []

def cutLine(cwiLine):
    try:
        sent, word, idx, label = cwiLine.strip().split("\t")
        return sent, word, int(idx), label
    except ValueError:
        return None

def consumeSentCwi():
    global cwiBuffer
    decisions = {}
    line = infile.readline()
    bufferCut = cutLine(cwiBuffer)
    decisions[bufferCut[2]] = bufferCut[3]
    lineCut = cutLine(line)
    while lineCut and lineCut[0] == bufferCut[0]:
        decisions[lineCut[2]] = lineCut[3]
        line = infile.readline()
        lineCut = cutLine(line)
    cwiBuffer = line
    return decisions
    
    
targetWords = []

def consumeSentProc():
    sent = []
    line = procfile.readline().strip()
    prev0 = prev1 = None
    count = 0
    
    tw = []
    line_prev = ""
    spl_prev = line_prev.split()
    
    while not line == "":

         spl = line.split()
         
         #print spl[0], spl[1], '----------------', spl[len(spl)-2]
         #spl[0] = str(int(spl[0])-1)
         
         
         if spl[:len(spl)-3] == spl_prev[:len(spl)-3]:
            tw.append(spl[len(spl)-2])
         else:
            targetWords.append(tw)
            tw = []
            tw.append(spl[len(spl)-2])
        
         line_prev = line
         spl_prev = line_prev.split()
         #print line_prev
         
         count = 1
         prev0 = spl[0]
         prev1 = spl[1]
         
         line = "\t".join(spl)
         sent.append(line)
         line = procfile.readline().strip()
         if line == "":
             targetWords.append(tw)
             
    return sent

i = 0
while (i==0):
    sent = consumeSentProc()
    if sent != []:
        #sys.exit(0)
    
        #print sent
        print targetWords
        print len(targetWords)
        '''
        #decisions = consumeSentCwi()
        i = 0
        while i < len(sent):
            #label = decisions.get(i, "-")
            #outfile.write(sent[i]+"\t%s\n" %label)
            outfile.write(sent[i]+"\n")
            i += 1
        outfile.write("\n")
    '''
    i += 1

#infile.close()
outfile.close()