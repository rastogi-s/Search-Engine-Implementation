'''
Created on March 15, 2018

@author: shubham rastogi
'''
import glob
import os
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# corpus directory without stopped and stemmed data
DIR_CORPUS='../CorpusGeneration/TokenizedCorpus'
# corpus directory with stopped data
DIR_CORPUS_WITH_STOPPING='../CorpusGeneration/TokenizedCorpusWithStopping'
# corpus directory with stemmed data
DIR_CORPUS_WITH_STEMMING='../CorpusGeneration/TokenizedCorpusWithStemming'
# directory for indexing pickle files
DIR_PICKLE_FILES='../Indexing/IndexPickleFiles/'
# directory for index text files
DIR_INDEX_TEXT_FILES='../Indexing/IndexTextFiles/'


# method to create index
#  TERM:{docId:tf}
def buildIndex(corpusDirectory):
    invertedList={}
    noOfTokensInDoc={}
    path=os.path.join(corpusDirectory,r"*.txt")
    tokenFiles=glob.glob(path)
    for fName in tokenFiles:
        f=open(fName,'r')
        listOfTokens=f.read().split()
        docName=fName.split('\\')[1].split('.')[0]
        noOfTokensInDoc[docName]=len(listOfTokens)
        for token in listOfTokens:
            if not token in invertedList:
                invertedList[token]={docName:1}
            else:
                doc=invertedList[token]
                if not docName in doc: 
                    doc[docName]=1
                else:
                    doc[docName]+=1
    
    print "No. of indexed terms: "+str(len(invertedList))
    return (invertedList,noOfTokensInDoc)
    

# create pickle files for unigram index
def writeIndexToPickleFile(index,mode):
    if not os.path.exists(DIR_PICKLE_FILES):
        os.makedirs(DIR_PICKLE_FILES)
    fName=DIR_PICKLE_FILES+'unigram-'+mode+'-index.pickle'
    if os.path.exists(fName):
        os.remove(fName)
    fileIndex=open(fName,'wb')
    pickle.dump(index,fileIndex)
    fileIndex.close()
  
# create text files for each unigram index
def writeIndexToTextFile(index,mode):
    if not os.path.exists(DIR_INDEX_TEXT_FILES):
        os.makedirs(DIR_INDEX_TEXT_FILES)
    fName=DIR_INDEX_TEXT_FILES+'unigram-'+mode+'-index.txt'
    if os.path.exists(fName):
        os.remove(fName)
    fileIndex=open(fName,'w')
    fileIndex.write('##########################################################################################\n')
    fileIndex.write('################################ Unigram Inverted Index ##################################\n')
    fileIndex.write('##########################################################################################\n\n\n')
    for term in index:
        fileIndex.write("'"+term+"' : ")
        docDic=index[term]
        s=""
        for doc in docDic:
            s+='('+doc+', '+str(docDic[doc])+'), '
        s=s[:-2]
        fileIndex.write(s+"\n")
    
    fileIndex.close()


def generateNoOfTermsPerDocFile(noOfTokensPerDoc,mode):
    if not os.path.exists(DIR_INDEX_TEXT_FILES):
        os.makedirs(DIR_INDEX_TEXT_FILES)
    fName=DIR_INDEX_TEXT_FILES+'NoTokensPerDoc-'+mode+'.txt'
    if os.path.exists(fName):
        os.remove(fName)
    fileIndex=open(fName,'w')
    fileIndex.write('##########################################################################################\n')
    fileIndex.write('#################################### Tokens Per Document #################################\n')
    fileIndex.write('##########################################################################################\n\n\n')
    for doc in noOfTokensPerDoc:
        fileIndex.write("'"+doc+"':  "+str(noOfTokensPerDoc[doc])+"\n")
    fileIndex.close()
    if not os.path.exists(DIR_PICKLE_FILES):
        os.makedirs(DIR_PICKLE_FILES)
    fName=DIR_PICKLE_FILES+'NoTokensPerDoc-'+mode+'.pickle'
    if os.path.exists(fName):
        os.remove(fName)
    fileIndex=open(fName,'wb')
    pickle.dump(noOfTokensPerDoc,fileIndex)
    fileIndex.close()
    
def selectTheCorpusForIndexing(x):
    if x==1:
        corpusDirectory=DIR_CORPUS
        mode="no_stopping_or_stemming"
    elif x==2:
        corpusDirectory=DIR_CORPUS_WITH_STOPPING
        mode="withStopping"
    elif x==3:
        corpusDirectory=DIR_CORPUS_WITH_STEMMING
        mode="withStemming"

    print('Generating Unigram Index.......')
    unigramIndex,noOfTokensInDoc=buildIndex(corpusDirectory)
    generateNoOfTermsPerDocFile(noOfTokensInDoc,mode)   
    writeIndexToPickleFile(unigramIndex,mode)
    writeIndexToTextFile(unigramIndex,mode)    
                

def selectTheCorpusForIndexing1():
    while True:
        print "\nSelect the corpus for indexing:"
        print "Enter 1 No Stopping and Stemming"
        print "Enter 2 With Stopping"
        print "Enter 3 With Stemming"
        print "Enter 4 to exit!!" 
        x=input()
        if x==1:
            corpusDirectory=DIR_CORPUS
            mode="no_stopping_or_stemming"
        elif x==2:
            corpusDirectory=DIR_CORPUS_WITH_STOPPING
            mode="withStopping"
        elif x==3:
            corpusDirectory=DIR_CORPUS_WITH_STEMMING
            mode="withStemming"
        else:
            break

        print('Generating Unigram Index.......')
        unigramIndex,noOfTokensInDoc=buildIndex(corpusDirectory)
        generateNoOfTermsPerDocFile(noOfTokensInDoc,mode)   
        writeIndexToPickleFile(unigramIndex,mode)
        writeIndexToTextFile(unigramIndex,mode)     
    
if __name__=='__main__':
    selectTheCorpusForIndexing1()