import math
import os
import pickle
import re
import string
from bs4 import BeautifulSoup
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# file name which contains list of queries
LIST_OF_QUERY_FILE_NAME='../CASM-Files/cacm.query.txt'
# output file name storing the top 100 results of BM25 score for all queries 
TOP_100_RESULT_BM25='Top_100_Query_Result_BM25.txt'
# output file name storing the top 100 results of TF-IDF score for all queries 
TOP_100_RESULT_TF_IDF='Top_100_Query_Result_TF-IDF.txt'
# output file name storing the top 100 results of Smoothed Query Likelihood Model score for all queries 
TOP_100_RESULT_QueryLikelihood='Top_100_Query_Result_QueryLikelihoodModel.txt'
# pickle file of inverted index 
INVERTED_INDEX=['../Indexing/IndexPickleFiles/unigram-no_stopping_or_stemming-index.pickle',
                '../Indexing/IndexPickleFiles/unigram-withStopping-index.pickle',
                '../Indexing/IndexPickleFiles/unigram-withStemming-index.pickle'] 
# pickle file of number of tokens per document in corpus
NUM_OF_TOKEN_PER_DOC=['../Indexing/IndexPickleFiles/NoTokensPerDoc-no_stopping_or_stemming.pickle',
                      '../Indexing/IndexPickleFiles/NoTokensPerDoc-withStopping.pickle',
                      '../Indexing/IndexPickleFiles/NoTokensPerDoc-withStemming.pickle'] 
# path of list of relevant document files
RELEVANT_DOCS='../CASM-Files/cacm.rel.txt'
# list of stemmed queries
STEMMED_QUERIES='../CASM-Files/cacm_stem.query.txt'
# Coefficient to control probability of unseen words
COEFFICIENT=0.35
# directory for no text transformation
DIR_OUTPUT='../Retrieval/No Text Transformation Runs Output'

# calculate BM25 score of each doc that has the query term
# input : inverted index of unigram, query term frequency dictionary
# output : a dictionary of docids storing the score in decreasing order
def calculateBM25Score(invertedIndex,queryTermFreq,noOfTokenPerDoc,queryID):
    docScore={}
    # relevance info
    relevantDocIDs= fetchRelevantDocIds(queryID)
    R=len(relevantDocIDs)
    # parameters value set empirically
    k1, k2, b = (1.2,100,0.75)
    # Total number of document in the corpus
    N = len(noOfTokenPerDoc)
    # Average document length
    avgDocLen = sum([ noOfTokenPerDoc[doc] for doc in noOfTokenPerDoc])/len(noOfTokenPerDoc)
    
    for qTerm in queryTermFreq:
        # fetch inverted list for the query term if it exists in the index
        if qTerm in invertedIndex:
            invertedList=invertedIndex[qTerm]
        else:
            continue
        # document length
        docLength=noOfTokenPerDoc[doc]
        # the number of documents in which qTerm occurs
        n = len(invertedList)
        # K : one of the parameters
        K = k1*((1-b) + b*(docLength)/float(avgDocLen)) 
        # query term frequency
        qf=queryTermFreq[qTerm]
        # fetch relevance info 
        r = len([docs for docs in relevantDocIDs if docs in invertedList])
        
        for doc in invertedList:
            # the number of times qTerm occur in the current doc
            f = invertedList[doc]
            param1 = math.log( ((r+0.5)/(R-r+0.5)) / ((n-r+0.5)/(N-n-R+r+0.5)))
            param2 = ((k1 + 1)* f)/(K+f)
            param3=  ((k2 + 1)* qf)/(k2+qf)
            score=param1*param2*param3
            if doc in docScore:
                docScore[doc]+=score
            else:
                docScore[doc]=score     
           
    return docScore

# calculate tf-idf score of each doc
# input : inverted index of unigram, query term frequency dictionary
# output : a dictionary of docids storing the score
def calculateTFIDF(invertedIndex,queryTermFreq,noOfTokenPerDoc):
    docScore={}
    idf={}
    tfidf={}
    newIndex={}
    
    for qTerm in queryTermFreq:
        if qTerm in invertedIndex:
            newIndex[qTerm]= invertedIndex[qTerm]
        else:
            newIndex[qTerm]={}
    
    for term in newIndex:
        idf=math.log(len(noOfTokenPerDoc)/float(len(newIndex[term])+1))
        for doc in newIndex[term]:
            tf= newIndex[term][doc]/float(noOfTokenPerDoc[doc])
            if term not in tfidf:
                tfidf[term]={}
            tfidf[term][doc]=tf*idf
            if doc not in docScore:
                docScore[doc]=tf*idf
            else:
                docScore[doc]+=tf*idf
           
    return docScore


# calculate Smoothed Query Likelihood score of each doc that has the query term
# input : inverted index of unigram, query term frequency dictionary
# output : a dictionary of docids storing the score
def calculateSMQL(invertedIndex,queryTermFreq,noOfTokenPerDoc):
    docScore={}
    # size of collection
    C=sum([noOfTokenPerDoc[doc] for doc in noOfTokenPerDoc])
    
    for qTerm in queryTermFreq:
        if qTerm not in invertedIndex:
            continue
        invertedList=invertedIndex[qTerm]
        # frequency of query term over entire collection
        cq=sum([invertedList[doc] for doc in invertedList])
        for doc in invertedList:
            # frequency of query term in doc
            fq=invertedList[doc]
            # document size
            docSize=noOfTokenPerDoc[doc]
            unseenPart=COEFFICIENT* cq / float(C)
            seenPart=(1-COEFFICIENT)*fq/float(docSize)
            if doc not in docScore:
                docScore[doc]=math.log(seenPart+unseenPart)
            else:
                docScore[doc]+=math.log(seenPart+unseenPart)
            
    return docScore

# fetch relevant docIDs for the given query id
def fetchRelevantDocIds(queryID):
    relDocIds=[]
    fName=RELEVANT_DOCS
    relFile=open(fName,'r')
    for rec in relFile.readlines():
        record=rec.split()
        if record[0]==str(queryID):
            relDocIds.append(record[2])
             
    return relDocIds

    
    
# generate the term frequency of each query term.
# input : query
# output : a dictionary of all terms and their frequency
def generateQueryTermsFreqDict(query):
    queryTermFreq={}
    for qTerm in query.split():
        if qTerm in queryTermFreq:
            queryTermFreq[qTerm]+=1
        else:
            queryTermFreq[qTerm]=1
    
    return queryTermFreq

# fetch the inverted index of unigram from the pickle file
# input : pickle file of the index
# output : inverted index of the unigrams
def fetchInvertedIndex(invertedIndexFile):
    f=open(invertedIndexFile)
    invertedIndex=pickle.load(f)
    
    return invertedIndex

# fetch the number of tokens per document from the pickle file
# generated in previous assignment
# output : number of tokens per document 
def fetchNoOfTokensPerDocDic(noOfTokensFile):
    f=open(noOfTokensFile)
    noOfTokensPerDoc=pickle.load(f)
    
    return noOfTokensPerDoc

# write the result (BM25 score for the given query
def writeResultToFile(docScore,qID,model,outputFileName):
    fileModel=open(outputFileName,'a')
    fileModel.write("\nQuery Q"+str(qID)+"\n\n")
    sortedDocScore=sorted(docScore.iteritems(),key=lambda(k,v):(v,k),reverse=True)
    count=0
    for doc,score in sortedDocScore:
        fileModel.write(str(qID)+" Q0 "+doc +" "+str(count + 1) + " " + str(score) +" "+ model+"\n") 
        count+=1
        if count+1 > 100:
            break  
    fileModel.close()
    

def fetchQueryMap():
    queryMap={}
    f=open(LIST_OF_QUERY_FILE_NAME,'r')
    content =f.read()
    content='<DATA>'+content+'</DATA>'
    soup = BeautifulSoup(content, 'xml')
    docList= soup.findAll('DOC')
    for doc in docList:
        child=doc.findChild()
        qID=int(child.get_text().encode('utf-8'))
        child.decompose()
        text = doc.get_text().encode('utf-8')
        caseFoldedtext= caseFold(text)
        tokens = generateTokens(caseFoldedtext)
        refinedText=removePunctuation(tokens)
        queryMap[qID]=refinedText
    
    return queryMap
         
    f.close()
    
# method to case-fold the text provided
# Given: plain text
# Return: case folded plain text  
def caseFold(plainText):
    return plainText.lower()

# method to remove punctuation from the text provided
# Given: plain text
# Return:  plain text with removed punctuation
def removePunctuation(tokens):
    newList=[]
    for tok in tokens:
        tok=tok.strip(string.punctuation)
        matchNum=re.compile(r'^[\-]?[0-9]*\.?[0-9]+$')
        if not matchNum.match(tok):
            tok=re.sub(r'[^a-zA-Z0-9\--]','',tok)
            tok=tok.strip(string.punctuation)
        newList.append(tok)
            
    return  (' ').join(newList)   
    
# method to generate tokens from plain text
# Given: plain text
# Return: list of tokens     
def generateTokens(plainText):
    return list(filter(re.compile('[a-zA-Z0-9_]').search,plainText.split()))

# fetch stemmed queries
def fetchStemmedQueries():
    queryMap={}
    f=open(STEMMED_QUERIES,'r')
    i=1
    for query in f.readlines():
        queryMap[i]=query 
        i+=1
    f.close()
    return queryMap


def selectRetrievalModel(invertedIndexFile,noOfTokensFile,x,outputDirectory,queryMap):

    if x==1:
        model="BM25"
        outputFileName=outputDirectory+"/"+TOP_100_RESULT_BM25
    elif x==2:
        model="TF-IDF"
        outputFileName=outputDirectory+"/"+TOP_100_RESULT_TF_IDF
    elif x==3:
        model="Smoothed Query Likelihood Model"
        outputFileName=outputDirectory+"/"+TOP_100_RESULT_QueryLikelihood
    
    fileModel=open(outputFileName,'w')
    topic=" Top 100 Query Results Using "+model+" "
    hashLen=90-len(topic)
    hashLen=hashLen/2
    filler="#"*hashLen+topic+"#"*hashLen
    if len(filler)<90:
        filler+="#"
    fileModel.write("#"*90+"\n"+filler+"\n"+"#"*90+"\n\n")
    fileModel.close()
    print "Loading inverted index from pickle file...."
    invertedIndex=fetchInvertedIndex(invertedIndexFile)
    print "Loading number of tokens per document from pickle file...."
    noOfTokenPerDoc=fetchNoOfTokensPerDocDic(noOfTokensFile)
    queryList=sorted(queryMap)
    docScorePerQuery={}
    for queryID in queryList:
        print "\nQuery --> "+str(queryID)+": " +queryMap[queryID]
        print "Generating query term frequency...."
        queryTermFreq=generateQueryTermsFreqDict(queryMap[queryID])
        print "Calculating "+model+" score for documents for the current query...."
        if model=="BM25":
            docScore=calculateBM25Score(invertedIndex,queryTermFreq,noOfTokenPerDoc,queryID)
        elif model=="TF-IDF":
            docScore=calculateTFIDF(invertedIndex,queryTermFreq,noOfTokenPerDoc)
        elif model=="Smoothed Query Likelihood Model":
            docScore=calculateSMQL(invertedIndex,queryTermFreq,noOfTokenPerDoc)
        print "Writing the top 100 results in the file...."
        writeResultToFile(docScore,queryID,model,outputFileName)
        docScorePerQuery[queryID]=docScore
    return docScorePerQuery

# default retrieval for no text transformation 
def selectRetrievalModel1(invertedIndexFile,noOfTokensFile):
    while True:
        print "\nSelect the retrieval model from the below list:"
        print "Enter 1 for BM25 retrieval model"
        print "Enter 2 for tf-idf retrieval model"
        print "Enter 3 for Smoothed Query Likelihood retrieval model"
        print "Enter 4 for exit"
        x=input()
        if x==1:
            model="BM25"
            outputFileName=DIR_OUTPUT+"/"+TOP_100_RESULT_BM25
        elif x==2:
            model="TF-IDF"
            outputFileName=DIR_OUTPUT+"/"+TOP_100_RESULT_TF_IDF
        elif x==3:
            model="Smoothed Query Likelihood Model"
            outputFileName=DIR_OUTPUT+"/"+TOP_100_RESULT_QueryLikelihood
        else:
            print "Exiting !!!!"
            break
        
        
        if not os.path.exists(DIR_OUTPUT):
            os.makedirs(DIR_OUTPUT)
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        fileModel=open(outputFileName,'a')
        topic=" Top 100 Query Results Using "+model+" "
        hashLen=90-len(topic)
        hashLen=hashLen/2
        filler="#"*hashLen+topic+"#"*hashLen
        if len(filler)<90:
            filler+="#"
        fileModel.write("#"*90+"\n"+filler+"\n"+"#"*90+"\n\n")
        fileModel.close()
        print "Loading inverted index from pickle file...."
        invertedIndex=fetchInvertedIndex(invertedIndexFile)
        print "Loading number of tokens per document from pickle file...."
        noOfTokenPerDoc=fetchNoOfTokensPerDocDic(noOfTokensFile)
     
        queryMap=fetchQueryMap()
        queryList=sorted(queryMap)
        for queryID in queryList:
            print "\nQuery --> "+str(queryID)+": " +queryMap[queryID]
            print "Generating query term frequency...."
            queryTermFreq=generateQueryTermsFreqDict(queryMap[queryID])
            print "Calculating "+model+" score for documents for the current query...."
            if model=="BM25":
                docScore=calculateBM25Score(invertedIndex,queryTermFreq,noOfTokenPerDoc,queryID)
            elif model=="TF-IDF":
                docScore=calculateTFIDF(invertedIndex,queryTermFreq,noOfTokenPerDoc)
            elif model=="Smoothed Query Likelihood Model":
                docScore=calculateSMQL(invertedIndex,queryTermFreq,noOfTokenPerDoc)
            print "Writing the top 100 results in the file...."
            writeResultToFile(docScore,queryID,model,outputFileName)
  
if __name__=='__main__':
    selectRetrievalModel1(INVERTED_INDEX[0],NUM_OF_TOKEN_PER_DOC[0])