import math
import os
import random
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Evaluation import PerformanceEvaluation
from Retrieval import RetrievalModels


SPELL_ERROR_FILE='../SpellErrorGenerator/SpellingErrorInducedQueries.txt'
TYPE_OF_OUTPUTS=['../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output']

def errorGenerator(query):
    tokenList=query.split()
    tokenMap={}
    for token in tokenList:
        tokenMap[token]=len(token)
    sortedTokensOnLength=sorted(tokenMap.iteritems(),key=lambda(k,v):(v,k),reverse=True)
    lengthOfQuery=len(sortedTokensOnLength)
    numOfErrorsToGenerate=int(math.floor((lengthOfQuery*40)/float(100)))
    for i in range(numOfErrorsToGenerate):
        oldWord=sortedTokensOnLength[i][0]
        start=oldWord[0:1]
        end=oldWord[-1:]
        oldReduced=oldWord[1:-1]
        newReduced=''.join(random.sample(oldReduced,len(oldReduced)))
        newWord=start+newReduced+end
        query=query.replace(oldWord,newWord)
    return query
    
def writeSpellingErrorInducedQueries(newQueries):
    fileName=SPELL_ERROR_FILE
    if os.path.exists(fileName):
        os.remove(fileName)
    f=open(fileName,'w')
    topic=" Spelling Error Induced Queries "
    hashLen=90-len(topic)
    hashLen=hashLen/2
    filler="#"*hashLen+topic+"#"*hashLen
    if len(filler)<90:
        filler+="#"
    f.write("#"*90+"\n"+filler+"\n"+"#"*90+"\n\n")
    for queryID in newQueries:
        f.write("Q"+str(queryID)+" --> "+newQueries[queryID]+"\n")
    f.close()

def main(qmap):
    print "Spelling Error Generation!!!"
    newQueries={}
    for queryID in qmap:
        newQueries[queryID]=errorGenerator(qmap[queryID])
    writeSpellingErrorInducedQueries(newQueries)
    
    return newQueries
    
if __name__=='__main__':

    qmap=RetrievalModels.fetchQueryMap()
    newQueries=main(qmap)
    if os.path.exists(TYPE_OF_OUTPUTS[5]):
        shutil.rmtree(TYPE_OF_OUTPUTS[5])
    if not os.path.exists(TYPE_OF_OUTPUTS[5]):
        os.makedirs(TYPE_OF_OUTPUTS[5])
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],\
                                                          RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[4],newQueries)
    DOC_SCORES_PER_QUERY_PER_RUN=PerformanceEvaluation.fetchDocScoresPerQueryPerRun()
    DOC_SCORES_PER_QUERY_PER_RUN['ErrorInducedQuery-BM25']=docScorePerQuery
    PerformanceEvaluation.main(DOC_SCORES_PER_QUERY_PER_RUN)