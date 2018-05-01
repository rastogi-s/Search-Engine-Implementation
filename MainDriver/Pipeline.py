import os
import pickle
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Display import DisplayResult
from CorpusGeneration import GenerateTokenizedCorpus
from Indexing import Indexer
from Evaluation import PerformanceEvaluation
from QueryEnhancement import QueryEnrichment
from Retrieval import RetrievalModels
from SoftMatching import SoftMatchingQuerHandler
from SpellErrorGenerator import SpellingErrorGenerator
from Utility import InitializeProject

DOC_SCORES_PER_QUERY_PER_RUN_PICKLE='../Utility/DOC_SCORES_PER_QUERY_PER_RUN.pickle'

TYPE_OF_OUTPUTS=['../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output']

# pickle file of inverted index 
INVERTED_INDEX=['../Indexing/IndexPickleFiles/unigram-no_stopping_or_stemming-index.pickle',
                '../Indexing/IndexPickleFiles/unigram-withStopping-index.pickle',
                '../Indexing/IndexPickleFiles/unigram-withStemming-index.pickle'] 
# pickle file of number of tokens per document in corpus
NUM_OF_TOKEN_PER_DOC=['../Indexing/IndexPickleFiles/NoTokensPerDoc-no_stopping_or_stemming.pickle',
                      '../Indexing/IndexPickleFiles/NoTokensPerDoc-withStopping.pickle',
                      '../Indexing/IndexPickleFiles/NoTokensPerDoc-withStemming.pickle'] 

DOC_SCORES_PER_QUERY_PER_RUN={}
ORDER_OF_EXECUTION=['BM25','TF-IDF','SQLM']  

def baselineRunsNoTextTrans():
    
    # generate tokens with no text transformation
    GenerateTokenizedCorpus.selectTypeOfTextTransformation(1)
    # generate unigram index with no text transformation corpus
    Indexer.selectTheCorpusForIndexing(1)
    # fetch queryMap
    qMap=RetrievalModels.fetchQueryMap()
    if os.path.exists(TYPE_OF_OUTPUTS[0]):
        shutil.rmtree(TYPE_OF_OUTPUTS[0])
    if not os.path.exists(TYPE_OF_OUTPUTS[0]):
        os.makedirs(TYPE_OF_OUTPUTS[0])
    # run all baselines (BM25, TF-IDF, Smoothed Query Likelihood Model)
    # with no text transformation corpus
    for i in range(1,4):
        docScorePerQuery=RetrievalModels.selectRetrievalModel(INVERTED_INDEX[0],NUM_OF_TOKEN_PER_DOC[0],i,TYPE_OF_OUTPUTS[0],qMap)
        DOC_SCORES_PER_QUERY_PER_RUN['NoTextTran-'+ORDER_OF_EXECUTION[i-1]]=docScorePerQuery
        
    
def queryEnrichment():
    # fetch queryMap
    qMap=RetrievalModels.fetchQueryMap()
    # run baseline for BM25 (no text transformation)
    docScorePerQuery=RetrievalModels.selectRetrievalModel(INVERTED_INDEX[0],NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[0],qMap)
    DOC_SCORES_PER_QUERY_PER_RUN['QueryRefinement-BM25']=QueryEnrichment.main(docScorePerQuery)


def baselineRunsWithStopping():
    global DOC_SCORES_PER_QUERY_PER_RUN
    # generate tokens with stopping as text transformation
    GenerateTokenizedCorpus.selectTypeOfTextTransformation(2)
    # generate unigram index with stopped  corpus
    Indexer.selectTheCorpusForIndexing(2)
    # fetch queryMap
    qMap=RetrievalModels.fetchQueryMap()
    if os.path.exists(TYPE_OF_OUTPUTS[1]):
        shutil.rmtree(TYPE_OF_OUTPUTS[1])
    if not os.path.exists(TYPE_OF_OUTPUTS[1]):
        os.makedirs(TYPE_OF_OUTPUTS[1])
    
    # run all baselines (BM25, TF-IDF, Smoothed Query Likelihood Model)
    # with stopped  corpus
    for i in range(1,4):
        # i=1. BM25 2. TFIDF 3. SQLM
        docScorePerQuery=RetrievalModels.selectRetrievalModel(INVERTED_INDEX[1],NUM_OF_TOKEN_PER_DOC[1],i,TYPE_OF_OUTPUTS[1],qMap)
        DOC_SCORES_PER_QUERY_PER_RUN['Stopped-'+ORDER_OF_EXECUTION[i-1]]=docScorePerQuery
        
def baselineRunsWithStemming():
    # generate tokens with stemming as text transformation
    GenerateTokenizedCorpus.selectTypeOfTextTransformation(3)
    # generate unigram index with stemmed  corpus
    Indexer.selectTheCorpusForIndexing(3)
    # fetch queryMap
    qMap=RetrievalModels.fetchStemmedQueries()
    if os.path.exists(TYPE_OF_OUTPUTS[2]):
        shutil.rmtree(TYPE_OF_OUTPUTS[2])
    if not os.path.exists(TYPE_OF_OUTPUTS[2]):
        os.makedirs(TYPE_OF_OUTPUTS[2])
    
    # run all baselines (BM25, TF-IDF, Smoothed Query Likelihood Model)
    # with stemmed  corpus
    for i in range(1,4):
        RetrievalModels.selectRetrievalModel(INVERTED_INDEX[2],NUM_OF_TOKEN_PER_DOC[2],i,TYPE_OF_OUTPUTS[2],qMap)
        

def snippetGeneration():
    # fetch queryMap
    baselineRunsNoTextTrans()
    qMap=RetrievalModels.fetchQueryMap()
    docScorePerQuery=RetrievalModels.selectRetrievalModel(INVERTED_INDEX[0],NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[0],qMap)
    DisplayResult.main(docScorePerQuery)

def evaluatePerformance():
    global DOC_SCORES_PER_QUERY_PER_RUN
    baselineRunsNoTextTrans()
    baselineRunsWithStopping()
    queryEnrichment()
    if os.path.exists(DOC_SCORES_PER_QUERY_PER_RUN_PICKLE):
        os.remove(DOC_SCORES_PER_QUERY_PER_RUN_PICKLE)
    if os.path.exists(DOC_SCORES_PER_QUERY_PER_RUN_PICKLE):
        DOC_SCORES_PER_QUERY_PER_RUN=PerformanceEvaluation.fetchDocScoresPerQueryPerRun()
    DOC_SCORES_PER_QUERY_PER_RUN['NoTextTran-Lucene']=PerformanceEvaluation.fetchLuceneFromDocScore()
    writeDocScoresToPickleFile(DOC_SCORES_PER_QUERY_PER_RUN)
    PerformanceEvaluation.main(DOC_SCORES_PER_QUERY_PER_RUN)

def induceNoise():
    global DOC_SCORES_PER_QUERY_PER_RUN
    baselineRunsNoTextTrans()
    qmap=RetrievalModels.fetchQueryMap()
    newQueries=SpellingErrorGenerator.main(qmap)
    if os.path.exists(TYPE_OF_OUTPUTS[5]):
        shutil.rmtree(TYPE_OF_OUTPUTS[5])
    if not os.path.exists(TYPE_OF_OUTPUTS[5]):
        os.makedirs(TYPE_OF_OUTPUTS[5])
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],\
                                                          RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[5],newQueries)
    
    if os.path.exists(DOC_SCORES_PER_QUERY_PER_RUN_PICKLE):
        DOC_SCORES_PER_QUERY_PER_RUN=PerformanceEvaluation.fetchDocScoresPerQueryPerRun()
    DOC_SCORES_PER_QUERY_PER_RUN['ErrorInducedQuery-BM25']=docScorePerQuery
    writeDocScoresToPickleFile(DOC_SCORES_PER_QUERY_PER_RUN)
    PerformanceEvaluation.main(DOC_SCORES_PER_QUERY_PER_RUN)

def softMatching():
    global DOC_SCORES_PER_QUERY_PER_RUN
    baselineRunsNoTextTrans()
    qmap=RetrievalModels.fetchQueryMap()
    newQueries=SoftMatchingQuerHandler.main(qmap)
    if os.path.exists(TYPE_OF_OUTPUTS[4]):
        shutil.rmtree(TYPE_OF_OUTPUTS[4])
    if not os.path.exists(TYPE_OF_OUTPUTS[4]):
        os.makedirs(TYPE_OF_OUTPUTS[4])
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],\
                                                          RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[4],newQueries)
    
    
    if os.path.exists(DOC_SCORES_PER_QUERY_PER_RUN_PICKLE):
        DOC_SCORES_PER_QUERY_PER_RUN=PerformanceEvaluation.fetchDocScoresPerQueryPerRun()
    DOC_SCORES_PER_QUERY_PER_RUN['SoftQueryMatching-BM25']=docScorePerQuery
    writeDocScoresToPickleFile(DOC_SCORES_PER_QUERY_PER_RUN)
    PerformanceEvaluation.main(DOC_SCORES_PER_QUERY_PER_RUN)
    
    
# create pickle files for DOC_SCORES_PER_QUERY_PER_RUN 
def writeDocScoresToPickleFile(map_):
    fName=DOC_SCORES_PER_QUERY_PER_RUN_PICKLE
    if os.path.exists(fName):
        os.remove(fName)
    fileIndex=open(fName,'wb')
    pickle.dump(map_,fileIndex)
    fileIndex.close()

def generateAllOutputs():
    baselineRunsWithStemming()
    evaluatePerformance()
    induceNoise()
    softMatching()
    snippetGeneration()
    
def merge(a, b):
    c = a.copy()
    c.update(b)
    return c   
    
def selectTasks():
    while True:
        print "\nSelect the task to be performed:"
        print "Enter 1 : To Perform BaseLine Runs(BM25, TF-IDF, Smoothed Query Likelihood) on CASM corpus without any Text Transformation(no stemming and no stopping)"
        print "Enter 2 : To Perform One BaseLine Run(Default BM25) on CASM corpus and perform query enrichment"
        print "Enter 3 : To Perform BaseLine Runs(BM25, TF-IDF, Smoothed Query Likelihood) on CASM corpus with Text Transformation(only stopping)"
        print "Enter 4 : To Perform BaseLine Runs(BM25, TF-IDF, Smoothed Query Likelihood) on CASM corpus with Text Transformation(only stemming)"
        print "Enter 5 : Display snippets and query highlighting for results of one of the BaseLine Run (Default BM25)"
        print "Enter 6 : Evaluate the performance of each run performed above in terms of effectiveness(except stemming run)"
        print "Enter 7 : Induce noise in the casm queries and perform one of the baseline run (Default BM25) and compare the overall effectiveness with the original baseline run (Default BM25)" 
        print "Enter 8 : Perform Soft matching query handling using BM25"
        print "Enter 9 : Perform all tasks and generate all outputs"
        print "Enter 10 : To Initialize the project i.e. delete all output files and folders "
        print "Enter 11 : To exit!!!!"
        options={1:baselineRunsNoTextTrans,
                2:queryEnrichment,
                3:baselineRunsWithStopping,
                4:baselineRunsWithStemming,
                5:snippetGeneration,
                6:evaluatePerformance,
                7:induceNoise,
                8:softMatching,
                9:generateAllOutputs,
                10:InitializeProject.initalize }
        print "Enter Your Choice >>> "
        x=input()
        if x==11:
            break
        else:
            options[x]()
        
if __name__=='__main__':
    print "Welcome to Your Own Search Engine Using CASM Corpus"
    selectTasks()    