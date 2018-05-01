import os
import pickle
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Retrieval import RetrievalModels

# name of relevant doc file:
RELEVANT_DOCS='../CASM-Files/cacm.rel.txt'
# ranks to calculate P@K
K1,K2=5,20
# name of pickle file to fetch doc scores per query per run
DOC_SCORES_PER_QUERY_PER_RUN='../Utility/DOC_SCORES_PER_QUERY_PER_RUN.pickle'
# method to generate MAP for each run
LUCENE_RUN_OUTPUT='../Retrieval/LuceneOutput/Top_100_Query_Result_Lucene.txt'
# Evaluation output directory
DIR_FOR_EVALUATION_OUTPUTS='../Evaluation/OutputFiles'


def generateMAP(tableQueryMap):
    return sum([tableQueryMap[queryID][1] for queryID in tableQueryMap])/float(len(tableQueryMap)) 

# method to generate MRR for each run
def generateMRR(tableQueryMap):
    return sum([tableQueryMap[queryID][2] for queryID in tableQueryMap])/float(len(tableQueryMap))


# method for generating Precision and Recall tables
def generatePrecisionRecallTables(docScores,relevantDocs,queryID):
    precRecallTable=[]
    sortedDocScore=sorted(docScores.iteritems(),key=lambda(k,v):(v,k),reverse=True)
    sortedDocScore=sortedDocScore[:100]
    numOfDocsCounter,relDocsCounter=0,0
    R=len(relevantDocs)
    relOrNonRel="N"
    sumPrecision=0
    flag=True
    RR=0
    for k in sortedDocScore:
        docId=k[0]
        numOfDocsCounter+=1
        if docId in relevantDocs:
            relDocsCounter+=1
            relOrNonRel="R"
        else:
            relOrNonRel="N"
        recall=relDocsCounter/float(R)
        precision=relDocsCounter/float(numOfDocsCounter)
        if relOrNonRel=='R':
            if flag:
                RR=1/float(numOfDocsCounter)
                flag=False
            sumPrecision+=precision
        precRecallTable+=[(queryID,docId,numOfDocsCounter,relOrNonRel,"{0:.2f}".format(recall),"{0:.6f}".format(precision))]
    AP=sumPrecision/float(R)    
        
    return (precRecallTable,AP,RR)


# method to write the Precision recall table to file
def writePrecisionRecallTablePerRunToFile(tableQueryMap,run,statistics):
    f=open(DIR_FOR_EVALUATION_OUTPUTS+'/EvalFileFor-'+run+".txt",'w')
    topic=" Effectiveness Evaluation for "+run+" "
    hashLen=90-len(topic)
    hashLen=hashLen/2
    filler="#"*hashLen+topic+"#"*hashLen
    if len(filler)<90:
        filler+="#"
    f.write("#"*90+"\n"+filler+"\n"+"#"*90+"\n\n")
    f.write('Evaluation Measures:\nMAP: '+str("{0:.2f}".format(statistics[0]))+'\nMRR: '+str("{0:.2f}".format(statistics[1]))+'\n')
    f.write('-'*80)
       
    for queryID in tableQueryMap:
        f.write('\n\nQueryID: '+str(queryID))
        f.write('\nP@K = '+str(K1)+': '+str(tableQueryMap[queryID][0][K1][5]))
        f.write('\nP@K = '+str(K2)+': '+str(tableQueryMap[queryID][0][K2][5]))
        f.write('\n\n\nPrecision & Recall Tables\n\n')
        f.write('QueryID'+' '*6+'DocId'+' '*10+'Rank'+' '*6+'R/N'+' '*6+'Recall'+' '*6+'Precision'+' '*6+'\n')
        f.write('-'*80+'\n')
        t=9         
        for qID,docId,numOfDocsCounter,relOrNonRel,recall,precision in tableQueryMap[queryID][0]:
            if numOfDocsCounter>9:
                t=8
            if numOfDocsCounter>99:
                t=7
            f.write(str(qID)+' '*9+docId+' '*9+str(numOfDocsCounter)+' '*t+\
                    relOrNonRel+' '*9+str(recall)+' '*9+str(precision)+'\n')
#             f.write(str(qID)+' '*(10-len(str(qID)))+docId+' '*(10-len(str(qID)))+str(numOfDocsCounter)+' '*(10-len(str(qID)))+\
#                     relOrNonRel+' '*(10-len(str(qID)))+str(recall)+' '*(10-len(str(qID)))+str(precision)+'\n')
    f.close()


def fetchDocScoresPerQueryPerRun():
    f=open(DOC_SCORES_PER_QUERY_PER_RUN)
    docScoresPerQueryPerRun=pickle.load(f)
    
    return docScoresPerQueryPerRun

def evaluate(docScoresPerQuery):
    tableQueryMap={}
    for queryID in docScoresPerQuery:
        #print "Generating precision recall table for queryID: "+str(queryID)
        relevantDocs=RetrievalModels.fetchRelevantDocIds(queryID)
        if len(relevantDocs)==0:
            continue
        tableQueryMap[queryID]=generatePrecisionRecallTables(docScoresPerQuery[queryID],relevantDocs,queryID)
    return tableQueryMap

def fetchLuceneFromDocScore():
    docScorePerQuery={}
    f=open(LUCENE_RUN_OUTPUT,'r')
    data=f.read()
    index=data.rfind('#')
    data=data[index+1:]
    listOfData=data.split('Query ')
    listOfData=listOfData[1:]
    newList=[]
    for text in listOfData:
        data="\n".join([data for data in text.splitlines() if data!=''])
        newList+=[data]
    print newList[0]
    for data in newList:
        splitData=data.splitlines()
        x=splitData[0]
        splitData=splitData[1:]
        docScore={}
        for lines in splitData:
            record=lines.split()
            if len(record)>0:
                docScore[record[2]]=record[4]
        docScorePerQuery[int(x)]=docScore
            
    f.close()
    return docScorePerQuery

def main(docScoresPerQueryPerRun):
    performanceMap={}
    print "Performance Evaluation!!!"
    if os.path.exists(DIR_FOR_EVALUATION_OUTPUTS):
        shutil.rmtree(DIR_FOR_EVALUATION_OUTPUTS)
    if not os.path.exists(DIR_FOR_EVALUATION_OUTPUTS):
        os.makedirs(DIR_FOR_EVALUATION_OUTPUTS)
    for run in docScoresPerQueryPerRun:
        print "Performance Evaluation for : "+run
        tableQueryMap=evaluate(docScoresPerQueryPerRun[run])
        MAP=generateMAP(tableQueryMap)
        MRR=generateMRR(tableQueryMap)
        statistics=(MAP,MRR)
        writePrecisionRecallTablePerRunToFile(tableQueryMap,run,statistics)
        performanceMap[run]=(tableQueryMap,statistics)
    return performanceMap

# def generateComparisonReport(performanceMap):
#     
#     if 'NoTextTran-BM25' in performanceMap and  \
#         'ErrorInducedQuery' in performanceMap and \
#         'SoftQueryMatchingBLM-BM25' in performanceMap:
#         
#         tempMap={'BM25 run without any query enhancement':performanceMap['NoTextTran-BM25'],
#                  'BM25 run with error induced queries':performanceMap['ErrorInducedQuery-BM25'],
#                  'BM25 run with soft matched queries':performanceMap['SoftQueryMatchingBLM-BM25']}
#         writeComparisonReportToCSV(tempMap,'Evaluation Comparision Report of BM25 runs -> Original Queries v/s Error Induced Queries v/s Soft Matched Queries')
# 
#     if 'NoTextTran-BM25' in performanceMap and  'ErrorInducedQuery' in performanceMap:
#         tempMap={'BM25 run without any query enhancement':performanceMap['NoTextTran-BM25'],
#                  'BM25 run with error induced queries':performanceMap['ErrorInducedQuery-BM25']}
#         #writeComparisonReportToCSV(tempMap,'Evaluation Comparision Report of BM25 runs -> Original Queries v/s Error Induced Queries')
# 
# 
# def writeComparisonReportToCSV(performanceMap,title):
#     name=" ".join([run for run in performanceMap])
#     f=open(name+".csv",'w')
#     
#     for run in performanceMap:
#         print ""
#         
#         
        
if __name__=='__main__':
    docScoresPerQueryPerRun=fetchDocScoresPerQueryPerRun()
    docScoresPerQueryPerRun['NoTextTran-Lucene']=fetchLuceneFromDocScore()
    performanceMap=main(docScoresPerQueryPerRun)
    #generateComparisonReport(performanceMap)
        