import math
import operator
import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Retrieval import RetrievalModels


TYPE_OF_OUTPUTS=['../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output']


#variable storing the path of the folder which contains the corpus with all the documents tokenized and processed
TOKENIZED_CORPUS_PATH="../CorpusGeneration/TokenizedCorpus"
#output filename for storing the enriched version of the queries
ENRICHED_QUERY_FILE_NAME='../QueryEnhancement/EnrichedCasmQueries.txt'

def performQueryEnrichment(docScore,current_query,inverted_index,query_id):
    newQuery=current_query
    
    relevance_set={}
    non_relevance_set={}
    initial_term_weights={}
    expanded_query={}    
    
    for k,v in inverted_index.items():
        initial_term_weights[k]=0
        relevance_set[k]=0
        non_relevance_set[k]=0
        
    document_score=sorted(docScore.items(),key=operator.itemgetter(1),reverse=True)
    count=0
    
    '''Generating relevance set'''

    for doc in document_score:
        count+=1
        doc_id=doc[0]
        corpus_path=os.path.join(TOKENIZED_CORPUS_PATH,doc_id+".txt")
        file_handle=open(corpus_path,"r")
        content=file_handle.readline()
        term_list=content.split()
        for term in term_list:
            relevance_set[term]+=1
        if(count==10):
            break
        
    
    '''Generating normalizer for relevence vector'''
    
    relevance_normalizer=0
    for k,v in relevance_set.items():
        relevance_normalizer+= float(v**2)
    relevance_normalizer=float(math.sqrt(relevance_normalizer))
        
            
        
     
    '''Generating non-relevance set'''   
    

    for i in range(10,len(document_score)):
        doc_id=document_score[i][0]
        corpus_path=os.path.join(TOKENIZED_CORPUS_PATH,doc_id+".txt")
        file_handle=open(corpus_path,"r")
        content=file_handle.readlines()
        for line in content:
            term_list=line.split()
            for term in term_list:
                non_relevance_set[term]+=1
                
                
                
       
       
    '''Generating normalizer for non_relevence vector'''
             
    non_relevance_normalizer=0
    
    for k,v in non_relevance_set.items():
        non_relevance_normalizer+= float(v**2)
    non_relevance_normalizer=float(math.sqrt(non_relevance_normalizer))
                
                
    '''Generating initial query term weights'''
                
    query_list=current_query.split()
    for term in query_list:
        if(term in initial_term_weights.keys()):
            initial_term_weights[term]+=1
        else:
            initial_term_weights[term]=1
            
        
        
    '''Generating expanded query'''
        
    for term in inverted_index.keys():
        initial_term_weight= 0.2*initial_term_weights[term]
        relevance_weightage= (0.75/relevance_normalizer)*relevance_set[term] 
        non_relevance_weightage=(0.05/non_relevance_normalizer)*non_relevance_set[term]   
        
        expanded_query[term]=initial_term_weight+relevance_weightage-non_relevance_weightage
        
    
    sorted_expanded_query_terms=sorted(expanded_query.items(),key=operator.itemgetter(1),reverse=True)
    

    '''Inserting the top 20 query words into the original query'''

    for i in range(0,20):
        t=sorted_expanded_query_terms[i]
        
        if(t[0] not in newQuery):
            newQuery+=" "+t[0]
            
    return newQuery


'''prints the expanded queries into a file called 'EnrichedCasmQueries.txt' '''

def writeNewQueryTofile(ENRICHED_QUERY_FILE_NAME,newQuery,query_id):
    
    
    fileIndex=open(ENRICHED_QUERY_FILE_NAME,'a')
    fileIndex.write("Enriched query for QueryID:"+str(query_id)+"\n")
    fileIndex.write("------------------------------------------\n\n")
    fileIndex.write(newQuery+"\n\n\n")
    
def main(docScorePerQuery):
    
    if os.path.exists(TYPE_OF_OUTPUTS[3]):
        shutil.rmtree(TYPE_OF_OUTPUTS[3])
    if not os.path.exists(TYPE_OF_OUTPUTS[3]):
        os.makedirs(TYPE_OF_OUTPUTS[3])
    qMap=RetrievalModels.fetchQueryMap()
    invertedIndex=RetrievalModels.fetchInvertedIndex(RetrievalModels.INVERTED_INDEX[0])
    newQMap={}
    for queryID in docScorePerQuery:
        newQuery=performQueryEnrichment(docScorePerQuery[queryID],qMap[queryID],invertedIndex,queryID)   
        writeNewQueryTofile(ENRICHED_QUERY_FILE_NAME,newQuery,queryID)
        newQMap[queryID]=newQuery
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],\
                                                          RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[3],newQMap)
    return docScorePerQuery
    
if __name__=='__main__':
    # fetch queryMap
    qMap=RetrievalModels.fetchQueryMap()
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],\
                                                          1,TYPE_OF_OUTPUTS[0],qMap)
    main(docScorePerQuery)