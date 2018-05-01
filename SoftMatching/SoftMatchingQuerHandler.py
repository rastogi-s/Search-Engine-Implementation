import math
import operator
import os
import re
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Evaluation import PerformanceEvaluation
from Retrieval import RetrievalModels
from SpellErrorGenerator import SpellingErrorGenerator


QUERIES_FILE_PATH="ListOfCACMQueries.txt"
QUERY_BIGRAM_INDEX={}
BIGRAM_INDEX={}
TERM_FREQUENCY_DICT={}
SOFT_MATCHING_OUTPUT_FILE="../SoftMatching/Softmatching_Output.txt"
DOC_SCORE_PER_QUERY_PER_RUN={}
DICTIONARY_DICT={}
TOKENIZED_CORPUS_PATH="../SoftMatching/TokenizedCorpus"
TYPE_OF_OUTPUTS=['../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output']

PATH_OF_DICTIONARY="../Utility/english_dictionary.txt"



def softmatching(query_id,query):
    new_query=""
    global INVERTED_INDEX
    global TERM_FREQUENCY_DICT
    global DICTIONARY_DICT

    query=query.split()
    corrected_previous_term=""
    for i in range(0,len(query)):

        term=query[i]
        
        if(term not in DICTIONARY_DICT[term[0]] and term not in TERM_FREQUENCY_DICT.keys()):
            corrected_term=suggest_terms(term,corrected_previous_term)
            corrected_previous_term=corrected_term
            new_query+=corrected_term+" "
        else:
            new_query+=term+" "
            corrected_previous_term=term

    return new_query
            
def create_bigram_index():
    corpus_path="../CorpusGeneration/TokenizedCorpus"
    global BIGRAM_INDEX
    for file_ in os.listdir(corpus_path):
        fh=open(corpus_path+"\\"+file_,"r")
        line=fh.readline()
        doc_id=file_[:-4]
        term_list=line.split()
        for i in range (0,len(term_list)-1):
            term=term_list[i]+" "+term_list[i+1]
            term=term.lower()
            if(BIGRAM_INDEX.has_key(term)):
                doc_count=BIGRAM_INDEX[term]
                if(doc_count.has_key(doc_id)):                   
                    doc_count[doc_id]+=1
                else:
                    doc_count[doc_id]=1                   
                BIGRAM_INDEX[term]=doc_count
                            
            else:                 
                BIGRAM_INDEX[term]={doc_id:1}


def suggest_terms(term,prev_term):
    global DICTIONARY_DICT
    global trigram_index
    global TERM_FREQUENCY_DICT
    global BIGRAM_INDEX
    
    possible_corrections={}
    word_count=len(TERM_FREQUENCY_DICT.keys())
    for word in DICTIONARY_DICT[term[0]]:
        if(len(word)==len(term)):
            if(word in TERM_FREQUENCY_DICT.keys()):
                dist=edit_distance(term,word)
                assumed_error_dist=math.ceil(0.6*len(word))
                if(dist<=assumed_error_dist):
                    possible_corrections[word]=0.2*(TERM_FREQUENCY_DICT[word]/float(word_count))
                    for entry in BIGRAM_INDEX.keys():
                        if(word in entry and prev_term and prev_term in entry):
                            co_occuring_probability=calculate_co_occuring_probability(entry)
                            possible_corrections[word]+=0.7*len(BIGRAM_INDEX[entry].keys())

                    for entry in QUERY_BIGRAM_INDEX.keys():
                        if(word in entry and prev_term and prev_term in entry):
                            co_occuring_probability=QUERY_BIGRAM_INDEX[entry]/float(len(QUERY_BIGRAM_INDEX.keys()))
                            
                            possible_corrections[word]+=0.1*co_occuring_probability
                    
            else:
                possible_corrections[word]=1/float(word_count)
                
    sorted_suggestions=sorted(possible_corrections.items(),key=operator.itemgetter(1),reverse=True)
    if(sorted_suggestions):

        return sorted_suggestions[0][0]
    else:
        return term     


def calculate_co_occuring_probability(entry):
    global BIGRAM_INDEX
    
    occurence=0
    for k in BIGRAM_INDEX[entry]:
        occurence+=BIGRAM_INDEX[entry][k]
    average_occurence=occurence/float(len(BIGRAM_INDEX.keys()))
    return average_occurence


def create_words_dict():
    global DICTIONARY_DICT
    
    fh=open(PATH_OF_DICTIONARY,"r")
    content=fh.readlines()
    alphabet="abcdefghijklmnopqrstuvwxyz"
    numbers="1234567890"

    for letter in alphabet:
        DICTIONARY_DICT[letter]=[]
     
    for number in numbers:
        DICTIONARY_DICT[number]=[]   
        
        
    for term in content:
        term=term[:-1]
        term=term.lower()
        if(re.match('^\w+$',term)):
            DICTIONARY_DICT[term[0]].append(term)
    DICTIONARY_DICT["i"].append("i")



def generate_term_frequency_dict(INVERTED_INDEX):
    global TERM_FREQUENCY_DICT
    
    for k,v in INVERTED_INDEX.items():
        term=k
        doc_dict=v
        term_count=0
        for k,v in doc_dict.items():
            term_count+=v
        term=term.lower()
        TERM_FREQUENCY_DICT[term]=term_count
        
    qMap=RetrievalModels.fetchQueryMap()       
    for queryId in qMap:
        line=qMap[queryId]
        for term in line.split():
            if(TERM_FREQUENCY_DICT.has_key(term)):
                TERM_FREQUENCY_DICT[term]+=1
            else:
                TERM_FREQUENCY_DICT[term]=1

def edit_distance(term1,term2):
    
    n=len(term1)
    m=len(term2)
    
    dp=[[0 for x in range(m+1)] for x in range(n+1)]
    
    for i in range(0,n+1):
        dp[i][0]=i
        
    for i in range(0,m+1):
        dp[0][i]=i
        
    for i in range(1,n+1):
        for j in range(1,m+1):
            if(term1[i-1]==term2[j-1]):
                cost=0
            else:
                cost=1
                
            dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+cost)
            
            if(i>1 and j>1 and term1[i-1]==term2[j-2] and term1[i-2]==term2[j-1]):
                dp[i][j]=min(dp[i][j],dp[i-2][j-2]+cost)
                
    return dp[n][m]


def create_query_bigram_index():
    global QUERY_BIGRAM_INDEX
              
    '''Including query in bigram index'''
    qMap=RetrievalModels.fetchQueryMap()       
    for queryId in qMap:
        line=qMap[queryId]
        line=line.split()
        for i in range(0,len(line)-1):
            term=line[i]+" "+line[i+1]
            if(QUERY_BIGRAM_INDEX.has_key(term)):
                QUERY_BIGRAM_INDEX[term]+=1
            else:
                QUERY_BIGRAM_INDEX[term]=1                     
    
def main(qmap):
    
    correctedQueries={}  
    create_words_dict()
    create_bigram_index()
    create_query_bigram_index()
    
    
        
    global INVERTED_INDEX
     
    print "Inducing errors in Queries:"
    error_queries=SpellingErrorGenerator.main(qmap)
    sorted_error_queries=sorted(error_queries.items(),key=operator.itemgetter(0))
    INVERTED_INDEX=RetrievalModels.fetchInvertedIndex(RetrievalModels.INVERTED_INDEX[0])      
    generate_term_frequency_dict(INVERTED_INDEX)
 
    print "Generating new queries:"  
    for t in sorted_error_queries:
        query_id=t[0]
        query=t[1]
        new_query=softmatching(query_id,query)
        print "Old Query -> "+str(query_id) +": "+query
        print "New Query -> "+str(query_id) +": "+new_query+"\n"
        correctedQueries[query_id]=new_query

    return correctedQueries   


if __name__=='__main__':
    
    qmap=RetrievalModels.fetchQueryMap()
    newQueries=main(qmap)
    if os.path.exists(TYPE_OF_OUTPUTS[4]):
        shutil.rmtree(TYPE_OF_OUTPUTS[4])
    if not os.path.exists(TYPE_OF_OUTPUTS[4]):
        os.makedirs(TYPE_OF_OUTPUTS[4])
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],\
                                                          RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],1,TYPE_OF_OUTPUTS[4],newQueries)
    docScoresPerQueryPerRun=PerformanceEvaluation.fetchDocScoresPerQueryPerRun()
    docScoresPerQueryPerRun['SoftQueryMatchingBLM-BM25']=docScorePerQuery

    PerformanceEvaluation.main(docScoresPerQueryPerRun)


