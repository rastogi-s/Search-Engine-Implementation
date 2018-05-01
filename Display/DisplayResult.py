import operator
import os
import re
import shutil
import traceback
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Retrieval import RetrievalModels
from CorpusGeneration import GenerateTokenizedCorpus

TYPE_OF_OUTPUTS=['../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output']


'''Represents the path of the folder which will contain top 10 retrieved documents for all queries with snippets'''
 
SNIPPETS_FOLDER_PATH="../Display/Retrieved_Docments_with_snippets"

'''Represents the path of the folder containing the un-processed corpus'''

CORPUS_PATH="../CASM-Files/Corpus"

'''Represents the path of the folder containing all the STOPWORDS'''

STOPWORDS_FILE_LOCATION="../CASM-Files/common_words.txt"

'''A dictionary for storing all the stopwords'''

STOPWORDS={}


'''Stores all the stopwords from a file called common_words.txt and stores them into a global dictionary called STOPWORDS'''
def fetch_stopwords():
    global STOPWORDS
    path=os.path.join(os.getcwd(),STOPWORDS_FILE_LOCATION)
    f=open(path,"r")
    content=f.readlines()
    
    for line in content:
        STOPWORDS[line[:-1]]=True
        

def generate_snippets(query_id,query,docScore):
    DOCUMENT_SNIPPET_DICT={}
    query=str(query).lower()
    sorted_doc_score=sorted(docScore.items(),key=operator.itemgetter(1),reverse=True)
    c=0
    for t in sorted_doc_score:
        c+=1
        doc_id=t[0]
        
        
        '''Stores frequencies of every term in the index'''
        WORD_FREQUENCY_DICT={}
        
        '''Stores significance scores for every sentence in a document'''
        SENTENCE_SCORES={}
        
        
        '''fetching the text content of the document from the corpus'''
        current_doc_path=os.path.join(CORPUS_PATH,doc_id+".html")
        f=open(current_doc_path,"r")
        content=f.read()
        text_content=GenerateTokenizedCorpus.parseHTML(content)
        text_content=text_content.lower()         
          
          
        sentence_count=0
        
        
        '''Calculating the frequency of each term and counting the number of sentences'''
        
        for line in text_content.split("\n"):
            if(line != ""):
                sentence_count+=1
                for term in line.split():
                    term=RetrievalModels.removePunctuation(term)
                    if(WORD_FREQUENCY_DICT.has_key(term)):
                        WORD_FREQUENCY_DICT[term]+=1
                    else:
                        WORD_FREQUENCY_DICT[term]=1
                        
                    
                    
        '''Calculating the significance score for each sentence'''
                
        for line in text_content.split("\n"):
            if(line!=""):
                significant_word_count=0
                first_index=0
                last_index=0
                term_list=line.split()

                for i in range(0,len(term_list)):

                    term=RetrievalModels.removePunctuation(term_list[i])
                    if(check_significant_term(term,sentence_count,str(query),WORD_FREQUENCY_DICT)):
                        significant_word_count+=1
                        if(first_index==0):
                            first_index=i+1
                        last_index=i+1
            
                span_length=(last_index-first_index)+1
                
                SENTENCE_SCORES[line]=float(significant_word_count**2/span_length)
                
        sorted_sentence_score=sorted(SENTENCE_SCORES.items(),key=operator.itemgetter(1),reverse=True)
        DOCUMENT_SNIPPET_DICT[doc_id]=sorted_sentence_score     

        if(c==10):
            break

            
            
    '''Generate retrived results with snippets for the currnt query'''  
    genarate_snippet_files(sorted_doc_score,DOCUMENT_SNIPPET_DICT,query_id,query)
    
  
 
 
'''Stores the top 10 retrieved results with snippets and highlighting significant terms in a text file named as <query_id>.txt
inside a folder called 'Retrieved_Documents_with_snippets'''
 
def genarate_snippet_files(sorted_doc_score,DOCUMENT_SNIPPET_DICT,query_id,query):
    global STOPWORDS
    try:
        c=0
        file_name="Query"+str(query_id)+".txt"
        file_location=open(SNIPPETS_FOLDER_PATH+"\\"+file_name,"a")    
        
        for t in sorted_doc_score:
            doc_id=t[0]
            c+=1
            sorted_sentences=DOCUMENT_SNIPPET_DICT[doc_id]
            file_location.write(doc_id+"\n")
            file_location.write("----------------------------------------------------"+"\n")
            for i in range(0,len(sorted_sentences)):
                sentence=sorted_sentences[i][0]
                sentence=re.sub(r'[^a-zA-Z0-9]'," ",sentence)
                for term in sentence.split():

                    if(term in query and term not in STOPWORDS):
                        '''if the term is present in query then it is printed in uppercase to represent highlighting'''
                        file_location.write(term.upper()+" ")
                    else:
                        file_location.write(term+" ")
                file_location.write("\n")
                if(i==3):
                    break
                
            file_location.write("\n\n\n")
            if(c==10):
                break
        file_location.close()
    except Exception:
        print(traceback.format_exc())
        
'''Checks whether a particular is significant or not'''        
def check_significant_term(term,sentence_count,query,WORD_FREQUENCY_DICT):
    global STOPWORDS
    
    query_term_list=query.split()

    term_frequency=WORD_FREQUENCY_DICT[term]
    if(term not in STOPWORDS):

        if(sentence_count<25):
            if(term_frequency>=(7-0.1*(25-sentence_count))  or term in query_term_list):
                return True
        elif(sentence_count>=25 and sentence_count<=40):
            if(term_frequency>=7):
                return True
        elif(sentence_count>40):
            if(term_frequency>=(7+0.1*(sentence_count-40))):
                return True
    return False
            
    



def main(docScorePerQuery):
    
    '''Checks if the folder storing all the retrieved results with snippets exists, if exists delete and make a new one'''
    if(os.path.exists(SNIPPETS_FOLDER_PATH)):
        shutil.rmtree(SNIPPETS_FOLDER_PATH)
    os.mkdir(SNIPPETS_FOLDER_PATH)
    
    '''Fetching Stopwords''' 
    fetch_stopwords()
    
    # fetch QueryMap
    queryMap=RetrievalModels.fetchQueryMap()
    print("Storing the top 10 documents with snippets for all queries")
    for queryID in docScorePerQuery:
        generate_snippets(queryID,queryMap[queryID],docScorePerQuery[queryID])

if __name__=='__main__':
    # fetch queryMap
    qMap=RetrievalModels.fetchQueryMap()
    docScorePerQuery=RetrievalModels.selectRetrievalModel(RetrievalModels.INVERTED_INDEX[0],RetrievalModels.NUM_OF_TOKEN_PER_DOC[0],\
                                                          1,TYPE_OF_OUTPUTS[0],qMap)
    main(docScorePerQuery)

    
    


