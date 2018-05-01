import glob
import os
import shutil
import re
import string
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bs4 import BeautifulSoup
from math import ceil

DIR_RAW_HTML='../CASM-Files/Corpus'
DIR_CORPUS='../CorpusGeneration/TokenizedCorpus'
DIR_CORPUS_WITH_STOPPING='../CorpusGeneration/TokenizedCorpusWithStopping'
STOPPED_WORDS_FILE='../CASM-Files/common_words.txt'
DIR_CORPUS_WITH_STEMMING='../CorpusGeneration/TokenizedCorpusWithStemming'
STEMMED_QUERIES='../CASM-Files/cacm_stem.txt'


# method to convert HTML content into plain text
# Given: HTML content
# Returns: plain text
def parseHTML(htmlContent):
    soup = BeautifulSoup(htmlContent, 'html.parser')
    soup.prettify('utf-8', 'html')
    text=soup.find('pre').get_text().encode('utf-8')
    matchObj=re.search(r'[ \t\n\r\f\v]AM|[ \t\n\r\f\v]PM',text,re.M|re.I)
    if matchObj:
        return text[:matchObj.end()]
    return text


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
        
    return newList
    
    
# method to generate tokens from plain text
# Given: plain text
# Return: list of tokens     
def generateTokens(plainText):
    return list(filter(re.compile('[a-zA-Z0-9_]').search,plainText.split()))


def parser(fileName):
    f = open(fileName, 'r')
    docName = fileName.split('\\')[1].split('.')[0]+".txt"
    content = f.read()
    f.close()
    plainText=parseHTML(content)
    caseFoldedPlainText=caseFold(plainText)
    tokens=generateTokens(caseFoldedPlainText)
    tokens=removePunctuation(tokens)
    
    return (tokens,docName)

def writeTokenizedFiles(joinTokens,docName,corpusDirectory):
    try:    
        if not os.path.exists(corpusDirectory):
            os.makedirs(corpusDirectory)
        if os.path.exists(corpusDirectory+docName):
            print 'same name file exists' ,corpusDirectory+'/'+docName 
        tokenFile=open(corpusDirectory+'/'+docName,'w')
        tokenFile.write(joinTokens)
        tokenFile.close()
    except Exception:
        print Exception
    
def performStopping(tokens):
    f=open(STOPPED_WORDS_FILE,'r')
    stopList=f.read().split()
    newTokenList=[]
    for t in tokens:
        if t not in stopList:
            newTokenList.append(t)
    f.close()
    
    return  newTokenList   

def stemParser(corpusDirectory):
    f=open(STEMMED_QUERIES,'r')
    stemmedCorpus={}
    for line in f.readlines():
        if line[0]=='#':
            line=line.strip('\n')
            docName='CACM-'+'0'*(abs(len(line[2:])-4))+line[2:]
            stemmedCorpus[docName]=""
        else:
            line=line.strip('\n')
            stemmedCorpus[docName]+=line
    i=1
    for docName in stemmedCorpus:
        data=stemmedCorpus[docName]
        matchObj=re.search(r'[ \t\n\r\f\v]AM|[ \t\n\r\f\v]PM',data,re.M|re.I)
        if matchObj:
            data=data[:matchObj.end()]
        
        stemmedCorpus[docName]=data
        writeTokenizedFiles(data, docName+'.txt',corpusDirectory)
        if i%160==0:
            print("Parsing and Tokenization "+str(ceil(i/float(len(stemmedCorpus))*100))+'% complete')
        i+=1
       
    f.close()
            
            

def selectTypeOfTextTransformation1():
     
    while True:
        print "\nSelect transformation technique:"
        print "Enter 1 No Stopping No Stemming"
        print "Enter 2 With Stopping"
        print "Enter 3 With Stemming"
        print "Enter 4 to exit"
        x=input()
        if x==1:
            corpusDirectory=DIR_CORPUS
            mode="withoutStopping"
        elif x==2:
            corpusDirectory=DIR_CORPUS_WITH_STOPPING
            mode="withStopping"
        elif x==3:
            corpusDirectory=DIR_CORPUS_WITH_STEMMING
            mode="withStemming"
        else:
            break
             
        if os.path.exists(corpusDirectory):
            shutil.rmtree(corpusDirectory)
        if mode=='withStemming':
            stemParser(corpusDirectory)
        else: 
            path=os.path.join(DIR_RAW_HTML,r"*.html")
            rawFiles=glob.glob(path)
            i=1
            print "Starting to generate corpus....."
            print "Parsing and Tokenization 0% complete"
            for fileName in rawFiles:
                tokens,docName=parser(fileName)
                if mode=="withStopping":
                    tokens=performStopping(tokens)
                joinTokens=" ".join(tokens)
                writeTokenizedFiles(joinTokens, docName,corpusDirectory)
                if i%160==0:
                    print("Parsing and Tokenization "+str(ceil(i/float(len(rawFiles))*100))+'% complete')
                i+=1
 
                
def selectTypeOfTextTransformation(x):

    if x==1:
        corpusDirectory=DIR_CORPUS
        mode="withoutStopping"
    elif x==2:
        corpusDirectory=DIR_CORPUS_WITH_STOPPING
        mode="withStopping"
    elif x==3:
        corpusDirectory=DIR_CORPUS_WITH_STEMMING
        mode="withStemming"

    if os.path.exists(corpusDirectory):
        shutil.rmtree(corpusDirectory)
    if mode=='withStemming':
        stemParser(corpusDirectory)
    else: 
        path=os.path.join(DIR_RAW_HTML,r"*.html")
        rawFiles=glob.glob(path)
        i=1
        print "Starting to generate corpus....."
        print "Parsing and Tokenization 0% complete"
        for fileName in rawFiles:
            tokens,docName=parser(fileName)
            if mode=="withStopping":
                tokens=performStopping(tokens)
            joinTokens=" ".join(tokens)
            writeTokenizedFiles(joinTokens, docName,corpusDirectory)
            if i%160==0:
                print("Parsing and Tokenization "+str(ceil(i/float(len(rawFiles))*100))+'% complete')
            i+=1  
                          
if __name__ == '__main__':
    selectTypeOfTextTransformation1()