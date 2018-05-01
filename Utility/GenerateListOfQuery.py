import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Retrieval import RetrievalModels

def generateQmap():
    qmap=RetrievalModels.fetchQueryMap()
    fName='../Utility/ListOfCACMQueries.txt'
    if os.path.exists(fName):
        os.remove(fName)
    f=open(fName,'w')
    for qid in qmap:
        f.write(qmap[qid]+"\n")
    f.close()
    return qmap