import os
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
DELETEFILES=['../SpellErrorGenerator/SpellingErrorInducedQueries.txt',
             '../QueryEnhancement/EnrichedCasmQueries.txt',
             '../Utility/DOC_SCORES_PER_QUERY_PER_RUN.pickle']
DELETEFOLDERS=['../CorpusGeneration/TokenizedCorpus',
               '../CorpusGeneration/TokenizedCorpusWithStopping',
               '../CorpusGeneration/TokenizedCorpusWithStemming',
               '../Indexing/IndexPickleFiles',
               '../Indexing/IndexTextFiles',
               '../Retrieval/No Text Transformation Runs Output',
                 '../Retrieval/Stopped Baseline Runs Output',
                 '../Retrieval/Stemmed Baseline Runs Output',
                 '../Retrieval/Query Enrichment Runs Output',
                 '../Retrieval/Soft Matched Query Runs Output',
                 '../Retrieval/Error Induced Query Runs Output',
                 '../Evaluation/OutputFiles',
                 '../Display/Retrieved_Docments_with_snippets']
def initalize():
    for folders in DELETEFOLDERS:
        if os.path.exists(folders):
            shutil.rmtree(folders)
    for files in DELETEFILES:
        if os.path.exists(files):
            os.remove(files)
            
initalize()            
