from helper import remove_punc
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from hw8_1 import *
import numpy as np

#Clean and prepare the contents of a document
#Takes in a file name to read in and clean
#Return a single string, without stopwords, spaces, and punctuation
# NOTE: Do not append any directory names to doc -- assume we will give you
# a string representing a file name that will open correctly
def read_and_clean_doc(doc) :
    #1. Open document, read text into *single* string
    with open(doc) as f:
        f_read = f.read()
    #2. Filter out punctuation from list of words (use remove_punc)
    dwords = word_tokenize(f_read)
    words=[]
    for word in dwords:
        if remove_punc(word):
            words.append(remove_punc(word))
    
    #3. Make the words lower case
    words = [word.lower() for word in words]
    #4. Filter out stopwords
    stopwords = set(stopwords.words('english'))
    all_no_stop = []
    for word in words:
        if not word in stopwords:
            all_no_stop.append(word)
    #5. Remove remaining whitespace
    
    return all_no_stop
    
#Builds a doc-word matrix for a set of documents
#Takes in a *list of filenames* and a number *n* corresponding to the length of each ngram
#
#Returns 1) a doc-word matrix for the cleaned documents
#This should be a 2-dimensional numpy array, with one row per document and one 
#column per ngram (there should be as many columns as unique words that appear
#across *all* documents. Also, Before constructing the doc-word matrix, 
#you should sort the list of ngrams output and construct the doc-word matrix based on the sorted list
#
#Also returns 2) a list of ngrams that should correspond to the columns in
#docword
def build_doc_word_matrix(doclist, n) :
    #1. Create the cleaned string for each doc (use read_and_clean_doc)
    ng = [get_ngrams( k, n) for k in [read_and_clean_doc( j) for j in doclist]]

    ngsorted = set()

    [ngsorted.update( i) for i in ng] 

    ngramlist = sorted(list( ngsorted))
    docword = np.zeros(( len( doclist), len( ngramlist)))

    for g in range(len(doclist)):
        dcs = [read_and_clean_doc(f) for f in doclist][ g]
        
        ng = []

        [ng.append(dcs[st : st + n]) for st in range(len(dcs) - n + 1)]

        for e in ng:
            docword[g, ngramlist.index(e)] = docword[g, ngramlist.index(e)] + 1
            
    return docword, ngramlist
    
#Builds a term-frequency matrix
#Takes in a doc word matrix (as built in build_doc_word_matrix)
#Returns a term-frequency matrix, which should be a 2-dimensional numpy array
#with the same shape as docword
def build_tf_matrix(docword) :


    temp = np.sum(docword, axis = 1)
    idf = (docword / (temp[: , np.newaxis ]))

    return idf
    
#Builds an inverse document frequency matrix
#Takes in a doc word matrix (as built in build_doc_word_matrix)
#Returns an inverse document frequency matrix (should be a 1xW numpy array where
#W is the number of ngrams in the doc word matrix)
#Don't forget the log factor!
def build_idf_matrix(docword) :

    idf = (docword.shape[ 0 ] / (np.sum(docword > 0, axis = 0).reshape( 1, -1)))
    
    idf = np.log10(idf)
    return idf
    
#Builds a tf-idf matrix given a doc word matrix
def build_tfidf_matrix(docword) :
    
    tfidf = (build_tf_matrix( docword ) * build_idf_matrix( docword ))

    return tfidf
    
#Find the three most distinctive ngrams, according to TFIDF, in each document
#Input: a docword matrix, a wordlist (corresponding to columns) and a doclist 
# (corresponding to rows)
#Output: a dictionary, mapping each document name from doclist to an (ordered
# list of the three most unique ngrams in each document
def find_distinctive_ngrams(docword, ngramlist, doclist) :
    distinctive_words = {}

    for k in range(len(doclist)):
        t = build_tfidf_matrix(docword)

        temp = np.argsort( -t[k, :])[ : 3]

        ar = list(np.array( ngramlist)[ temp])

        distinctive_words[ doclist[ k]] = ar
    return distinctive_words


if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join, splitext
    #Some main code:
    '''directory='lecs'
    path = join(directory, '1_vidText.txt')
    read_and_clean_doc(path)
    #build document list
    
    path='lecs'
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    path_list = [join(path, f) for f in file_list]
    
    mat,wlist=build_doc_word_matrix(path_list)
    
    tfmat = build_tf_matrix(mat)
    idfmat = build_idf_matrix(mat)
    tfidf = build_tfidf_matrix(mat)
    results = find_distinctive_words(mat,wlist,file_list)'''
    
    ### Test Cases ###
    directory='lecs'
    path1 = join(directory, '1_vidText.txt')
    path2 = join(directory, '2_vidText.txt')
    
    
    print("*** Testing read_and_clean_doc ***")
    print(read_and_clean_doc(path1)[0:20])
    print("*** Testing build_doc_word_matrix ***") 
    doclist =[path1, path2]
    docword, wordlist = build_doc_word_matrix(doclist, 3)
    print(docword.shape)
    print(len(wordlist))
    print(docword[0][0:10])
    print(wordlist[0:10])
    print(docword[1][0:10])
    print("*** Testing build_tf_matrix ***") 
    tf = build_tf_matrix(docword)
    print(tf[0][0:10])
    print(tf[1][0:10])
    print(tf.sum(axis =1))
    print("*** Testing build_idf_matrix ***") 
    idf = build_idf_matrix(docword)
    print(idf[0][0:10])
    print("*** Testing build_tfidf_matrix ***") 
    tfidf = build_tfidf_matrix(docword)
    print(tfidf.shape)
    print(tfidf[0][0:10])
    print(tfidf[1][0:10])
    print("*** Testing find_distinctive_words ***")
    print(find_distinctive_ngrams(docword, wordlist, doclist))
