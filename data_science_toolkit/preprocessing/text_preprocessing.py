'''
typical steps 
Steps:
	Lemmatizing 
	Remove punctuation 
	Unify (e.g. lowercase) 
	Remove stop words 
    stemming 

Lemmatize first so that you can reduce dimensionality first 

Stemming is more computationally effective 

Lemmatize for smaller corpuses because computationally intensive 

'''


#TODO look into gensim 

import pandas as pd
import spacy

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

import re


lemmatizer = WordNetLemmatizer()

sp = spacy.load('en_core_web_sm')


def tokenize_corpus(corpus):
    '''
    breaks the words into tokens for each document in the corpus
    '''
    tokenized_corpus = [nltk.word_tokenize(doc) for doc in corpus]
    return tokenized_corpus

def lemmatize_corpus(corpus): 
    # lemmatizer is more computationally intensive but essentially finds a 'root' synonym for each word 
    lemmatized_corpus = []
    for doc in corpus:
        
        lemmatized_doc = [lemmatizer.lemmatize(word) for word in doc]
        lemmatized_corpus.append(" ".join(lemmatized_doc))
    
    return lemmatized_corpus

def spacy_lemmatize_corpus(corpus):
    '''
        lemmatizes based on the part of speech of the word
        .pos_ to access part of speech
        .lemma_ to access the lemmatized word
    '''
    lemmatized_corpus = []
    for doc in corpus:
        sp_text = sp(doc)
        lemmatized_doc = [word.lemma_ for word in sp_text]
        lemmatized_corpus.append(" ".join(lemmatized_doc))

    return lemmatized_corpus

def remove_non_words_corpus(corpus): 
    return [' '.join(re.split(r'\W+', doc)).strip() for doc in corpus]

def lowercase_corpus_no_abbr(corpus):
    processed_corpus = []
    for doc in corpus:
        doc_arr = []
        for word in doc.split(' '):
            if re.match('([A-Z]+[a-z]*){2,}', word):
                doc_arr.append(word)
            else:
                doc_arr.append(word.lower())
        processed_corpus.append(' '.join(doc_arr))
    return processed_corpus

def remove_stop_words_corpus(corpus):
    stop_words = set(stopwords.words('english'))
    processed_corpus = []
    for doc in corpus:
        words = doc.split()
        lowered = [w for w in words if w not in stop_words]
        processed_corpus.append(' '.join(lowered))
    return processed_corpus

def stem_corpus(corpus, stem_type='porter', language=None):
    stemmer = None
    if stem_type == 'lancaster':
        stemmer = LancasterStemmer()
    elif stem_type == 'porter':
        stemmer = PorterStemmer()
    elif language is not None:
        stemmer = SnowballStemmer(language)
    
    processed_corpus = []
    for doc in corpus:
        words = doc.split()
        stemmed = [stemmer.stem(w) for w in words]
        processed_corpus.append(' '.join(stemmed))
    return processed_corpus

def get_token_counts(corpus):
    '''
    gets word counts in order to build a document vector matrix
    '''
    vocab, index = {}, 1  # start indexing from 1
    for doc in corpus:
        doc = doc.split(' ')
        for token in doc:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

def get_document_vectors(corpus, vocab):
    ''' 
    for a list of tokens count the occurence of each vocab word 
    into a list 
    used to build document vector matrix 
    '''
    vectors=[]
    for doc in corpus:
        vector=[]
        for w in vocab:
            vector.append(doc.count(w))
        vectors.append(vector)
    print(len(vectors))
    return vectors

    
def build_document_vector_matrix(corpus):
    document_vectors = []
    vocab_dic = None
    # need to split doc into list first 
    vocab_dic = get_token_counts(corpus)
    print(vocab_dic)
    document_vectors = get_document_vectors(corpus, vocab_dic)
    print(len(document_vectors))
    document_vector_matrix = pd.DataFrame(document_vectors)
    print(document_vector_matrix)
    document_vector_matrix.columns = vocab_dic.keys()
    document_vector_matrix=document_vector_matrix

    return document_vector_matrix

def text_pipeline(corpus):
    print(corpus)
    lemmatized_corpus = spacy_lemmatize_corpus(corpus)
    print(lemmatized_corpus)
    no_punctuation_corpus = remove_non_words_corpus(lemmatized_corpus)
    print(no_punctuation_corpus)
    lowercase_corpus = lowercase_corpus_no_abbr(no_punctuation_corpus)
    print(lowercase_corpus)
    no_stop_words = remove_stop_words_corpus(lowercase_corpus)
    print(no_stop_words)
    stemmed_corpus = stem_corpus(no_stop_words, stem_type='porter')
    print(stemmed_corpus)
    dv_matrix = build_document_vector_matrix(stemmed_corpus)
    print(dv_matrix)    


#TODO: retain dictionary from lemmatization and create a list of all words 
# belonging to the root in dictionary form where the key is the root word