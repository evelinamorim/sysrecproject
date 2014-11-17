import sys
import os
import json
import math
import nltk
import string
from gensim import corpora
from sklearn.feature_extraction.text import CountVectorizer
#from gensim import corpora,models
import numpy as np

stopwords = nltk.corpus.stopwords.words('english')
stemmer = nltk.stem.porter.PorterStemmer()

class MyDocument:
    def __init__(self,max_id):
	#merchant name
	self.merchant = []
	#title of document
	self.title = []
	#highlight of deal
	self.highlight = []
	#description of the deal
	self.description = []
	#bag of words
	self.bow = []
	#the concatenation of all other features
	self.concatenation = []
	self.max_id = max_id
        self.tokens = []
	#dicionario que mapeia numeros -> palavras
	#self.dict_words = {}

    def map_words_concatenation(self,lst_word,global_dict):
	#map words to numbers
	#max_id = len(dict_words.keys())
	lst_numbers = []
	tf = {}#palavras desta "rodada"

	for w in lst_word:
	    n = -1
	    if (w not in global_dict):
		global_dict[w] = [self.max_id]
		tf[w] = True
		n = self.max_id
		self.max_id = self.max_id + 1
	    else:
		#esta no dicionario global, mas nao esta no local
		if (w not in tf):
		    global_dict[w].append(self.max_id)
		    tf[w] = True
		    n = self.max_id
		    self.max_id = self.max_id + 1
		else:
		    #o ultimo elemento a ser acrescentado no dicionario global
		    #eh considerado o id da palavra
		    n = global_dict[w][-1]
	    lst_numbers.append(n)


	return lst_numbers

    def map_words(self,lst_word,global_dict):
	#map words to numbers
	#max_id = len(dict_words.keys())
	lst_numbers = []
	new_lst_words = []
	for w in lst_word:
	    result = ''.join([i for i in w if not i.isdigit()])
	    new_lst_words.append(result)

	for w in new_lst_words:
	    if (w not in global_dict):
		global_dict[w]= [self.max_id]
		self.max_id = self.max_id + 1
	    lst_numbers.append(global_dict[w][0])


	return lst_numbers

    def read_json(self,data,global_dict,global_dict_num):
	#usa separadamente cada um para teste

	if ("merchant" in data):
	    lst_merchant = nltk.tokenize.word_tokenize(data["merchant"].lower())
	    word_list = [stemmer.stem(w) for w in lst_merchant if w.lower() not in stopwords and w not in string.punctuation]
	    #self.merchant = self.map_words(word_list,global_dict)
	    self.merchant = word_list
	    self.concatenation += self.map_words_concatenation(word_list,global_dict)
	    self.bow += self.merchant
            self.tokens += word_list

	if ("title" in data):
	    lst_title = nltk.tokenize.word_tokenize(data["title"].lower())
	    word_list = [stemmer.stem(w) for w in lst_title if w.lower() not in stopwords and w not in string.punctuation]
	    #self.title = self.map_words(word_list,global_dict)
	    self.title = word_list
	    self.concatenation += self.map_words_concatenation(word_list,global_dict)
	    self.bow += self.title
            self.tokens += word_list

	if ("highlight" in data):
	    lst_highlight = nltk.tokenize.word_tokenize(data["highlight"].lower())
	    word_list = [stemmer.stem(w) for w in lst_highlight if w.lower() not in stopwords and w not in string.punctuation]
	    #self.highlight = self.map_words(word_list,global_dict)
	    self.highlight = word_list
	    self.concatenation += self.map_words_concatenation(word_list,global_dict)
	    self.bow += self.highlight
            self.tokens += word_list

	if ("description" in data):
	    lst_description = nltk.tokenize.word_tokenize(data["description"].lower())
	    word_list = [stemmer.stem(w) for w in lst_description if w.lower() not in stopwords and w not in string.punctuation]
	    #self.description = self.map_words(word_list,global_dict)
	    self.description = word_list
	    self.concatenation = self.map_words_concatenation(word_list,global_dict) 
	    self.bow += self.description
            self.tokens += word_list


class Features:
    def __init__(self,ftr_list):
	self.avg = {}
	self.aiff_vector = {}
	self.iff = {}
	for f in ftr_list:
	    self.avg[f] = 0.0
	    self.iff[f] = {}

    def count_term(self,s,ts,dictionary):
	 tf = {}
	 tok2id = dictionary.token2id
	 word_list = [] 
	 for w in s:
	     try:
	         word_list.append(tok2id[w])
	     except KeyError:
		 pass
		 #print "*** Warning *** word ",w," does not exist"
           
	 for tok in word_list:
	     #print "-->",t,t in ts
	     if (tok in ts):
                 #primeira vez nesta feature
		 #if (t not in tf):
		  ts[tok] = ts[tok] + 1
		  tf[tok] = True
	     else:
		 #comecando a contagem para este termo
		 ts[tok] = 1
		 tf[tok] = True

    def term_spread(self,mydoc,dictionary):
	#dada umadeal
        ts = {}

	#percorrendo cada feature
	if (mydoc.title != []):
	    self.count_term(mydoc.title,ts,dictionary)
	    #print "Title: ",mydoc.title

	if (mydoc.merchant != []):
	    self.count_term(mydoc.merchant,ts,dictionary)
	    #print "Merchant: ",mydoc.merchant

	if (mydoc.highlight != []):
	    self.count_term(mydoc.highlight,ts,dictionary)

	if (mydoc.bow != []):
	    self.count_term(mydoc.bow,ts,dictionary)
	    #print "BoW: ",mydoc.bow

	if (mydoc.description != []):
	    self.count_term(mydoc.description,ts,dictionary)
	    #print "Description: ",mydoc.description

	#if (mydoc.concatenation != []):
	#    self.count_term(mydoc.concatenation,ts)
	#import sys
	#print ts
	#sys.exit()
        return ts.items()

    def feature_spread(self,mydoc,ts):
	fs = {}

	#percorrendo cada feature
	if (self.title != []):
	    fs_title = 0.0
	    for w in self.title:
		fs_title += ts[w]
	    fs["title"] = (fs_title/self.title.size())

	if (self.merchant != []):
	    fs_merchant = 0.0
	    for w in self.merchant:
		fs_merchant += ts[w]
	    fs["merchant"] = (fs_merchant/self.merchant.size())

	if (self.highlight != []):
	    fs_highlight = 0.0
	    for w in self.highlight:
		fs_highlight += ts[w]
	    fs["highlight"] = (fs_highlight/self.highlight.size())

	if (self.bow != []):
	    fs_bow = 0.0
	    for w in self.bow:
		fs_bow += ts[w]
	    fs["bow"] = (fs_bow/self.bow.size())

	if (self.description != []):
	    fs_description = 0.0
	    for w in self.description:
		fs_description += ts[w]
	    fs["description"] = (fs_description/self.description.size())

	if (self.concatenation != []):
	    fs_concatenation = 0.0
	    for w in self.concatenation:
		fs_concatenation += ts[w]
	    fs["concatenation"] = (fs_concatenation/self.concatenation.size())
	return fs

    def average_feature_spread(self,deals):

	#deals deve ser uma lista de mydocs
	for d in deals:
	    ts = self.term_spread(d)
	    fs = self.feature_spread(d,ts)
	    for k in self.avg:
		self.avg[k] += fs[k]
	for k in self.avg:
	    self.avg[k] = self.avg[k]/deals.size()

    def freq_term_feature(self,mydoc,freq):
	#dada uma deal, contar a frequencia de um termo por 
	# feature
	fq = {}

	#percorrendo cada feature
	if (self.title != []):
	    fq_title = {}
	    for w in self.title:
		if w not in fq_title:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["title"]):
		         freq["title"][w] = freq["title"][w] + 1
		    else:
		         freq["title"][w] =  1
	            fq_title[w] = True

	if (self.merchant != []):
	    fq_merchant = {}
	    for w in self.merchant:
		if w not in fq_merchant:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["merchant"]):
		         freq["merchant"][w] = freq["merchant"][w] + 1
		    else:
		         freq["merchant"][w] =  1
	            fq_merchant[w] = True

	if (self.highlight != []):
	    fq_highlight = {}
	    for w in self.highlight:
		if w not in fq_highlight:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["highlight"]):
		         freq["highlight"][w] = freq["highlight"][w] + 1
		    else:
		         freq["highlight"][w] =  1
	            fq_highlight[w] = True

	if (self.bow != []):
	    fq_bow = {}
	    for w in self.bow:
		if w not in fq_bow:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["bow"]):
		         freq["bow"][w] = freq["bow"][w] + 1
		    else:
		         freq["bow"][w] =  1
	            fq_bow[w] = True

	if (self.description != []):
	    fq_description = {}
	    for w in self.description:
		if w not in fq_description:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["description"]):
		         freq["description"][w] = freq["description"][w] + 1
		    else:
		         freq["description"][w] =  1
	            fq_bow[w] = True

	if (self.concatenation != []):
	    fq_concatenation = {}
	    for w in self.concatenation:
		if w not in fq_concatenation:
		    #se o termo corrente ainda nao foi computada para esta deal,
		    #acrescentar
		    if (w in freq["concatenation"]):
		         freq["concatenation"][w] = freq["concatenation"][w] + 1
		    else:
		         freq["concatenation"][w] =  1
	            fq_bow[w] = True

    def inverse_feature_frequency(self,deals):
	#inverse feature frequency
	freq = {}
	for k in self.agv:
	    freq[k] = {}
	
	for d in deals:

	    #para cada termo, computar a freq(t,s), que quer dizer o numero de 
	    #deals que t aparece na feature s
	    self.freq_term_feature(d,freq)

        ndeals = float(deals.size())
	for s in freq:
	     for t in freq[s]:
		 self.iff[s][t] = math.log(ndeals/freq[s][t])

    def aiff(self,deals):

	self.inverse_feature_frequency(deals)

        aiff_vector = {}
	for k in self.agv:
	    aiff_vector[k] = 0.0
	    for t in self.iff[k]:
		aiff_vector[k] += self.iff[k][t]
	    aiff_vector[k] = aiff_vector[k]/self.iff[k].size()
	    

class MyCorpus:

    def __init__(self,ftr_list):
	self.featuresObj = Features(ftr_list)
	#o maior id de uma palavra do vocabulario no momento
	self.max_id = 0

    def read(self,file_name,global_dict,global_dict_num):

	fd = open(file_name)
	data = json.load(fd)

	doc = MyDocument(self.max_id)
	self.max_id = doc.max_id
	doc.read_json(data,global_dict,global_dict_num)
	fd.close()
	return doc

    def ts_deal(self,deal,dictionary):
	return self.featuresObj.term_spread(deal,dictionary)

    def load_corpus(self,file_data):
	pass
    def process(self,dir_data,dir_out=""):

        #percorrer diretorio de deals
        deals = {}
        global_dict = {}
        global_dict_num = {}
        concat_str = lambda x,y: x + " " + y
        for subdir, dirs, files in os.walk(dir_data):
	    for f in files:
	        fileName, fileExtension = os.path.splitext(f)
	        #print "Processing features ",fileName,"..."
                d = self.read(os.path.join(subdir,f),global_dict,global_dict_num)

	        #guardar cada vetor de features de d, que neste 
	        #caso eh o term spread

		deals[int(fileName)] = (d,reduce(concat_str,d.tokens))

        vectorizer = CountVectorizer(min_df=1)
        tok2str_list = []
	token_lists = []
        for doc in deals:
	     token_lists.append(deals[doc][0].tokens)
             tok2str_list.append(deals[doc][1])

	dictionary = corpora.Dictionary(token_lists)
	dictionary.filter_extremes(no_below=1,no_above=0.2)
	corpus = []

        docids = []
	for iddoc in deals:
	    (doc,tok2str) = deals[iddoc]
	    docids.append(iddoc)
	    corpus.append(self.ts_deal(doc,dictionary))

        X = vectorizer.fit_transform(tok2str_list).todense()
        #docids = deals.keys()
        #docids.sort()

	return (corpus,docids,dictionary)

if __name__ == "__main__":
    dir_json = sys.argv[1]
    #ftr_list = ["highlight","title","concatenation","merchant","bow","description"]
    ftr_list = ["title","concatenation","merchant","bow","description"]
    mycorpusObj = MyCorpus(ftr_list)
    dir_out = "../daily-deals-data/ls-txt-ftrs/"
    mycorpusObj.process(dir_json,dir_out)


