## machine learning to given objects
import mycorpus
import config
import util
import sys
import re
import os

import time 

from sklearn.externals import joblib
from sklearn import svm
from gensim import corpora, models, similarities
from gensim.models import ldamodel

open_quote = re.compile("\"[\w]+")
close_quote = re.compile("[\w\.\-]+\"")

def compara_datas(d1,d2):
    #dada duas datas como strings e cuja repsentacao eh: mm/dd/yyyy
    #retorna 1 se d1 eh maior, 0 se sao iguais e -1 se d1 eh menor
    d1_lst = d1.split("/")
    d2_lst = d2.split("/")

    #comparando ano
    if (int(d1_lst[2])>int(d2_lst[2])):
	return 1
    else:
	if (int(d1_lst[2])<int(d2_lst[2])):
	    return -1
	else:
	    #comparar mes
	    if (int(d1_lst[0])>int(d2_lst[0])):
		return 1
	    else:
		if  (int(d1_lst[0])<int(d2_lst[0])):
		    return -1
		else:
		    #comparar dia
		    if (int(d1_lst[1])>int(d2_lst[1])):
			return 1
		    else:
			if (int(d1_lst[1])<int(d2_lst[1])):
			    return -1
			else:
			    return 0


class machine_learning:

    def __init__(self,config_file,k=10):
	self.__config_obj = config.Config(config_file)
	self.__config_obj.read()

	self.lda_dir = self.__config_obj.get_value("lda_dir")[0]
	self.svr_dir = self.__config_obj.get_value("svr_dir")[0]
	self.data_name = self.__config_obj.get_value("data_name")[0]

	self.ftr_list = self.__config_obj.get_value("ftr_names")
	self.header = []
	self.target_ftr = -1
	self.target = {}
	#k eh o numero de treinos em temas
	self.k = k
	self.model_topic = None
	self.model_deal_size = None

    def pre_process_header(self):
	"""
	    * get only desired features
	    * apply types in values of features
	    *, etc
	"""
	new_values = []
	new_header_pos = []

	pos_key = -1
	nvalues = len(self.header)

	target_name = self.__config_obj.get_value("target_ftr")

	#quais as posicoes em header que serao efetivamente usadas
	for i in xrange(0,nvalues):
	    if (self.header[i] in self.ftr_list):
		new_header_pos.append(i)
	        if (self.header[i] == "started_at"):
                    #capturar a data para montar o historico
		    pos_key = len(new_header_pos)-1
            if (self.header[i]==target_name):
		self.target_ftr = i

	#print "1:",self.header
        #print "2:",new_header_pos,pos_key
        self.header[nvalues-1] =self.header[nvalues-1].replace('\n','')
        if (pos_key == -1):
	    print "*** WARNING *** Key feature value has not been identified"
        return (pos_key,new_header_pos)

    def process_values(self,header_pos,ftrs_value):
	"""
	pre process values of features w.r.t.:
	* types
	* whether it is discarded
	* target
	* key feature historical
	"""

        #print "3",ftrs_value
	new_ftrs_value = []
	for i in header_pos:
	    name = self.header[i]
	    type_ftr = self.__config_obj.get_value(name + "_type")[0]
	    new_ftrs_value.append(util.turn2type(ftrs_value[i],type_ftr))

        #capturando o header
        name = self.header[self.target_ftr]
	type_ftr = self.__config_obj.get_value(name + "_type")[0]
        target = util.turn2type(ftrs_value[self.target_ftr],type_ftr)

        #print "4",new_ftrs_value
        return new_ftrs_value,target

    def tokenize_features(self,line):
	ftr_lst = []
	line_lst = line.split(",")

	nitems = len(line_lst)
	i = 0

	while (i<nitems):

	    #isso se chegar em uma feature que esta entre aspas
	    #e se esta feature entre aspas nao foi dividida? Ela vai ter 
	    # um open quote e um close quote

	    if (open_quote.match(line_lst[i])):
		f = ""
		#enquanto nao for close quote
		while (not(line_lst[i]).endswith("\"")):
		    f+=line_lst[i]
		    i=i+1
		f+=line_lst[i]
		ftr_lst.append(f)
	    else:
		ftr_lst.append(line_lst[i])
	    i = i+1
	return ftr_lst

    def read_features(self,file_name,init_id=1):
	#dado um arquivo de features (csv) de varios documentos
	#que sao mapeados de acordo com a data que a oferta 
	#foi publicada

	doc_list = {}
	target_value = {}
	fd = open(file_name,"r")
	ftrs_lines = fd.readlines()
	self.header = ftrs_lines.pop(0).split(",")
	pos_key,header_pos = self.pre_process_header()
	for l in ftrs_lines:
	    #tem que transformar estas features em seus respectivos tipos
	    ftrs_values,target = self.process_values(header_pos,self.tokenize_features(l))
	    n = len(ftrs_values)
	    key = ""
	    #colocar o identificador do documento na lista
	    #mas ai cho que vai ser dificil localizar tbm? Nao, pois daqui 
	    #ja pego as features textuais do documento
	    ftrs_values = [init_id ] + ftrs_values
	    #pos_key+1 pois acrescentei o id do documento na frente
	    #print ftrs_values[pos_key+1]
	    if (ftrs_values[pos_key+1] in doc_list):
	         doc_list[ftrs_values[pos_key+1]].append(ftrs_values)
		 target_value[ftrs_values[pos_key+1]].append(target)
	    else:
	         doc_list[ftrs_values[pos_key+1]] = [ftrs_values]
		 target_value[ftrs_values[pos_key+1]] = [target]
	    init_id = init_id +1 
	fd.close()
	return doc_list,target_value

    def split_by_topic(self,i,dates,data,target):
	"""
	Given the topics of the current self.model_topic,
	separate topic by 
	"""
	new_data = []

	#TODO: Acho que farei o processo a seguir na funcao que a chama mesmo
	#separar o historico ateh a data dates[i] remover a feature docid
	for k in xrange(0,i):
	    for deal in data[dates[k]]:
	        doc_id = deal[0]
		     ftrs_txt = doc_list_txt[doc_id]
	             new_doc_list_txt.append(ftrs_txt)
		
	data_bytopic = {}
	target_bytopic = {}

    def train(self,idate,date,doc_list,target_value,doc_list_txt,dict_num):

	date_nums = date[idate].split("/")
	date_nums_str = ""

	for d in date_nums:
	    date_nums_str += "_" + d

	lda_model_file = self.lda_dir + self.data_name + date_nums_str

	if (os.path.isfile(lda_model_file)):
	    model_topic = ldamodel.LdaModel.load(lda_model_file)
	else:
	    #dados textuais para o lda
	    new_doc_list_txt = []

            #capturando os dados textuais das ofertas do dia
	    i = 0
	    for deal in doc_list[date[idate]]:
	        doc_id = deal[0]
		try:
		     ftrs_txt = doc_list_txt[doc_id]
	             new_doc_list_txt.append(ftrs_txt)
		except KeyError:
		    print "*** Warning *** There is no document ",doc_id

	    #fazer o treino do lda
	    if (idate == 1):
	         self.model_topic = models.LdaModel(corpus=new_doc_list_txt,id2word=dict_num,num_topics=self.k)
	    else:
		self.model_topic.update(new_doc_list)
	    self.model_topic.save(lda_model_file)

	#separar os docs de acordo com os topicos encontrados
	#dados de treino por topico
	train_data_topic = self.split_by_topic(idate,date,doc_list,target_value)

        #TODO: tem que aplicar no conjunto de teste, mas como esta 
	#em um self vou fazer em outra funcao. Fique ciente que 
	#isso muda na proxima iteracao


	#fazer o treino do svr: para cada topico!!
	#TODO: salvar e carregar modelos
	for t in xrange(0,self.k):
	     svr_model_file = self.svr_dir + self.data_name + date_nums_str + "_"\
		     + t + ".plk"

	     if (os.path.isfile(svr_model_file)):
	         self.model_deal_size = joblib.load(svr_model_file)
	     else:
	         self.model_deal_size = svm.SVR()
	         self.model_deal_size.fit(new_data_list,new_target_data)
	         joblib.dump(self.model_deal_size,svr_model_file)

    def process(self,dir_data,file_features):

	#dir_data: guarda o diretorio dos arquivos json de features textuais
	#lendo features textuais
	mydocObj = mycorpus.MyCorpus(self.ftr_list)
	#doc_txt_list: documentos em formato de lista de numeros
	#dict_num: dicionario que mapeia numero->palavra
	doc_txt_list,dict_num = mydocObj.process(dir_data)

        #lendo features para o SVR
	doc_list,target_value = self.read_features(file_features)
	#percorrer um dia de cada vez de ordem crescente
	#pegar os dias...
	dias = doc_list.keys()
	#..e ordenar crescentemente
	dias = sorted(dias,cmp=compara_datas)
        ndias = len(dias)
	deals_of_day = []
	#print "-->",dias[ndias-1]
	for i in xrange(0,ndias):
	    if (i < (ndias-1)): 
	        deals_of_day = doc_list[dias[i]]
	        deals_of_nextday = doc_list[dias[i+1]]

		#treino nas deals 
		#descobre topicos com estas deals
		# e treina o SVR
		print "Treino: ",dias[i]
		print "Teste: ",dias[i+1]
		self.train(i,dias,doc_list,target_value,doc_txt_list,dict_num)



    def compare_dates(self):
	#TODO
	pass

if __name__ == "__main__":
    ml_obj = machine_learning("options.conf")
    ml_obj.process("../daily-deals-data/ls-json/","../daily-deals-data/ls-deals.csv")
