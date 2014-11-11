## machine learning to given objects
import mycorpus
import deal

import config
import util
import sys
import re
import os
import time 

from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn import svm
from gensim import corpora, models, similarities
from gensim.models import ldamodel

open_quote = re.compile("\"[\w]+")
close_quote = re.compile("[\w\.\-]+\"")



class machine_learning:

    def __init__(self,config_file,k=10):
	self.__config_obj = config.Config(config_file)
	self.__config_obj.read()

	self.lda_dir = self.__config_obj.get_value("lda_dir")[0]
	self.svr_dir = self.__config_obj.get_value("svr_dir")[0]
	self.data_name = self.__config_obj.get_value("data_name")[0]

	self.ftr_list = self.__config_obj.get_value("ftr_names")
	self.header = []
	#o header que foi pre processado
	self.new_header = [] 
	self.target_ftr = -1
	self.target = {}
	#k eh o numero de treinos em temas
	self.k = k
	self.model_topic = None
	self.model_deal_size = {}

	#historico de features de ofertas ateh uma data corrente
	self.__deal_history = []
	self.__deal_txt_history = []
	self.__target_history = []

	self.__vec = feature_extraction.DictVectorizer()
	#numero de features transformadas
	self.__nvec = -1

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

	target_name = self.__config_obj.get_value("target_ftr")[0]

	#quais as posicoes em header que serao efetivamente usadas
	for i in xrange(0,nvalues):
	    if (self.header[i] in self.ftr_list):
		new_header_pos.append(i)
		self.new_header.append(self.header[i])
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
	general_doc_list = []
	for l in ftrs_lines:
	    #tem que transformar estas features em seus respectivos tipos
	    ftrs_values,target = self.process_values(header_pos,self.tokenize_features(l))
	    n = len(ftrs_values)
	    key = ""
	    #colocar o identificador do documento na lista
	    #mas ai cho que vai ser dificil localizar tbm? Nao, pois daqui 
	    #ja pego as features textuais do documento
	    general_doc_list.append(deal.Deal(ftrs=ftrs_values).get_ftrs_dict(self.new_header))
	    ftrs_values = [init_id ] + ftrs_values
	    #pos_key+1 pois acrescentei o id do documento na frente
	    #print ftrs_values[pos_key+1]
	    if (ftrs_values[pos_key+1] in doc_list):
	         doc_list[ftrs_values[pos_key+1]].append(ftrs_values)
		 target_value[ftrs_values[self.target_ftr+1]].append(target)
	    else:
	         doc_list[ftrs_values[pos_key+1]] = [ftrs_values]
		 target_value[ftrs_values[self.target_ftr+1]] = [target]
	    init_id = init_id +1 
	fd.close()
	self.__vec.fit(general_doc_list)
	return doc_list,target_value

    def split_by_topic(self,i,dates,data,target,data_txt):
	"""

	Given the topics of the current self.model_topic,
	separate topic by

	TODO: esse historico eu poderia guardar direto na classe nao?

	@i: indice que determina qual data devo coletar como historico 
	    nos dados das ofertas fornecidas
	@date: vetor de datas existentes nos dados. Estas datas estao 
	       ordenadas em forma crescente
	@data: dicionario que mapeia data -> features das ofertas
	@target: dicionario que mapeia data para valor do deal size. A ordem 
	        que os valores do target foram adicionados, eh a mesma 
		ordem que os valores foram adicionados em data
	@data_txt:dicionario que mapeia um documento (doc_id) para sua features 
	textuais
	"""

        #separar os dados por data
	ndeal_day = len(data[dates[i]])#pegar as deals do dia dado
	for k in xrange(0,ndeal_day):
            #adicionando as ofertas e ignorando o docid, que esta
            #na posicao 0

	    #k-esima deal no dia date[i]
	    try:
	        doc_id = data[dates[i]][k][0]
                #print data[dates[i]][k],len(data[dates[i]][k])
		#print ">",doc_id,target[dates[i]][k]
	        #Deal(doc_id,daeal_ftrs,deal_ftrs_txt,target_deal)
	        d = deal.Deal(doc_id,data[dates[i]][k][1:],data_txt[doc_id],target[dates[i]][k])
	        self.__deal_history.append(d)
	    except KeyError:
		print "*** Warning *** There is no document ",doc_id


        #separar os dados por topico
	self.data_bytopic = {}
	target_bytopic = {}

	#para cada deal no historico verificar em qual topico ela esta
	# o modelo eh sempre atualizado, entao eu tenho que reclassificar 
	#o historico sempre
	for deal_item in self.__deal_history:
	    topic_list = self.model_topic[deal_item.get_ftrs_txt()]
	    #ordenar da menor probabilidade para a maior
	    topic_list.sort(key=lambda tup: tup[1])

	    self.add2topic(topic_list,deal_item,self.data_bytopic)

    def add2topic(self,topic_list,deal_item,data_bytopic):
	"""
	Dada uma lista de probabilidades adicionar o item para cada topico
	de acordo com alogum criterio definido:
	* de acordo com o threshold da probabilidade
	* de acordo com um threshold para numero de topicos
	* apenas um topico
	"""

	ntopics = len(topic_list)-1
	t,prob = topic_list[ntopics]
	if (t in data_bytopic):
	     data_bytopic[t].append(deal_item)
	else:
	     data_bytopic[t] = [deal_item]

    def test(self,i,dias,doc_list,doc_txt_list,dict_num):
	"""
	Prediz com o SVR o deal size
	TODO: predizer com o expectation-maximization

	@i: predizer o i+1-esimo dia  com base no topico das 
	deals dadas e em suas features textuais
	@dias: data de dias na base de dados
	@doc_list: features das deals
	@doc_txt_list: features textuais
	@dict_num: dicionario de mapeamento num->palavras
	"""

	#predizer o topico de cada oferta em test_data
	deals_topic ={} 
	for d in doc_list[dias[i+1]]:
	    doc_id = d[0]
	    try:
	        ftrs_txt = doc_txt_list[doc_id]
	        #usar o modelo corrente para classificar o topico
		topic_list = self.model_topic[ftrs_txt]
		topic_list.sort(key=lambda tup: tup[1])
		deal_obj = deal.Deal(doc_id,d[1:],doc_txt_list[doc_id])
		self.add2topic(topic_list,deal_obj,deals_topic)
		#print d,(len(d))!=11
	    except KeyError:
		print "*** Warning *** There is no documento ",doc_id

        #apos a divisao por topicos, classificar com o svr
	pred = {}
	for t in deals_topic:
	    #como acessar o doc_id para depois fazer a avalicao?
	    #for k in deals_topic[t]:
	    #     print len(k.get_ftrs_dict(self.new_header))
	    data_t = [k.get_ftrs_dict(self.new_header) for k in deals_topic[t]]
	    #print data_t
            vec_test_data = self.__vec.transform(data_t)

	    pred_t = self.model_deal_size[t].predict(vec_test_data)
	    ndealstopic = len(deals_topic[t])
	    #TODO: esse i  esta estranho
	    pred[t] = [(deals_topic[t][k].get_docid(),pred_t[k]) for k in xrange(0,ndealstopic)]

        return pred

    def one_class(self,target):
	"""
	Checa se soh existe apenas uma classe neste grupo
	"""
	if (len(target)==1):
	    return True
	else:
	    i = 0
	    elem  = target[i]
	    ntargets = len(target)
	    i = i+1
	    while (i < ntargets):
		if (target[i]!=elem):
		    return False
		i = i+1
	    return True

    def train(self,idate,date,doc_list,target_value,doc_list_txt,dict_num):

	date_num = date[idate]

	lda_model_file = self.lda_dir + self.data_name + str(date_num)

	if (os.path.isfile(lda_model_file)):
	    self.model_topic = ldamodel.LdaModel.load(lda_model_file)
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
	    if (self.model_topic == None):
	         self.model_topic = models.LdaModel(corpus=new_doc_list_txt,id2word=dict_num,num_topics=self.k)
	    else:
		self.model_topic.update(new_doc_list_txt)
	    self.model_topic.save(lda_model_file)

	#separar os docs de acordo com os topicos encontrados
	#dados de treino por topico
        self.split_by_topic(idate,date,doc_list,target_value,doc_list_txt)

	#print "Data by Topic:",self.data_bytopic.keys()

	#treinando cada topico um svr
	for t in self.data_bytopic:
	    train_data_topic = []
	    target_data_topic = []

	    for d in self.data_bytopic[t]:
		 train_data_topic.append(d.get_ftrs_dict(self.new_header))
		 target_data_topic.append(d.get_target())
            #print "==>",train_data_topic
            if (not(self.one_class(target_data_topic))):


		 vec_train_data_topic = self.__vec.transform(train_data_topic)



	         #TODO: se existe apenas uma classe o svr da divisao por 0!
	         # e ai? nao treinar e deixar no historico? Ou usar outro classificador?
	         #pq de certa forma nao tem nada a aprender
	         #print "1: ",target_data_topic
	         # print "2: ",train_data_topic
                 svr_model_file = self.svr_dir + self.data_name + str(date_num) \
			 + "_" + str(t) + ".plk"
	         if (os.path.isfile(svr_model_file)):
	             self.model_deal_size[t] = joblib.load(svr_model_file)
		 else:
	             self.model_deal_size[t] = svm.SVR()
		     #print target_data_topic
	             self.model_deal_size[t].fit(vec_train_data_topic,target_data_topic)
	             joblib.dump(self.model_deal_size[t],svr_model_file)
            else:
	         #nao sei bem ainda o que faezr na predicao deste caso
	         self.model_deal_size[t] = None
	    


    def process(self,dir_data,file_features,pred_file="pred.out"):

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
	dias = sorted(dias)
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
		#print "#doc: ",len(doc_txt_list.keys())
		print "Treino: ",dias[i]
		self.train(i,dias,doc_list,target_value,doc_txt_list,dict_num)
		print "Teste: ",dias[i+1]
		pred = self.test(i,dias,doc_list,doc_txt_list,dict_num)
		#print pred
		self.write_predictions(pred,pred_file)


    def write_predictions(self,predictions,pred_file):

	fd = open(pred_file,"a")

	for topic in predictions:
	    for (doc,pred) in predictions[topic]:
	         fd.write("%s,%f\n" %(doc,pred))
            fd.write('\n')

	fd.close()

    def compare_dates(self):
	#TODO
	pass

if __name__ == "__main__":
    #TODO: ver o que tem no target, pois as predicoes estao com numero baixo!
    # no target esta com valor errado tbm. Conferir se esta pegando certo mesmo
    # Os numeros estao melhores
    #Esquema de peso utilizado: term spread
    ml_obj = machine_learning("options.conf")
    ml_obj.process("../daily-deals-data/ls-json/","../daily-deals-data/ls-deals.csv")
