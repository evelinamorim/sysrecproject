## machine learning to given objects
import mycorpus
import deal
import evaluation

import config
import util
import sys
import os
import time 

from sklearn.externals import joblib
from sklearn import feature_extraction
from sklearn import svm
from sklearn import linear_model,grid_search,preprocessing,decomposition
from gensim import corpora, models, similarities
from gensim.models import ldamodel

import lda
import numpy as np
import random



class machine_learning:

    def __init__(self,config_file,k=10):
	random.seed("2001")
	self.__config_obj = config.Config(config_file)
	self.__config_obj.read()
	self.__config_filename = config_file

	self.lda_dir = self.__config_obj.get_value("lda_dir")[0]
	self.svr_dir = self.__config_obj.get_value("svr_dir")[0]
	self.data_name = self.__config_obj.get_value("data_name")[0]

	self.ftr_list = self.__config_obj.get_value("ftr_names")
	self.ftr_txt = self.__config_obj.get_value("ftr_txt")
	self.header = []
	#o header que foi pre processado
	self.new_header = [] 
	self.target_ftr = -1
	self.key_ftr = self.__config_obj.get_value("key_ftr")[0]
	self.target = {}
	#k eh o numero de treinos em temas
	self.k = k
	self.model_topic = None
	self.model_deal_size = {}	
	self.__svr_parameters = {'kernel':['linear','rbf'],'C':[0.01,0.1,1,5,10,50],'gamma': [0.0,0.001,0.0000001]}

	self.__scaler = {}
        #self.__svr_parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	#historico de features de ofertas ateh uma data corrente
	self.__deal_history = []
	self.__pred  = {}

	self.__vec = feature_extraction.DictVectorizer()
	#numero de features transformadas
	self.__nvec = -1

    def get_conf_field(self,field):
	return self.__config_obj.get_value(field)

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
		pos_key = i
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


    def market_proportion(self):
	"""
	Calcula a  proporcao em termos de deal size de cada mercado
	"""
        mp = {}
	catalog_size = 0.0
	for t in self.data_bytopic:
	    topic_size = 0.0
	    for d in self.data_bytopic[t]:
		topic_size += d.get_target()
	    catalog_size += topic_size
	    mp[t] = topic_size

	for t in mp:
	    mp[t] = mp[t]/catalog_size
	return mp

    def compute_sigma(self,rho,t,docids):
	"""
	computar sigma para cada oferta no conjunto de treino
	"""

	sigma = {}

        #calculando primeira parte da formula
	competition = 0.0
	nonpredict = 0
	#data_bytopic eh de dados do treino, entao esta valendo
	for d in self.data_bytopic[t]:
	    try:
	        predict = self.__pred[d.get_docid()]
	        print ">>",docids[d.get_docid()],d.get_target(),predict
	        competition = competition + (predict*rho)
	    except KeyError:
		nonpredict = nonpredict + 1
                print "-->",d.get_docid(),d.get_docid() in self.__pred
		print "compute_sigma Warning Keyerror"

        real_pred = len(self.data_bytopic[t])-nonpredict
	if (real_pred == 0):
	    return []
	competition = competition/ real_pred
	print "Competition: ",competition,t

	sigma = []
	for d in self.data_bytopic[t]:
	     try:
	         predict = self.__pred[d.get_docid()]
                 s = (0.5*competition) + (0.5*predict)
		 print "==>",predict,competition,s,d.get_target()
	         d.set_sigma(s)
	         sigma.append(d)
	     except KeyError:
		 pass
        return sigma

    def global_sigma(self,sigma):
	global_size = 0.0
	for t in sigma:
	    topic_size = 0.0
	    for d in sigma[t]:
	        topic_size = topic_size + d.get_sigma()
	    global_size += topic_size
	return global_size

    def local_sigma(self,sigma,t):

	topic_size = 0.0
	for d in sigma[t]:
	    topic_size = topic_size + d.get_sigma()

        return topic_size

    def expectation_maximization(self,i,dates,global_pred,docids):
	#global_pred 
	rho = self.market_proportion()
	print rho

        target_doc = {}
	for t in self.data_bytopic:
	    for deal in self.data_bytopic[t]:
		target_doc[docids[deal.get_docid()]] = deal.get_target()

	k = 0
	max_iter = 5
        eval_obj = evaluation.Evaluation(self.__config_filename)
	sigma = {}
	#descobrindo rho: soh quem esta no treino
	e = 0
	while (k<max_iter):
	    pred = {}
	    diff = e
	    e = 0
	    for t in self.data_bytopic:
	        sigma[t] = self.compute_sigma(rho[t],t,docids)
		    #for deal in sigma[t]:
		#	did = deal.get_docid()
		#	self.__pred[did] = random.uniform(-1,1)*e + self.__pred[did]
		if (sigma[t] == []):
		    continue
	        global_s = self.global_sigma(sigma)
		local_s = self.local_sigma(sigma,t)
	        rho[t] = local_s/global_s
	        pred[t] = [(docids[d.get_docid()],d.get_sigma()) for d in sigma[t]]
	        e += eval_obj.rmse(target_doc,pred[t])
	    e  = e/len(self.data_bytopic)
	    #print ">>>",e
            k = k+1
	import sys
	sys.exit()
	#predizer os novos valores de teste
	pred = {}
	market = {}
	for t in global_pred:
	    market[t] = 0.0
	    for (docid,p) in global_pred[t]:
		 if (t in rho):
	             market[t] += p*rho[t]
            market[t] = market[t]/len(global_pred[t])

        #print "EM"
	for t in global_pred:
	    pred[t] = []
	    for (docid,p) in global_pred[t]:
	        new_p = 0.5*(market[t]) + 0.5*p
		pred[t].append((docid,new_p))
		self.__pred[docid] = new_p
            #print pred[t]
        return pred

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
	    tokens = util.tokenize_features(l)
	    ftrs_values,target = self.process_values(header_pos,tokens)

	    general_doc_list.append(deal.Deal(ftrs=ftrs_values).get_ftrs_dict(self.new_header))
	    ftrs_values = [init_id ] + ftrs_values

	    type_key = self.__config_obj.get_value( self.key_ftr + "_type")[0]
	    key = util.turn2type(tokens[pos_key],type_key)

	    #pos_key+1 pois acrescentei o id do documento na frente
	    #print ftrs_values[pos_key+1]
	    if (key in doc_list):
	         doc_list[key].append(ftrs_values)
		 target_value[key].append(target)
	    else:
	         doc_list[key] = [ftrs_values]
		 target_value[key] = [target]
	    init_id = init_id +1 

	fd.close()
	self.__vec.fit(general_doc_list)
	return doc_list,target_value

    def split_by_topic(self,i,dates,data,target,docids):
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
        @docids: lista de documentos na base
	"""

        #separar os dados por data
	ndeal_day = len(data[dates[i]])#pegar as deals do dia dado
	for k in xrange(0,ndeal_day):
            #adicionando as ofertas e ignorando o docid, que esta
            #na posicao 0
	    #k-esima deal no dia date[i]
	    try:
	        doc_id = data[dates[i]][k][0]
                doc_id_real = docids.index(doc_id)
	        d = deal.Deal(doc_id_real,data[dates[i]][k][1:],target[dates[i]][k])
	        self.__deal_history.append(d)
	    except ValueError:
		print "*** Warning *** There is no document ",doc_id


        #separar os dados por topico
	self.data_bytopic = {}
	target_bytopic = {}

	#para cada deal no historico verificar em qual topico ela esta
	for deal_item in self.__deal_history:
	    #topic_list = self.model_topic.doc_topic_[deal_item.get_docid()].tolist()
	    topic_list = self.model_topic[self.txt_ftrs[deal_item.get_docid()]]

	    #if (docids[deal_item.get_docid()] == 1 or docids[deal_item.get_docid()] == 1686):
            #print ">>>>",docids[deal_item.get_docid()],deal_item.get_docid()
	    #word_list = [self.global_dict[w] for (w,ts) in self.txt_ftrs[deal_item.get_docid()]]
	    #print word_list
	    #print deal_item.get_ftrs()
	    #print topic_list
            #topic_list = [(i,topic_list[i]) for i in xrange(self.k)]
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

        
	self.topic = self.__config_obj.get_value("topic")[0]
	if (self.topic=="top"):
	    ntopics = len(topic_list)-1
	    t,prob = topic_list[ntopics]
	    if (t in data_bytopic):
	        data_bytopic[t].append(deal_item)
	    else:
	        data_bytopic[t] = [deal_item]
	else:
	    if (self.topic=="all"):
		for (t,prob) in topic_list:
		    deal_item.set_prob(t,prob)
		    if (t in data_bytopic):
		        data_bytopic[t].append(deal_item)
		    else:
			data_bytopic[t] = [deal_item]
	    elif (self.topic == "threshold"):
	        thr = float(self.__config_obj.get_value("topic")[1])
		for (t,prob) in topic_list:
		    if (prob>=thr):
			deal_item.set_prob(t,prob)
			if (t in data_bytopic):
			    data_bytopic[t].append(deal_item)
			else:
			    data_bytopic[t] = [deal_item]
	    else:
		print "** Error ** add2topic - topic not founnd"
		sys.exit()

    def pred_local(self,X,deals_topic,t,docids):
	"""
	Realiza predicoes de acordo com a opcao dada no arquivo de 
	configuracoes:
	* random: utiliza o rotulo e acrescenta uma pertubacao como predicao
	* svr: utiliza a predicao do svr

	"""

	self.pred_type = self.__config_obj.get_value("prediction_type")[0]

	if (self.pred_type == "svr"):
             if (t in self.model_deal_size and self.model_deal_size[t]!=None):
                 vec_test_data = self.__vec.transform(X)
		 vec_scale = self.__scaler[t].transform(vec_test_data)
		 pred_t = self.model_deal_size[t].predict(vec_scale)

		 ndealstopic = len(deals_topic[t]) 

	         return pred_t
	     else:
		ndealstopic = len(deals_topic[t]) 
		for k in xrange(ndealstopic):
		    doc_id_real = deals_topic[t][k].get_docid() 
		    self.__pred[doc_id_real] = deals_topic[t][k].get_target()

		return []
	elif (self.pred_type == "rand"):
	    pred_t = []
            ndealstopic = len(deals_topic[t]) 

            for k in xrange(ndealstopic):
	        real_target = deals_topic[t][k].get_target()
	        rand_target = random.uniform(0,0.1)*real_target*\
		        random.choice([1,-1])
		docid = docids[deals_topic[t][k].get_docid()]
		#sys.stdout.write( ">>> " + str(docid) + " "+ str(deals_topic[t][k].get_target()) + " " + str(rand_target) + " ")
	        rand_target = deals_topic[t][k].get_target()+rand_target
		sys.stdout.write(str(rand_target) + "\n")
	        pred_t.append(rand_target)
	    return pred_t
        else:
	    print "** Error ** Type of prediction not found"
	    sys.exit()

    def add2pred(self,pred_t,deals_topic,t):
	"""
	Inicializa as predicoes e faz o ensemble de acordo com 
	o tipo dado no arquivo de configuracao
	"""
	self.topic = self.__config_obj.get_value("topic")[0]

	if (self.topic == "top"):
             ndealstopic = len(deals_topic[t]) 
             for k in xrange(ndealstopic):
		 doc_id_real = deals_topic[t][k].get_docid() 
		 deals_topic[t][k].set_pred(pred_t[k])

	elif (self.topic == "all" or self.topic == "threshold"):
             ndealstopic = len(deals_topic[t]) 
             for k in xrange(ndealstopic):
		 doc_id_real = deals_topic[t][k].get_docid() 
		 new_pred_t = deals_topic[t][k].get_prob(t)*pred_t[k]
		 new_pret_t = new_pred_t + deals_topic[t][k].get_pred()
		 deals_topic[t][k].set_pred(new_pred_t)
	else:
	    print "** Error ** add2topic - topic not founnd"
	    sys.exit()

    def test(self,i,dias,doc_list,docids,target_values):
	"""
	Prediz com o SVR o deal size
	TODO: predizer com o expectation-maximization

	@i: predizer o i+1-esimo dia  com base no topico das 
	deals dadas e em suas features textuais
	@dias: data de dias na base de dados
	@doc_list: features das deals
        @docids: lista de ids de documentos
	"""

	#predizer o topico de cada oferta em test_data
	deals_topic ={} 
	k = 0
	for d in doc_list[dias[i+1]]:
	    doc_id = d[0]
	    if (doc_id in docids):
                doc_id_real = docids.index(doc_id)

	        topic_list = self.model_topic[self.txt_ftrs[doc_id_real]]
                target = target_values[dias[i+1]][k]
	        #usar o modelo corrente para classificar o topico
		topic_list.sort(key=lambda tup: tup[1])
		deal_obj = deal.Deal(doc_id_real,d[1:],target)
		self.add2topic(topic_list,deal_obj,deals_topic)
	    else:
		print "*** Warning *** There is no documento ",doc_id
            k = k+1

        #apos a divisao por topicos, classificar com o svr
	pred = {}
	for t in deals_topic:
	     data_t = [k.get_ftrs_dict(self.new_header) for k in deals_topic[t]]

             #de acordo com a configuracao faz a predicao paras deals 
	     #de um dado topico t
	     pred_t = self.pred_local(data_t,deals_topic,t,docids)

	     if (pred_t !=[]):

	          self.add2pred(pred_t,deals_topic,t)

                  ndealstopic = len(deals_topic[t]) 
	          for k in xrange(ndealstopic):
		      doc_id_real = deals_topic[t][k].get_docid() 
		      self.__pred[doc_id_real] = deals_topic[t][k].get_pred()
		      pred[docids[doc_id_real]] = deals_topic[t][k].get_pred()

        
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

    def train(self,idate,date,doc_list,target_value,docids):
        """
          Treino com os modelos SVRs
          @docids: lista de documentos disponiveis na base de dados
        """

	date_num = date[idate]
        
	lda_model_file = self.lda_dir + self.data_name + str(date_num)

	#separar os docs de acordo com os topicos encontrados
	#dados de treino por topico
        self.split_by_topic(idate,date,doc_list,target_value,docids)

	#treinando cada topico um svr
	for t in self.data_bytopic:
	    train_data_topic = []
	    target_data_topic = []

            id_data_topic = []
	    for d in self.data_bytopic[t]:
		 docid = d.get_docid()
		 train_data_topic.append(d.get_ftrs_dict(self.new_header))
                 id_data_topic.append(docid)
		 target_data_topic.append(d.get_target())
		 #print docids[docid],",",d.get_target()
            #print "-->",t,id_data_topic
            if (not(self.one_class(target_data_topic))):

		 vec_train_data_topic = self.__vec.transform(train_data_topic)

		 if (len(train_data_topic)>=3):
		      self.model_deal_size[t] = grid_search.GridSearchCV(\
		    	   svm.SVR(),\
		               self.__svr_parameters)
		 else:
		      self.model_deal_size[t] = svm.SVR(kernel='linear')

	         self.__scaler[t] = preprocessing.StandardScaler(\
		         with_mean=False).fit(\
		         vec_train_data_topic)

		 vec_scale = self.__scaler[t].transform(vec_train_data_topic)

		 self.model_deal_size[t].fit(vec_scale, target_data_topic)
		 pred_t = self.model_deal_size[t].predict(vec_scale)
		 #print pred_t,target_data_topic
		 #soh no primeiro dia do treino
		 if (idate==0):
		      k = 0
		      for iddoc in id_data_topic:
		          self.__pred[iddoc] = pred_t[k]
		          k = k+1

	         #joblib.dump(self.model_deal_size[t],svr_model_file)
            else:
	         #nao sei bem ainda o que faezr na predicao deste caso
	         self.model_deal_size[t] = None
            k = 0
            for iddoc in id_data_topic:
	         self.__pred[iddoc] = target_data_topic[k]
		 k = k+1
	    
        

    def process(self,dir_data,file_features,pred_file="pred.out"):

	#dir_data: guarda o diretorio dos arquivos json de features textuais
	#lendo features textuais
	mydocObj = mycorpus.MyCorpus(self.ftr_txt)

	#doc_txt_list: documentos em formato de lista de numeros
	#lista de ids de documentos
	doc_list_txt_ftrs,docids,dictionary = mydocObj.process(dir_data)
	self.global_dict = dictionary
	self.txt_ftrs = doc_list_txt_ftrs

        #separando em mercados
	self.model_topic = ldamodel.LdaModel(\
		corpus=doc_list_txt_ftrs,id2word=dictionary,\
		num_topics=self.k,iterations=100,passes=2,\
		gamma_threshold=0.00001)

        #lendo features para o SVR
	doc_list,target_value = self.read_features(file_features)

	#percorrer um dia de cada vez de ordem crescente
	#pegar os dias...
	dias = doc_list.keys()
        #print "Dias:",dias
	#..e ordenar crescentemente
	dias = sorted(dias)
        ndias = len(dias)
        total = 0
	for i in xrange(0,ndias):
	    if (i < (ndias-1)): 

		#treino nas deals 
		#descobre topicos com estas deals
		# e treina o SVR
		#print "#doc: ",len(doc_txt_list.keys())
		print "Treino: ",dias[i]
		self.train(i,dias,doc_list,target_value,docids)
		print "Teste: ",dias[i+1]
                total += len(doc_list[dias[i+1]])
		pred = self.test(i,dias,doc_list,docids,target_value)
		#target_value esta por data
		#pred_em = self.expectation_maximization(i,dias,pred,docids)
		#print pred
	        #self.write_predictions(pred_em,pred_file)
	        self.write_predictions(pred,pred_file)


    def write_predictions(self,predictions,pred_file):

	fd = open(pred_file,"a")

	for doc in predictions:
	         #fd.write("%d,%s,%f\n" %(topic,doc,pred))
	         fd.write("%s,%f\n" %(doc,predictions[doc]))
            #fd.write('\n')

	fd.close()


if __name__ == "__main__":
    #TODO: ver o que tem no target, pois as predicoes estao com numero baixo!
    # no target esta com valor errado tbm. Conferir se esta pegando certo mesmo
    # Os numeros estao melhores
    #Esquema de peso utilizado: term spread
    k = int(sys.argv[1])
    ml_obj = machine_learning("options.conf",k)

    topic_strategic = ml_obj.get_conf_field("topic")[0]
    pred_file = "pred_" + topic_strategic + "_" + str(k) + ".out"

    jsondir = ml_obj.get_conf_field("jsondir")[0]

    cvsfile = ml_obj.get_conf_field("cvsfile")[0]

    ml_obj.process(jsondir,cvsfile,pred_file)
