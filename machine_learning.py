## machine learning to given objects
import mycorpus
from gensim import corpora, models, similarities

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
    def __init__(self,ftr_list,k=5):
	#k eh o numero de treinos em temas
	self.ftr_list = ftr_list
	self.k = k

    def read_features(self,file_name,init_id=1):
	#dado um arquivo de features de varios documentos
	#que sao mapeados de acordo com a data que a oferta 
	#foi publicada

	doc_list = {}
	fd = open(file_name,"r")
	ftrs_lines = fd.readlines()
	header = ftrs_lines.pop(0).split(",")

	pos_key = -1
	len_header = len(header)
	for i in xrange(0,len_header):
	    if (header[i] == "started_at"):
		pos_key = i
		break

        if (pos_key == -1):
	    print "*** WARNING *** Key feature value has not been identified"

	for l in ftrs_lines:
	    ftrs_values = l.split(",")
	    n = len(ftr_values)
	    key = ""
	    #colocar o identificador do documento na lista
	    #mas ai cho que vai ser dificil localizar tbm? Nao, pois daqui 
	    #ja pego as features textuais do documento
	    ftrs_values = [init_id ] + ftrs_values
	    if (ftrs_values[pos_key+1] in doc_list):
	         doc_list[ftrs_values[pos_key+1]].append(ftrs_values)
	    else:
	         doc_list[ftrs_values[pos_key+1]] = [ftrs_values]
	    init_id = init_id +1 
	fd.close()
	return doc_list

    def train(self,data_list,doc_list_txt,dict_num):

	new_data_list = []
	new_doc_list_txt = []
	#ler cada deal e retirar o id dentre as features
	#TODO: ver qual o label de treino para o SVR tambem
	for deal in data_list:
	    doc_id = deal.pop(0)
	    new_data_list.append(deal)
	    new_doc_list_txt.append(doc_list_txt[doc_id])

	#fazer o treino do lda
	model_topic = models.LdaModel(corpus=new_doc_list_txt,id2word=dict_num,num_topics=self.k)

	#fazer o treino do svr


    def process(self,data,file_features):

	#dir_data: guarda o diretorio dos arquivos json de features textuais
	mydocObj = mycorpus.MyCorpus(self.ftr_list)
	doc_txt_list,dict_num = mydocObj.process(data)

	doc_list = self.read_features(file_features)
	#percorrer um dia de cada vez de ordem crescente
	#pegar os dias...
	dias = doc_list.keys()
	#..e ordenar crescentemente
	dias = sorted(dias,cmp=cmp_datas)

        ndias = len(dias)
	deals_of_day = []
	for i in xrange(0,ndias):
	    if (i < (ndias-1)): 
	        deals_of_day += doc_list[dias[i]]
	        deals_of_nextday = doc_list[dias[i+1]]

		#treino nas deals 
		#descobre tópicos com estas deals
		# e treina o SVR
		self.train(deals_of_day,doc_list_txt,dict_num)



    def compare_dates(self):
	#TODO
	pass
