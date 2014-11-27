import math
import config
import util
import sys
import os
import re
import time

class Evaluation:

    def __init__(self,config_file):

	self.__config_obj = config.Config(config_file)
	self.__config_obj.read()
	self.__config_file = config_file
	self.__mean = None
	self.__std_dev = None

    def read_truth(self,file_name):

	target_name = self.__config_obj.get_value("target_ftr")[0]
	target_type = self.__config_obj.get_value(target_name + "_type")[0]

	fd = open(file_name,"r")
	lines = fd.readlines()

	#pegar posicao da target feature
	#retirando o header
	h = lines.pop(0).split(",").index(target_name)

	docid = 1
	truth = {}
	mean = 0.0
	for l in lines:
	    tokens = util.tokenize_features(l)
	    truth[docid] = util.turn2type(tokens[h],target_type)
	    mean = mean + truth[docid]
	    docid = docid+1

        mean = mean / docid
	self.__mean = mean
	std_dev = 0.0
	for docid in truth:
	    term = truth[docid] - mean
	    std_dev = std_dev + (term*term)
	std_dev = math.sqrt(std_dev/docid)
	self.__std_dev = std_dev

        normalized_truth = {}
	for docid in truth:
	    normalized_truth[docid] = (truth[docid]-mean)/std_dev
	fd.close()
	return normalized_truth

    def read_pred(self,file_name):

	fd = open(file_name,"r")
	pred = []

	for l in fd:
	    tokens = l.split(",")
	    if (len(tokens)>=2):
		docid = int(tokens[0])
		p = float(tokens[1])
		norm_p = (p-self.__mean)/self.__std_dev
		pred.append((docid,norm_p))
	fd.close()
	return pred

    def read_pred_topic(self,file_name):
	"""
	le um arquivo de predicao com topico
	"""

	fd = open(file_name,"r")
	pred = {}
	pred_regular = []

	for l in fd:
	    tokens = l.split(",")
	    #print tokens
	    if (len(tokens)>=2):
		docid = int(tokens[0])
		p = float(tokens[1])
		norm_p = (p-self.__mean)/self.__std_dev
		topics = map(int,tokens[2:])
		pred[docid] = (norm_p,topics)
		pred_regular.append((docid,norm_p))
	fd.close()
	return pred,pred_regular

    def rmse(self,truth,pred):
	mean_truth = 0.0

	for doc_id in truth:
	    mean_truth += truth[doc_id]
	mean_truth = mean_truth/len(truth.keys())

	err_total = 0.0

	for (doc_id,p) in pred:
	    err = (p - truth[doc_id])
	    #print  "-->",err,doc_id,p,truth[doc_id],p/truth[doc_id]
	    err_total += err*err

	err_total = err_total/len(pred)

	return math.sqrt(err_total)

    def mean_pred(self,pred):
	"""
	Dada uma lista de predicoes, calcula a media
	"""
	media = 0.0
	for (docid,p) in pred:
	    media = media + p

	return (media/len(pred))

    def std_dev_pred(self,m,pred):
	s = 0.0
	for (docid,p) in pred:
	    s = s + (p-m)*(p-m)
        s = s/len(pred)
	return math.sqrt(s)

    def compare_pred(self,pred1,pred2,truth):
	#quem sao os documentos que possuem 
	#o menor erro na predicao 2
	diff2_lst = {}
	pred1_err_topic = {}
	pred2_err_topic = {}
	conta_docid = {}

	for docid in pred1:
	    if ((docid in pred2) and (docid in truth)):
		(p1,topics1) = pred1[docid]
		(p2,topics2) = pred2[docid]
	        diff1 = abs(truth[docid]-p1)
		diff2 = abs(truth[docid]-p2)

		print docid,diff1,diff2

		for t in topics1:
		    if t in pred1_err_topic:
			pred1_err_topic[t].append((docid,p1))
			pred2_err_topic[t].append((docid,p2))
		    else:
			pred1_err_topic[t] = [(docid,p1)]
			pred2_err_topic[t] = [(docid,p2)]


		if (diff2 < diff1):
		    for t in topics2:
	                if t in diff2_lst:
		             diff2_lst[t].append((docid,p1,p2,diff2,diff1))
		        else:
		            diff2_lst[t] = [(docid,p1,p2,diff2,diff1)]
        print "Num. de docs em cada topico em 1"
	topics1_pred = {}
	for docid in pred1:
	    (p1,topics1) = pred1[docid]
	    for t in topics1:
		if t in topics1_pred:
		    topics1_pred[t] = topics1_pred[t] + 1
		else:
		    topics1_pred[t] = 1

	for t in topics1_pred:  
	     media = self.mean_pred(pred1_err_topic[t])
	     print t,topics1_pred[t],self.rmse(truth,pred1_err_topic[t]),\
		     media,self.std_dev_pred(media,pred1_err_topic[t])
		     
        print "Num. de docs em cada topico em 2"
	topics2_pred = {}
	for docid in pred2:
	    (p2,topics2) = pred2[docid]
	    for t in topics2:
		if t in topics2_pred:
		    topics2_pred[t] = topics2_pred[t] + 1
		else:
		    topics2_pred[t] = 1

	for t in topics2_pred:    
	     media = self.mean_pred(pred2_err_topic[t])
	     print t,topics2_pred[t],self.rmse(truth,pred2_err_topic[t])

        print "Docs em dois com menor erro:"
	for t in diff2_lst:
	    print t,diff2_lst[t],len(diff2_lst[t])
	    #,diff2_lst
	#contar por topico 

    def filtra(self,f,old):
	new = []
	for docid in old:
	    if docid in f:
		(p,topics) = old[docid]
		new.append((docid,p))
        return new

    def compare_results(self,truth,pred_top,pred1,pred2):

	pred1_err_topic = {}
	pred2_err_topic = {}
	pred_err_top = {}
	ndocs_pred = {}

        ts = "%.0f" % time.time()
	rmse_per_topic_file = "rmse_per_topic_" + ts + ".csv"
	rmse_per_file = "rmse.csv"
	fd_rmse = open(rmse_per_file,"a")
	fd =  open(rmse_per_topic_file,"w")
	fd.write("topico,ndocs,rmse top,rmse1,rmse2")

	for docid in pred_top:

	    (p_top,topics_top) = pred_top[docid]

	    for t in topics_top:

	        if (docid in truth):
	            if (t in pred_err_top):
			    pred_err_top[t].append((docid,p_top))
		    else:
			    pred_err_top[t] = [(docid,p_top)]

		    if (docid in pred2):
			(p2,topics2) = pred2[docid]
			if (t in pred2_err_topic):
			    pred2_err_topic[t].append((docid,p2))
			else:
			    pred2_err_topic[t] = [(docid,p2)]

		    if (docid in pred1):
			(p1,topics1) = pred1[docid]
			if (t in pred1_err_topic):
			    pred1_err_topic[t].append((docid,p1))
			else:
			    pred1_err_topic[t] = [(docid,p1)]
			    

		if t in ndocs_pred:
		    ndocs_pred[t] = ndocs_pred[t] + 1
		else:
		    ndocs_pred[t] = 1

        pred1_regular = []
        pred2_regular = []
	pred_top_regular = []
	for t in ndocs_pred:
	    pred1_regular += pred1_err_topic[t]
	    pred2_regular += pred2_err_topic[t]
	    pred_top_regular += pred_err_top[t]

	    err_top = self.rmse(truth,pred_err_top[t])
	    err1 = self.rmse(truth,pred1_err_topic[t])
	    err2 = self.rmse(truth,pred2_err_topic[t])
	    fd.write("%d,%d,%f,%f,%f\n" % (t,ndocs_pred[t],err_top,err1,err2))

        err_top_reg = self.rmse(truth,pred_top_regular)
        err1_reg = self.rmse(truth,pred1_regular)
        err2_reg = self.rmse(truth,pred2_regular)
	fd_rmse.write("%f,%f,%f\n" % (err_top_reg,err1_reg,err2_reg))
        fd.close()
	fd_rmse.close()

if __name__ == "__main__":
    eva = Evaluation("options.conf")
    #truth_file = sys.argv[1]
    top_file_lst = []
    th_file_lst = []
    all_file_lst = []

    re_top = re.compile("pred\_top[\_\w]*\.out")
    re_all = re.compile("pred\_all[\_\w]*\.out")
    re_th = re.compile("pred\_threshold[\_\w]*\.out")
    for subdir, dirs, files in os.walk('.'):
	for f in files:
	    if (subdir == '.'):
		if (re_top.match(f)):
		   top_file_lst.append(f)
		elif (re_th.match(f)):
		    th_file_lst.append(f)
		elif (re_all.match(f)):
		    all_file_lst.append(f)
    #print top_file_lst
    #print th_file_lst
    #print all_file_lst
    truth = eva.read_truth("../ls-deals.csv")
    #truth = eva.read_truth(truth_file)
    #pred = eva.read_pred("pred.out")
    #pred_file = sys.argv[2]

    nfiles = min(len(top_file_lst),len(th_file_lst),len(all_file_lst))
    for i in xrange(nfiles):
        pred_top_file =  top_file_lst[i]
        pred_all_file =  all_file_lst[i]
        pred_th_file =  th_file_lst[i]

        pred_top,pred_top_regular = eva.read_pred_topic(pred_top_file)
        pred_all,pred_all_regular = eva.read_pred_topic(pred_all_file)
        pred_th,pred_th_regular = eva.read_pred_topic(pred_th_file)

        eva.compare_results(truth,pred_top,pred_all,pred_th)

    #pred_all_file = "pred_all_30.out"

    #pred_th_file = "pred_threshold_30.out"

    #cria tabela de de todos os tres tipos estrategias
    #print "top vs all"
    #eva.compare_pred(pred_top,pred_all,truth)
    #print "top vs th"
    #eva.compare_pred(pred_top,pred_th,truth)
    #print "all vc th"
    #eva.compare_pred(pred_all,pred_th,truth)

    #pred_all_new = eva.filtra(pred_top,pred_all)

    #print eva.rmse(truth,pred_all_new)
    #print eva.rmse(truth,pred_th_regular)
    #print eva.rmse(truth,pred_top_regular)
