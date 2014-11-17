import math
import config
import util
import sys

class Evaluation:

    def __init__(self,config_file):

	self.__config_obj = config.Config(config_file)
	self.__config_obj.read()
	self.__config_file = config_file

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
	for l in lines:
	    tokens = util.tokenize_features(l)
	    truth[docid] = util.turn2type(tokens[h],target_type)
	    docid = docid+1

	fd.close()
	return truth

    def read_pred(self,file_name):

	fd = open(file_name,"r")
	pred = []

	for l in fd:
	    tokens = l.split(",")
	    if (len(tokens)>=2):
		docid = int(tokens[0])
		p = float(tokens[1])
		pred.append((docid,p))
	fd.close()
	return pred

    def rmse(self,truth,pred):
	mean_truth = 0.0

	for doc_id in truth:
	    mean_truth += truth[doc_id]
	mean_truth = mean_truth/len(truth.keys())

	err_total = 0.0
	den = 0.0

	for (doc_id,p) in pred:
	    err = (p - truth[doc_id])
	    print  "-->",err,doc_id,p,truth[doc_id],p/truth[doc_id]
	    err_total += err*err
	    err_m = (mean_truth-truth[doc_id])
	    den += err_m*err_m

	err_total = err_total/den

	return math.sqrt(err_total)

if __name__ == "__main__":
    eva = Evaluation("options.conf")
    truth_file = sys.argv[1]
    #truth = eva.read_truth("../daily-deals-data/ls-deals.csv")
    truth = eva.read_truth(truth_file)
    #pred = eva.read_pred("pred.out")
    pred_file = sys.argv[2]
    pred = eva.read_pred(pred_file)

    print eva.rmse(truth,pred)
