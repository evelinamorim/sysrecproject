

class Deal:

    def __init__(self,doc_id=-1,ftrs=[],target=-1):
	self.__doc_id = doc_id
	self.__ftrs = ftrs
	self.__ftrs_txt = []
	#deal size
	self.__target = target
	self.__pred = 0.0
	#valor intermediario de predicao
	self.__sigma = -1
	self.__topic_prob = {}

    def set_prob(self,topic,prob):
	self.__topic_prob[topic] = prob

    def get_prob(self,topic):
	return self.__topic_prob[topic]
    def get_topic_prob(self):
	return self.__topic_prob

    def get_ftrs(self):
	return self.__ftrs

    def get_docid(self):
	return self.__doc_id

    def get_ftrs_dict(self,header):
	"""
	Tranforma as features em forma de dicionario
	"""
	i = 0
	dict_ftrs = {}
	for f in self.__ftrs:
	    dict_ftrs[header[i]] = f
	    i = i+1
	return dict_ftrs

    def get_ftrs_txt(self):
	return self.__ftrs_txt

    def get_target(self):
	return self.__target

    def set_pred(self,pred):
	self.__pred = pred

    def get_pred(self):
	return self.__pred

    def set_sigma(self,s):
	self.__sigma = s

    def get_sigma(self):
	return self.__sigma
