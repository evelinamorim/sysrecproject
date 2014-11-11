

class Deal:

    def __init__(self,doc_id=-1,ftrs=[],ftrs_txt=[],target=-1):
	self.__doc_id = doc_id
	self.__ftrs = ftrs
	self.__ftrs_txt = ftrs_txt
	#deal size
	self.__target = target

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
