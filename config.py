

class Config:
    """
    Classe que le configuracoes de um arquivo e coloca em um dicionario
    O arquivo deve ter o formato:
    NOME_VALOR=VALOR1,VALOR2,...,VALOR3
    """
    def __init__(self,config_file):
	self.__config_file = config_file
	self.__config_values = {}

    def read(self):
	"""
	Read configuration file and append for each option a lista 
	of values for that option
	"""
	fd_config = open(self.__config_file,"r")

	for line in fd_config:
	    lst_line = line.split("=")
	    if (len(lst_line)>=2):
	        name = lst_line[0]

	        lst_values = lst_line[1].split(",")
	        nvalues = len(lst_values)
	        lst_values[nvalues-1] = lst_values[nvalues-1].replace("\n","")
	        self.__config_values[name] = lst_values

	fd_config.close()

    def get_value(self,option):
	"""
	Given a option read from configuration file, it returns 
	the values for that option
	"""
	return self.__config_values[option]
    
