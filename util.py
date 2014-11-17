import sys
import time
import datetime
import re

open_quote = re.compile("\"[\w]+")
close_quote = re.compile("[\w\.\-]+\"")

def turn2type(value,t):
    """
    Given a value and type, both in string format, then 
    turn value into type given by t
    """
    k = None
    if (t=="str"):
	k = value
    if (t=="int"):
	k = int(value)
    if (t=="float"):
	k = float(value)
    if (t=="date"):
	k = time.mktime(datetime.datetime.strptime(value, "%m/%d/%Y").timetuple())
    return k

def tokenize_features(line):
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
