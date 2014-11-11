import sys
import time
import datetime
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
