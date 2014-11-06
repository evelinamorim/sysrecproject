import sys
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
    return k
