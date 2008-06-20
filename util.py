def factorial(n):              
    """
    Returns n!
    """
    p = 1
    for j in range(1,n+1):
        p *= j
    return p

def chooses(n, r):
    """
    Return n chooses r, or n!/r!/(n-r)!
    """
    return factorial(n)/factorial(r)/factorial(n-r)

def print_info(s):
    import sys
    sys.stderr.write("INFO: %s\n" % s)
    sys.stderr.flush()
