#!/usr/bin/env python


import sys

def performance():
    sumv = 0
    for line in sys.stdin:
        line = line.strip()
        if line:
            lines = line.split()
            sumv += float(lines[3])
    print sumv
    print "expected at least", sumv/5.0
performance()
