#!/usr/bin/env python
from . import sparse4d
import sys

for fn in sys.argv[1:]:
    try:
        i = sparse4d.Sparse4D(fn)
        print("--- %s ---" % fn)
        
        for key, val in i.header.items():
            print('%20s: %s' % (key, val))
            
        print()
    
    except:
        print("--- Couldn't open '%s' ---" % fn)
    
    
