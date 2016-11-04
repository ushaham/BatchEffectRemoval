'''
Created on Jul 29, 2016

@author: Kelly P. Stanton
'''
import sys,os
def we_are_frozen():
    # All of the modules are built-in to the interpreter, e.g., by py2exe
    return hasattr(sys, "frozen")
#OS independent way to find the root
def DeepLearningRoot():
    if we_are_frozen():
        return os.path.join(os.path.dirname(sys.executable),os.pardir, os.pardir)
    return os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)