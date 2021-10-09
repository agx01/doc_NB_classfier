# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:45 2021

@author: Arijit Ganguly
"""
import os

class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        print(self.labels)

if __name__ == "__main__":
    doc_class = DocClassifier()
