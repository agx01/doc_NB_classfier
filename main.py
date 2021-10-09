# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:45 2021

@author: Arijit Ganguly
"""
import os

class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
    
    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            #print(f.read())
            print(file_path)
    
    def print_filenames(self):
        #iterate through the sub directories
        for label in self.labels:
            k = 0
            #Iterate through the files in sub directories
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                self.read_file(file_path)
                k += 1
        
        print("Number of Files"+str(k))
        
if __name__ == "__main__":
    doc_class = DocClassifier()
    doc_class.print_filenames()