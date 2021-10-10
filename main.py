# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:45 2021

@author: Arijit Ganguly
"""
import os
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
    
    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        return content
        
    def print_filenames(self):
        k=0
        #iterate through the sub directories
        for label in self.labels:
            #Iterate through the files in sub directories
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                self.read_file(file_path)
                k += 1
        print("Number of Files"+str(k))
    
    def remove_stop_words(self, stop_words, content):
        return " ".join([word for word in str(content).split() if word not in stop_words])
    
    def prepare_data(self):
        #Initializing the stopwords
        stop_words = set(stopwords.words('english'))
        
        k =0
        #iterate through the sub directories
        for label in self.labels:
            #Iterate through the files in sub directories
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                content = self.read_file(file_path)
                
                #Lower case the entire content
                content = content.lower()
                
                #Remove the punctuation
                content = re.sub('[%s]' % re.escape(string.punctuation), '', content)
                
                #Remove words and digits like game47,g4me
                content = re.sub('W*dw*', '', content)
                
                #Remove stopwords
                content = self.remove_stop_words(stop_words, content)                  
                k += 1
                print("Document %d preprocessing complete", k)    
                
        
        print("Documents prepared: "+str(k))
        
if __name__ == "__main__":
    doc_class = DocClassifier()
    #doc_class.print_filenames()
    doc_class.prepare_data()