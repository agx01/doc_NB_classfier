# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:45 2021

@author: Arijit Ganguly
"""
import os
import re
import string
from nltk.corpus import stopwords
import pandas as pd

class GlobalDict:
    
    def __init__(self, labels):
        self.global_dict = []
        self.stop_words = set(stopwords.words('english'))
        self.words_per_class = {}
        for label in labels:
            self.words_per_class[label] = 0
        self.prior_list = []
        self.total_words = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        del self.global_dict
        del self.file_vec_list
        
    def add_word2dict(self, word, label):
        if word not in self.global_dict:
            self.global_dict.append(word)
        self.increase_words_per_class(label)
    
    def increase_words_per_class(self, label):
            self.words_per_class[label] += 1
            #increase the total words list
            self.total_words += 1
    
    def get_file_vec(self):
        return self.file_vec_list   

    def get_label_wordcount(self, label):
        return self.words_per_class[label]
    

class Document:
    
    def __init__(self, label, file_path, global_dict):
        self.label = label
        self.vec, self.doc_len = self.preprocessing(file_path, global_dict)
        self.prob_list = {}
        
    def calculate_prob(self, global_dict):
        for word in self.vec.keys():
            prob = self.vec[word]/global_dict.get_label_wordcount(self.label)
    
    def read_file(self, file_path):
        """
        This function is to read the file specified in the file path 
        and return the content of the file.

        Parameters
        ----------
        file_path : String
            The file path suffixed by the file name that needs to be read

        Returns
        -------
        content : String
            raw content of the file in string format

        """
        with open(file_path, 'r') as f:
            content = f.read()
        return content
    
    def remove_stop_words(self, content, stop_words):
        """
        Function to remove the stop words from the content.
        For example: the,a, an
        
        Process followed by the code:
            1. Split the content into words
            2. Loop through each word of the content and check if it exists
                it exists in the list of stop words
            3. Add into a new list if the word does not exist in the stop
                words list
            4. join the list of words to form a new string and return it

        Parameters
        ----------
        stop_words : List[String]
            List of stop words that need to be removed from content of file    
        
        content : String
            The content that needs to be filtered for stop words

        Returns
        -------
        String
            Content that has been filtered out for any stop words

        """
        return " ".join([word for word in str(content).split() if word not in stop_words])
    
    def preprocessing(self, file_path, global_dict):
        """
        Pre-Process Steps for each file
            1. Get the content from the file
            2. Lower the capital letters in the content
            3. Replace all the punctuations with a space
            4. Remove all the words with letter/number combination
            5. Remove all the stop words in the content

        Parameters
        ----------
        file_path : String
            File path for which preprocessing is required

        Returns
        -------
        content : String
            Preprocessed content of the file.

        """
        #Get content from the file
        content = self.read_file(file_path)
        
        #Lower case the entire content
        content = content.lower()
         
        #Replace the punctuation
        content = re.sub('[%s]' % re.escape(string.punctuation), ' ', content)
        
        #Remove all numbers
        content = re.sub(r'[0-9]+', '', content)
                
        #Remove stopwords
        content = self.remove_stop_words(content, global_dict.stop_words)
        
        #Creating the vector for each document
        vec = self.create_doc_vector(content, global_dict)
        
        doc_length = len(vec)
        
        return vec, doc_length
    
    def create_doc_vector(self, content, global_dict):
        word_list = [word for word in str(content).split()]
        vec = {}
        for word in word_list:
            global_dict.add_word2dict(word, self.label)
            if word in vec.keys():
                vec[word] = vec[word] + 1
            else:
                vec[word] = 1
        return vec

        
class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
        self.doc_list = []
        self.global_dict = GlobalDict(self.labels)
    
    def fit(self):
        for doc in self.doc_list:
            doc.calculate_prob()
    
    def Classify(self):
        k=0
        for label in self.labels:
            j = 0
            files_num = len(os.listdir(f"{self.path}\{label}"))
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                doc = Document(label, file_path, self.global_dict)
                self.doc_list.append(doc)
                j += 1
                if j == int(files_num/2):
                    break
                k+=1
                print(f"Document {k} processed for class {label}")
        print(f"Total Documents processed: {k}")
    
    def print_doclist(self):
        i = 1
        k = len(self.doc_list)
        for doc in self.doc_list:
            print(f"Document {i} / {k}")
            print(f"Document for class: {doc.label}")
            print(f"Document Length: {doc.doc_len}")
            print(f"Document Vector: {doc.vec}")
            i+=1
                                
    def count_files(self):
        for label in self.labels:
            file_list = os.listdir(f"{self.path}\{label}")
            self.file_per_class[label] = len(file_list)
        print(self.file_per_class)
        
        
if __name__ == "__main__":
    doc_class = DocClassifier()
    #doc_class.print_filenames()
    #doc_class.prepare_data()
    doc_class.Classify()