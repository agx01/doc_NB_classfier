# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:59 2021

@author: Arijit Ganguly
"""

import os

# Get labels/classes of the documents
labels = [x for x in os.listdir("data/20_newsgroups")]

print(labels)

path = "data\\20_newsgroups\\"
def read_file(file_path):
    with open(file_path, 'r') as f:
        #print(f.read())
        print(file_path)

#iterate through the sub directories
for label in labels:
    k = 0
    #Iterate through the files in sub directories
    for file in os.listdir(f"{path}\{label}"):
        file_path = f"{path}\{label}\{file}"
        read_file(file_path)
        k += 1

print("Number of Files"+str(k))

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
from sklearn.feature_extraction.text import CountVectorizer

class GlobalDict:
    
    def __init__(self):
        self.global_dict = []
        self.file_vec_list = []
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        del self.global_dict
        del self.file_vec_list
        
    def add_word2dict(self, word):
        self.global_dict.append(word)
    
    def add_file_vec(self, vec):
        self.global_dict.append(vec)
    
    def get_file_vec(self):
        return self.file_vec_list                

class Document:
    
    def __init__(self, label, file_path):
        self.label = label
        self.vec, self.doc_len = self.preprocessing(file_path)
    
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
    
    def remove_stop_words(self, content):
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
        return " ".join([word for word in str(content).split() if word not in self.stop_words])
    
    def preprocessing(self, file_path):
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
        content = self.remove_stop_words(content)
        
        #Creating the vector for each document
        vec = self.create_doc_vector(content)
        
        doc_length = len(vec)
        
        return vec, doc_length
    
    def create_doc_vector(self, content):
        word_list = [word for word in str(content).split()]
        vec = {}
        for word in word_list:
            #self.global_dict.add_word2dict(word)
            if word in vec.keys():
                vec[word] = vec[word] + 1
            else:
                vec[word] = 1
        return vec

class Preprocessor:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
        self.stop_words = set(stopwords.words('english'))
        self.global_dict = GlobalDict()
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        del self.labels
        del self.path
        del self.stop_words
    
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
        
    def remove_stop_words(self, content):
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
        return " ".join([word for word in str(content).split() if word not in self.stop_words])
    
    def preprocessing(self, file_path):
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
         
        #Remove words and digits like game47,g4me
        #content = re.sub('W*dw*', '', content)
        
        #Remove all numbers
        content = re.sub(r'[0-9]+', '', content)
                
        #Remove stopwords
        content = self.remove_stop_words(content)
        
        #Creating the vector for each document
        vec = self.create_doc_vector(content)
        #self.global_dict.add_file_vec(vec)

        return content, vec
    
    def create_doc_vector(self, content):
        word_list = [word for word in str(content).split()]
        vec = {}
        for word in word_list:
            #self.global_dict.add_word2dict(word)
            if word in vec.keys():
                vec[word] = vec[word] + 1
            else:
                vec[word] = 1
        return vec
        
class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
        self.training_data = pd.DataFrame()
        self.testing_data = pd.DataFrame()
        self.file_per_class = {}
    
    def Classify(self):
        if os.path.exists('processed_data.csv'):
            df = pd.read_csv('processed_data.csv')
        else:
            k=0
            preprocessor = Preprocessor()
            df = pd.DataFrame()
            for label in self.labels:
                for file in os.listdir(f"{self.path}\{label}"):
                    file_path = f"{self.path}\{label}\{file}"
                    content, vec = preprocessor.preprocessing(file_path)
                    vec['category_class'] = label
                    new_df = pd.DataFrame(vec, index=[0])
                    preprocessor.global_dict.add_file_vec(vec)
                    #df = pd.concat([df,new_df], axis=0, ignore_index=True)
                    k+=1
                    print(f"Document {k} processed for class {label}")
            print(f"Total Documents processed: {k}")
            print(preprocessor.global_dict.get_file_vec())
            #print(df.head())
            #self.count_files()
            #df = pd.DataFrame(self.file_per_class, columns = self.file_per_class.keys(), index=[0])
            #df = df.fillna(0)
            #df.to_csv('processed_data.csv')
        print(df)
                                
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