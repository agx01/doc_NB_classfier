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

class Preprocessor:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
        self.stop_words = set(stopwords.words('english'))
    
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
        content = re.sub('W*dw*', '', content)
                
        #Remove stopwords
        content = self.remove_stop_words(self.stop_words, content)
         
        return content
    
    def prepare_data(self):
        """
        Function to pre-process the data mentioned in the files for each 
        class/label to be able to convert the content into frequency vectors
        
        Pre-Process Steps for each file:
            1. Lower any capital letters
            2. Replace all punctuations with a space
            3. Remove all words with letter/number combo
            4. Remove all the stop words in the content
            
        Returns
        -------
        None.

        """
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
                
                #Replace the punctuation
                content = re.sub('[%s]' % re.escape(string.punctuation), ' ', content)
                
                #Remove words and digits like game47,g4me
                content = re.sub('W*dw*', '', content)
                
                #Remove stopwords
                content = self.remove_stop_words(stop_words, content)                  
                k += 1
                print(f"Document {k} preprocessing complete")    
        print("Documents prepared: "+str(k))

class DocClassifier:
    
    def __init__(self):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
         
    def create_doc_vector(self, content):
        pass
    
    def Classify(self):
        preprocessor = Preprocessor()
        
        for label in self.labels:
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                preprocessor.preprocessing(file_path)
                                
    def count_files(self):
        file_per_class = {}
        for label in self.labels:
            file_list = os.listdir(f"{self.path}\{label}")
            file_per_class[label] = len(file_list)
        print(file_per_class)
        
if __name__ == "__main__":
    doc_class = DocClassifier()
    #doc_class.print_filenames()
    #doc_class.prepare_data()
    doc_class.count_files()