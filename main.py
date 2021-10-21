# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:38:45 2021

@author: Arijit Ganguly
"""
import os
import re
import string
from nltk.corpus import stopwords
from sklearn import metrics
import numpy as np

class GlobalDict:
    
    def __init__(self, labels):
        self.stop_words = stopwords.words('english')
        self.stop_words.append('aa')
        self.stop_words.append('aaa')
        self.stop_words.append('aaaahhh')
        self.stop_words.append('aab')
        self.stop_words.append('aachen')
        self.stop_words.append('xref')
        self.words_per_class = {}
        self.class_vec_list = {}
        self.prob_list = {}
        for label in labels:
            self.words_per_class[label] = 0
            self.class_vec_list[label] = {}
            self.prob_list = {}
        self.total_words = 0
        self.alpha = 1
        #self.word_freq_per_class = {}
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        del self.global_dict
        del self.file_vec_list
        
    def add_word2dict(self, doc_vec, label):
        class_vec = self.class_vec_list[label]
        for word in doc_vec.keys():
            if word not in class_vec.keys():
                class_vec[word] = doc_vec[word]
            else:
                class_vec[word] += doc_vec[word]
        self.class_vec_list[label] = class_vec

    def calculate_words_per_class(self, labels):
        for label in labels:
            class_vec = self.class_vec_list[label]
            for word in class_vec:
                self.words_per_class[label] += class_vec[word]
    
    def get_class_vec(self):
        return self.class_vec_list   

    def get_label_wordcount(self, label):
        return self.words_per_class[label]
    
    def calculate_prob(self, labels, alpha = 0, n = 1):
        self.calculate_words_per_class(labels)
        for label in labels:
            class_vec = self.class_vec_list[label]
            prob_vec = {}
            for word in class_vec:
                prob = (class_vec[word]+alpha)/(self.words_per_class[label]+n*alpha)
                prob_vec[word] = prob
            self.prob_list[label] = prob_vec
        
class Document:
    
    def __init__(self, label, file_path, global_dict):
        self.label = label
        self.vec, self.doc_len = self.preprocessing(file_path, global_dict)
        #self.prob_list = {}
    
    
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
        
        #Remove all the single letters
        content = re.sub(r'(?:^| )\w(?:$| )', '', content).strip()
                
        #Remove stopwords
        content = self.remove_stop_words(content, global_dict.stop_words)
        
        #Creating the vector for each document
        vec = self.create_doc_vector(content)
        
        doc_length = len(vec)
        
        return vec, doc_length
    
    def create_doc_vector(self, content):
        word_list = [word for word in str(content).split()]
        vec = {}
        for word in word_list:
            if word in vec.keys():
                vec[word] = vec[word] + 1
            else:
                vec[word] = 1
        return vec

        
class DocClassifier:
    
    def __init__(self, train_test_split=0.5):
        self.labels = [x for x in os.listdir("data/20_newsgroups")]
        self.path = "data\\20_newsgroups\\"
        self.global_dict = GlobalDict(self.labels)
        self.prior_list = {}
        self.train_test_split = train_test_split
        self.file_per_class = {}
        
    def get_accuracy(self):
        Y_act = []
        Y_pred = []
        Y_log_pred = []
        k = 0
        for label in self.labels:
            file_list = os.listdir(f"{self.path}\{label}")
            i = int(len(file_list)*self.train_test_split) + 1
            file_list = file_list[i:]
            for file in file_list:
                print(f"Testing on Document: {k}")
                file_path = f"{self.path}\{label}\{file}"
                doc = Document(label, file_path, self.global_dict)
                Y_act.append(label)
                Y_pred.append(self.predict(doc.vec))
                Y_log_pred.append(self.predict_with_log(doc.vec))
                k+=1
        print("Accuracy of the algorithm on the existing dataset is:")
        print(str(metrics.accuracy_score(y_true=Y_act, y_pred=Y_pred)*100)+"%")
        print("Accuracy of the algorithm using log in the existing data is :")
        print(str(metrics.accuracy_score(y_true=Y_act, y_pred=Y_log_pred)*100)+"%")
        
    def get_multiNBaccuracy(self):
        Y_act = []
        Y_pred = []
        k = 0
        for label in self.labels:
            file_list = os.listdir(f"{self.path}\{label}")
            i = int(len(file_list)*self.train_test_split) + 1
            file_list = file_list[i:]
            for file in file_list:
                print(f"Testing on Document: {k}")
                file_path = f"{self.path}\{label}\{file}"
                doc = Document(label, file_path, self.global_dict)
                Y_act.append(label)
                Y_pred.append(self.multinomial_NB_predict(doc.vec))
                k+=1
        print("Accuracy of the algorithm on the existing dataset is:")
        print(str(metrics.accuracy_score(y_true=Y_act, y_pred=Y_pred)*100)+"%")
    
    def multinomial_NB_predict(self, doc_vec):
        word_prob_list = {}
        n_list = []
        missing_word_freq = {}
        for label in self.labels:
            n = 0
            for word in doc_vec:
                class_prob_list = self.global_dict.prob_list[label]
                if word not in class_prob_list:
                    n += 1
                    if word in missing_word_freq.keys():
                        missing_word_freq[word] += 1
                    else:
                        missing_word_freq[word] = 1
                    n_list.append(n)
        
        i = 0
        for label in self.labels:
            prior = self.prior_list[label]
            prob = 1
            self.global_dict.calculate_prob(labels = self.labels, n = n_list[i], alpha = self.global_dict.alpha)
            for word in doc_vec:
                class_prob_list = self.global_dict.prob_list[label]
                if word in class_prob_list:
                    prob *= np.log(pow(class_prob_list[word], doc_vec[word]))
                else:
                    prob *= np.log(pow((missing_word_freq[word]/self.global_dict.words_per_class[label]), doc_vec[word]))
            
            prob *= prior
            word_prob_list[label] = prob
        Y_pred = max(word_prob_list, key=word_prob_list.get)
        return Y_pred
    
    
    def predict_with_log(self, doc_vec):
        word_prob_list = {}
        for label in self.labels:
            prior = self.prior_list[label]
            prob = 1
            for word in doc_vec:
                class_prob_list = self.global_dict.prob_list[label]
                if word in class_prob_list:
                    prob *= np.log(pow(class_prob_list[word], doc_vec[word]))
            prob *= prior
            word_prob_list[label] = prob
        Y_pred = max(word_prob_list, key=word_prob_list.get)
        return Y_pred
    
    def predict(self, doc_vec):
        word_prob_list = {}
        for label in self.labels:
            prior = self.prior_list[label]
            prob = 1
            for word in doc_vec:
                class_prob_list = self.global_dict.prob_list[label]
                if word in class_prob_list:
                    prob *= pow(class_prob_list[word], doc_vec[word])
            prob *= prior
            word_prob_list[label] = prob
        Y_pred = max(word_prob_list, key=word_prob_list.get)
        return Y_pred
        
    def test_predict(self, label):
        file_list = os.listdir(f"{self.path}\{label}")
        files_num = len(file_list)
        i = int(files_num*self.train_test_split) + 1
        file = file_list[i]
        file_path = f"{self.path}\{label}\{file}"
        test_doc = Document(label, file_path, self.global_dict)
        word_prob_list = {}
        doc_vec = test_doc.vec
        for label in self.labels:
            prior = self.prior_list[label]
            prob = 1
            for word in doc_vec:
                class_prob_list = self.global_dict.prob_list[label]
                if word in class_prob_list:
                    prob *= class_prob_list[word]
            prob *= prior
            word_prob_list[label] = prob
        for key in word_prob_list.keys():
            print(f"{key}:{word_prob_list[key]}")
        
        print("Predicted Class is:")
        print(max(word_prob_list, key=word_prob_list.get))
        
    def calculate_prior(self, total_records):
        for label in self.labels:
            file_num = len(os.listdir(f"{self.path}\{label}"))
            prior_label = (file_num*self.train_test_split)/total_records
            self.prior_list[label] = prior_label
        
    def fit(self):
        k=0
        for label in self.labels:
            j = 0
            files_num = len(os.listdir(f"{self.path}\{label}"))
            for file in os.listdir(f"{self.path}\{label}"):
                file_path = f"{self.path}\{label}\{file}"
                doc = Document(label, file_path, self.global_dict)
                #self.doc_list.append(doc)
                self.global_dict.add_word2dict(doc.vec, label)
                j += 1
                if j == int(files_num*self.train_test_split):
                    break
                k+=1
                print(f"Document {k} processed for class {label}")
        self.calculate_prior(k)
        self.global_dict.calculate_prob(labels=self.labels)
        
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
            print(f"{label}:{len(file_list)}")
        
        
if __name__ == "__main__":
    doc_class = DocClassifier(train_test_split=0.5)
    doc_class.count_files()
    doc_class.fit()
    doc_class.get_accuracy()
    #doc_class.count_files()
    #doc_class.get_multiNBaccuracy()
    """
    i = 0
    print("Please select a label class to test function on:")
    for label in doc_class.labels:
        print(f"{i}. {label}")
        i += 1
    selection = int(input("Selection:"))
    doc_class.test_predict(doc_class.labels[selection])
    """