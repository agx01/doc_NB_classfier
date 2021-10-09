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