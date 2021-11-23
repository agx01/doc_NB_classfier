# Document Classifier using Naive Bayes Alorithm
Use the data provided by the website to classify 1000 documents into 20 newsgroup categories.

**Data :**
https://www.cs.cmu.edu/afs/cs/project/theo-11/www/naive-bayes.html

## Stop words setup required
The code utilizes a **nltk** library for importing of the stopwords.
To setup, please use the below mentioned code in the Python console.
```
import nltk
nltk.download('stopwords')
```

## Problem Statement
The data set contains 19997 documents which belong to 20 different classes. We need to train our naïve bayes algorithm on 50% of the data set, i.e., 9998 documents(approximately 500 documents from each class) and use the remaining 50% as the testing set and predict the predicted classes (newsgroups) to calculate the accuracy of the naïve bayes algorithm.

## Strategy
The frequency of words in the document can be used to predict the class of the document.
We calculate the number of words in each class by pre-processing the document and then we calculate the probability of each word in the class. We also calculate prior value for each class.
Then using the Bayes theorem, we calculate the predicted class values.
We use the above-mentioned formula in our algorithm to calculate the probability of a test document belonging to a class by calculating the probabilities for each class and the maximum value among that is our predicted value.
