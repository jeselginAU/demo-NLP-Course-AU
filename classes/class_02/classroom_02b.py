#!/usr/bin/env python
# coding: utf-8

# # Document vectors
# The first thing we're going to do, as usual, is begin by importing libraries and modules we're going to use today. We're introducing a new library, called ```datasets```, which is part of the ```huggingface``` universe. 
# 
# ```datasets``` provides easy access to a wide range of example datasets which are widely-known in the NLP world, it's worth spending some time looking around to see what you can find. For example, here are a collection of [multilabel classification datasets](https://huggingface.co/datasets?task_ids=task_ids:multi-class-classification&sort=downloads).
# 
# We'll be working with the ```huggingface``` ecosystem more and more as we progress this semester.

# In[1]:


# data processing
import pandas as pd
import numpy as np

# huggingface datasets
from datasets import load_dataset

# scikit learn tools
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

# plotting tools
import matplotlib.pyplot as plt


# ## Load data
# We're going to be working with actual text data data, specifically a subset of the well-known [GLUE Benchmarks](https://gluebenchmark.com/). These benchmarks are regularly used to test how well certain models perform across a range of different language tasks. We'll work today specifically with the Stanford Sentiment Treebank 2 (SST2) - you can learn more [here](https://huggingface.co/datasets/glue) and [here](https://nlp.stanford.edu/sentiment/index.html).
# 
# The dataset we get back is a complex, hierarchical object with lots of different features. I recommend that you dig around a little and see what it contains. For today, we're going to work with only the training dataset right now, and we're going to split it into sentences and labels.

# In[2]:


# load the sst2 dataset
dataset = load_dataset("glue", "sst2")
# select the train split
train_data = dataset["train"]
X = train_data["sentence"]
y = train_data["label"]


# Let's split the data into a training and a test set. We will later train a simple classifier to start looking at what one can do with vector representations of text, that's why we need a set of documents that are left aside. For now, let's simply focus on the training set to estimate our document-term model.

# In[3]:


import random
train_idx = random.sample(range(len(X)), k=int(len(X)*.7)) # we are sampling 70% as training set
train_X, test_X, train_y, test_y = [], [], [], []
for i in train_idx:
    train_X.append(X[i])
    train_y.append(y[i])
for i in set(range(len(X))) - set(train_idx):
    test_X.append(X[i])
    test_y.append(y[i])


# In[4]:


list(zip(train_X[:10], train_y[:10]))


# In[5]:


print('Number of training examples: ', len(train_X))
print('Number of test examples: ', len(test_X))


# 
# ## Create document representations
# We're going to work with a bag-of-words model (like the ones we talked about in class), which we can create quite simply using the ```CountVectorizer()``` class available via ```scikit-learn```. You can read more about the default parameters of the vectorizer [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html).
# 
# After we initialize the vectorizer, we first _fit_ this vectorizer to our data (the model learns parameters such as which words to include in the vocabulary, based on the statistics of the text and the parameters passed to  `CountVectorizer`) and then _transform_ the original data into the bag-of-words representation.
# 
# Let's start by fitting a model where default constraints are placed on vocabulary size.

# In[6]:


simple_vectorizer = CountVectorizer()
X_vect = simple_vectorizer.fit_transform(train_X)


# This is the number of words the vectorizer uses as features (i.e., words that are *not* excluded because too frequent, or too infrequent)

# In[7]:


len(simple_vectorizer.vocabulary_)


# In[8]:


print(X_vect.shape)
print(X_vect.toarray())


# As you can see, the resulting matrix has dimensions `[n_documents, n_words]`.
# Note that there is a simple way to get a term-term matrix (in how many documents two words co-occur) by computing the dot product of the term-document matrix and its transpose.

# In[12]:


np.dot(X_vect.T, X_vect).toarray() # the diagonal essentially indicates how often a term occurs overall.


# What happens to dimensionality if manipulate input parameters, e.g., `min_df`? Try to play with `CountVectorizer` parameters to get familiar with the function.

# ### Dimensionality reduction
# Our current matrix is fairly sparse. Could we apply what we have learned during the lecture to convert it to a dense and more compact matrix? Let's apply the `SVD` algorithm we discussed in class.

# In[20]:


svd = TruncatedSVD(n_components=500)
svd.fit(X_vect)
X_svd = svd.transform(X_vect)


# How does our vector space look like?

# In[21]:


X_svd


# ### Classifying sentiment

# Congratulations! You have created your first document representation. 
# 
# We will dive deeper into classification in the coming weeks, but to demonstrate what we can do with these representations, let's go through an example.
# 
# As we saw earlier, our documents have labels indicating the sentiment of each of the document. Can we predict sentiment on the basis of bag of words representations of our documents?
# Let's use a simple `scikit-learn` classifier to learn to predict sentiment from text. We will learn more about this later on, for now all you need to know is that the classifier estimates a relation between input and output such that it is able to predict the output (in this case, the sentiment of the sentence, which is `0` for negative sentences, `1` for positive) from the input.
# 
# We will use a `LogisticRegression` classifier (not necessarily best, but one the fastest), but you can experiment with multiple classifiers (e.g., https://scikit-learn.org/stable/modules/svm.html).

# In[16]:


classifier = LogisticRegression(max_iter=2000).fit(X_vect, train_y)


# Let's transform the test data, which we need for evaluation.

# In[17]:


X_vect_test = simple_vectorizer.transform(test_X)


# And finally, let's compute how often the model predictions match the true labels.

# In[18]:


print('Model accuracy: ', np.mean(classifier.predict(X_vect_test) == test_y))


# That's pretty good: let's take a look at a couple of examples.

# In[20]:


list(zip(test_X, classifier.predict(X_vect_test)))


# ### Some optional tasks
# - Does performance change if we use a `TfidfVectorizer`?
# - Can you write your own version of `CountVectorizer()`? In other words, a function that takes a corpus of documents and creates a bag-of-words representation for every document?
# - What about `TfidfVectorizer()`? Look over the formulae in the slides from Tuesday.

# 
