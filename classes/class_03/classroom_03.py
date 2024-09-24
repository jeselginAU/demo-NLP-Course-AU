#!/usr/bin/env python
# coding: utf-8

# # Classroom 3 - Working with word embeddings

# So far we've seen a couple of key Python libraries for doing specific tasks in NLP. For example, ```scikit-learn``` provides a whole host of fundamental machine learning algortithms; ```spaCy``` allows us to do robust linguistic analysis; ```huggingface``` is the place to go for pretrained models (more on that in coming weeks); ```pytorch``` is the best framework for building complex deep learning models.
# 
# Today, we're going to meet ```gensim``` which is the best way to work with (static) word embeddings like word2vec. You can find the documentation [here](https://radimrehurek.com/gensim/).

# In[ ]:


import gensim
import gensim.downloader
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


# ## Choose a language
# 
# I've downloaded a number of pretrained word2vec models for different languages. Feel free to experiment with a couple (or with other models, if you want to download more: you can also download FastText embeddings: https://fasttext.cc/), but make sure to use different variable names for the models.
# 
# NB: The English embeddings are 300d; all other word2vec models here are 100d. Notice also that different word2vec models are loaded in different ways. This is due to way that they were saved after training - the saved formats are not consistently the same.
# 
# **Note**: depending on where your notebook is located, you may need to change the paths!

# In[ ]:


# Danish embeddings https://korpus.dsl.dk/resources/details/word2vec.html
#model = gensim.models.KeyedVectors.load_word2vec_format("models/danish.bin", binary=True)

# Polish embeddings https://github.com/sdadas/polish-nlp-resources#word2vec
#model = gensim.models.KeyedVectors.load("models/polish/model.bin")

# English embeddings http://vectors.nlpl.eu/repository/ (English CoNLL17 corpus)
model = gensim.models.KeyedVectors.load_word2vec_format("models/english/model.bin", binary=True)


# I've outlined a couple of tasks for you below to experiment with. Use these just a stepping off points to explore the nature of word embeddings and how they work.
# 
# Work in small groups on these tasks and make sure to discuss the issues and compare results - preferably across languages!

# ### Task 1: Finding polysemy
# 
# Find a polysemous word (for example, "leaves" or "scoop") such that the top-10 most similar words (according to cosine similarity) contains related words from both meanings. An example is given for you below in English. 
# 
# Are there certain words for which polysemy is more of a problem?

# In[ ]:


model.most_similar("leaves")


# ### Task 2: Synonyms and antonyms
# 
# In the lecture, we saw that _cosine similarity_ can also be thought of as _cosine distance_, which is simply ```1 - cosine similarity```. So the higher the cosine distance, the further away two words are from each other and so they have less "in common".
# 
# Find three words ```(w1,w2,w3)``` where ```w1``` and ```w2``` are synonyms and ```w1``` and ```w3``` are antonyms, but where: 
# 
# ```Cosine Distance(w1,w3) < Cosine Distance(w1,w2)```
# 
# For example, w1="happy" is closer to w3="sad" than to w2="cheerful".
# 
# Once you have found your example, please give a possible explanation for why this counter-intuitive result may have happened. Are there any inconsistencies?
# 
# You should use the the ```model.distance(w1, w2)``` function here in order to compute the cosine distance between two words. I've given a starting example below.

# In[ ]:


model.distance("happy", "sad")


# In[ ]:


model.distance("happy","cheerful")


# In[ ]:


model.distance("happy", "sad") < model.distance("happy","cheerful")


# ### Task 3: Word analogies
# 
# We saw in the lecture on Wednesday that we can use basic arithmetic on word embeddings, in order to conduct word analogy task.
# 
# For example:
# 
# ```man::king as woman::queen```
# 
# So we can say that if we take the vector for ```king``` and subtract the vector for ```man```, we're removing the gender component from the ```king```. If we then add ```woman``` to the resulting vector, we should be left with a vector similar to ```queen```.
# 
# NB: It might not be _exactly_ the vector for ```queen```, but it should at least be _close_ to it.
# 
# ```gensim``` has some quirky syntax that allows us to perform this kind of arithmetic.

# In[ ]:


model.most_similar(positive=['king', 'woman'], 
                   negative=['man'])[0]


# Try to find at least three analogies which correctly hold - where "correctly" here means that the closest vector corresponds to the word that you as a native speaker think it should.

# ### Task 3b: Wrong analogies
# 
# Can you find any analogies which _should_ hold but don't? Why don't they work? Are there any similarities or trends?

# In[ ]:





# ### Task 4: Exploring bias

# As we spoke briefly about in the lecture, word embeddings tend to display bias of the kind found in the training data.
# 
# Using some of the techniques you've worked on above, can you find some clear instances of bias in the word embedding models that you're exploring

# In[ ]:


model.most_similar(positive=['doctor', 'woman'], 
                   negative=['man'])


# ### Task 5: Dimensionality reduction and visualizing

# In the following cell, I've written a short bit of code which takes a given subset of words and plots them on a simple scatter plot. Remember that the word embeddings are 300d (or 100d here, depending on which language you're using), so we need to perform some kind of dimensionality reduction on the embeddings to get them down to 2D.
# 
# Here, I'm using a simply PCA algorithm implemented via ```scikit-learn```. An alternative approach might also be to use Singular Value Decomposition or SVD, which works in a similar but ever-so-slightly different way to PCA. You can read more [here](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/) and [here](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491) - the maths is bit mind-bending, just FYI.
# 
# Experiment with plotting certain subsets of words by changing the ```words``` list. How useful do you find these plots? Do they show anything meaningful?
# 

# In[ ]:


# the list of words we want to plot
words = ["man", "woman", "doctor", "nurse", "king", "queen", "boy", "girl"]

# an empty list for vectors
X = []
# get vectors for subset of words
for word in words:
    X.append(model[word])

# Use PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# or try SVD - how are they different?
#svd = TruncatedSVD(n_components=2)
# fit_transform the initialized PCA model
#result = svd.fit_transform(X)

# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])

# for each word in the list of words
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()


# ### Bonus tasks
# 
# If you run out of things to explore with these embeddings, try some of the following tasks:
# 
# [Easier]
# - make new plots like those above but cleaner and more informative
# - write a script which takes a list of words and produces the output above
#   
# [Very advanced]
# - work through [this](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html) documentation which demonstrates how to train word embedding using ```pytorch```. Compare this to the training documentation [here](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html) and think about how you would train a larger model on your own data.

# 

# 
