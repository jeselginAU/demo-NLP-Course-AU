#!/usr/bin/env python
# coding: utf-8

# ## Introduction to SpaCy
# SpaCy is one of the main libraries for NLP in Python. It is especially tailored towards the (development and) deployment of pipelines for a number of NLP tasks, from named entity recognition to dependency parsing. Recent additions to spaCy also make it possible to use advanced models (LLMs) as building block for such pipelines. 
# 
# In this notebook, we will explore some of SpaCy's basic functionality. As we will mainly focus on understanding and implementing methods to represent and generate text, we will not be using SpaCy extensively for the rest of the course -- but it is a great tool for many applications, and it has wonderful documentation.
# 
# Let's start by importing SpaCy.
# You may need to install this first (I am using `spacy==3.6.1`) -- add it to your requirements file if you have one.

# In[2]:


import spacy
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import re
from collections import defaultdict
from sklearn.decomposition import PCA


# When spaCy is loaded, we then need to initialize a model.
# 
# NB: Models first have to be downloaded from the command line. An overview of avaiable models from spaCy can be found [here](https://spacy.io/usage/models):
# ```
# python -m spacy download en_core_web_md
# ```
# Note that models for a wide range of languages are available on SpaCy, feel free to experiment!

# In[3]:


nlp = spacy.load("en_core_web_md")


# We first create a `spaCy` pipeline which is going to be used for all of our analysis. Essentially we feed our examples of language down the pipeline, and get annotated texts out the end. Let's load the first 10 chapters of "War and Peace", the text file we worked with last week.

# In[4]:


book = open('../data/book-war-and-peace.txt').read()


# In[6]:


print(book[:2000])


# Let's now find where chapter 10 starts, and only keep text until there.

# In[7]:


chap_10_index = re.search('CHAPTER X', book).start()
book = book[:chap_10_index]


# The final object that comes out of the end is known as a `spaCy` `Doc` which is essentially a list of tokens. However, rather than just being a list of strings, each of the tokens in this list have their own attributes, which can be accessed using the dot notation.

# In[8]:


doc = nlp(book)


# The resulting `doc` parses the text in sentences and tokens within sentences.

# In[9]:


for s in doc.sents:
    print(s)
    for t in s:
        print(t)


# Each token in the doc is annotated for a number of attributes.

# In[10]:


i = 0
for token in doc:
    i += 1
    print(token.text, "\t\t", token.pos_, "\t\t", token.dep_,"\t\t", token.lemma_, "\t\t")
    if i == 50:
        break


# ### Exploring named entities

# Let's explore one of the features of spaCy: named entity recognition. These are all the named entities spaCy finds in the text.

# In[11]:


doc.ents


# SpaCy also has some nice utils to visualize like named entities or dependency relations between individual words (try replace `ent` with `dep`). Let's see how this looks:

# In[12]:


spacy.displacy.serve(doc[:300], style="ent")


# Now let's try to look at the frequency of each entity, and see what information we can extract on the characters named in the book.

# In[13]:


entity_counts = defaultdict(lambda: 0)
for e in doc.ents:
    if e.label_ == 'PERSON':
        entity_counts[e.text] += 1


# In[14]:


import pandas as pd
entity_df = pd.DataFrame.from_dict(entity_counts, 
                                   orient='index').reset_index()
entity_df = entity_df.rename({'index': 'entity', 0: 'count'}, axis=1).sort_values(by='count', ascending=False)


# In[15]:


entity_df


# In[16]:


sns.catplot(data=entity_df.head(n=50), x='entity', y='count', kind='bar', height=5, aspect=2)
plt.xticks(rotation=90)
plt.show()


# We have a pretty accurate model of who they main characters in War and Peace are! You may notice that some entities are actually duplicates (`Anna Pavlovna` and `Anna Pavlovna's`: these could in principle be manually normalized or, more elegantly, clustered using pipelines for a task called "coreference resolution"). 

# ### Character time series
# As a demonstration of what you can do with this, let's focus on the top 3 characters. Can we plot a time series, visualizing how many times they are mentioned in each of the 9 chapters?

# In[17]:


idxs = []
for t in doc:
    if t.text == 'CHAPTER':
        idxs.append(t.idx) # append start index of the chapter


# In[18]:


char_count_by_chapter = dict(zip(entity_df.entity.head(n=3).tolist(),
                                 [dict(zip(range(1,10),[0]*9)),
                                  dict(zip(range(1,10),[0]*9)),
                                  dict(zip(range(1,10),[0]*9))])) # there are more elegant ways to do this, with defaultdict

for e in doc.ents:
    if e.text in char_count_by_chapter.keys():
        for nr, i in enumerate(idxs):
            if e.start_char < i:
                char_count_by_chapter[e.text][nr] += 1 # is this correct?
                break
            else:
                if nr == 8:
                    char_count_by_chapter[e.text][9] += 1


# In[19]:


dfs = []
for k,v in char_count_by_chapter.items():
    df = pd.DataFrame.from_dict(v, orient='index').reset_index().rename({'index': 'chapter',
                                                                         0: 'count'}, axis=1)
    df['character'] = k
    dfs.append(df)
char_df = pd.concat(dfs)


# In[20]:


sns.lineplot(data=char_df, x='chapter', y='count', hue='character', marker='o')


# ## Word vectors
# One of the attributes that spaCy models provide is easy access to word vectors. These are not based on counts, but on more sophisticated algorithms that we will look into in detail next week, but the intuition is the same as count-based vectors. Let's use this to put some of the notions we explored in our lecture into practice. First, let's take a look at some of the tokens in our text

# In[21]:


for i, t in enumerate(doc[:20]):
    print(i, t.text)


# Let's focus on the token "family" (occurring, e.g., at index 15). We want to identify the words that are most similar to family (of those present in the text). With SpaCy, we can compute cosine similarity between vectors using in-build functionality. The following piece of code computes the similarity between "family" (the token at index 15) and "estates" (the token at index 16).

# In[22]:


doc[15].similarity(doc[16])


# Based on this, can you identify the 20 tokens, of those occurring in our doc, whose vectors are *most similar* to "family"? Do the results make sense?

# In[23]:


sims = {}
for i, t in enumerate(doc):
    if t.text not in sims.keys():
        cosine_sim = doc[15].similarity(doc[i])
        sims[t.text] = cosine_sim
cos_df = pd.DataFrame.from_dict(sims, orient='index').reset_index() 
cos_df.columns=['token', 'cosine_sim']
cos_df = cos_df.sort_values(by='cosine_sim', ascending=False)


# In[24]:


cos_df.head(n=20)


# This looks very promising: can you do the same with other words? What happens if you look at the *most dissimilar* words?

# In[25]:


cos_df.tail(n=20)


# Finally, let's visualize some vectors. Let's sample 200 random nouns, reduce the vector dimensionality with a technique called principal component analysis, and let's visualize the resulting space.

# In[26]:


random_indices = []
for i, t in enumerate(doc):
    if t.pos_ == 'NOUN':
        random_indices.append(i)
        if len(random_indices) == 200:
            break


# In[27]:


pca = PCA(n_components=2)
vectors = np.vstack([doc[i].vector for i in random_indices])
reduced_vectors = pca.fit_transform(vectors) # transform into a 2d space


# In[28]:


reduced_vectors


# Now let's plot the reduced vectors in 2D space:

# In[29]:


plt.subplots(figsize=(20,20))
sns.scatterplot(x=reduced_vectors[:200,0], y=reduced_vectors[:200,1])
for i in range(200):
    plt.text(reduced_vectors[i,0], 
             reduced_vectors[i,1], 
             doc[random_indices[i]])


# Do you notice anything promising in terms of relations between vectors?

# Note that you can also use `scikit-learn` to compute `euclidean_distances` and `cosine_similarity`, see: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise. These functions require a matrix as input, and they compute all pairwise similarities between rows of the matrix.

# **Optional**: can you implement your own versions functions to compute euclidean distance and cosine similarity? Look at the formulas from Lecture 2, and use `spaCy` or `scikit-learn` functions to check that they are correct. We will experiment more with word vectors next week.

# ## Task
# In the shared data drive on UCloud, there is a folder called `data`, where you can find a file called `News_Category_Dataset_v2.json`. This is taken from [this Kaggle exercise](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and comprises some 200k news headlines from [HuffPost](https://www.huffpost.com/). The data is a json lines format, with one JSON object per row. You can load this data into pandas in the following way:
# ```
# data = pd.read_json(filepath, lines=True)
# ```
# Select a couple of sub-categories of news data and use spaCy to find the relative frequency per **10k words** of each of the following word classes - NOUN, VERB, ADJECTIVE, ADVERB (in the headlines).
# Save the results as a CSV file (again using pandas).
# Are there any differences in the distributions?

# In[30]:


import pandas as pd


# In[33]:


df = pd.read_json('../data/News_Category_Dataset_v3.json', lines=True) # path to files on UCloud


# In[73]:


df.category.unique()


# In[82]:


df = df[df['category'].isin(['COMEDY', 'CRIME', 'POLITICS'])]


# In[85]:


df[df['category']=='COMEDY'] # ['headline'].tolist()


# In[92]:


from collections import Counter

# deactivate expensive computations
nlp.select_pipes(disable=['ner','parser'])
# convenience trick to avoid limits, there's other ways to avoid them -- e.g., applying the pipeline to each element in parallel 
nlp.max_length = 2323037 

cats = df.category.unique().tolist()
outs = []
for c in cats:
    text = ' '.join(df[df['category']==c]['headline'].tolist())
    doc = nlp(text)
    pos_list = []
    for t in doc:
        pos = t.pos_
        if pos in ['ADJ', 'ADV', 'NOUN', 'VERB']:
            pos_list.append(t.pos_)
    count_df = pd.DataFrame.from_dict(dict(Counter(pos_list)), 
                                      orient='index').reset_index()
    count_df.columns = ['pos', 'count']
    count_df['n_tokens'] = len(doc)
    count_df['category'] = c
    count_df['count_normalized'] = round((count_df['count'] / count_df['n_tokens']) * 10000,4)
    outs.append(count_df[['pos', 
                          'count_normalized', 
                          'category']].sort_values(by='count_normalized'))


# In[94]:


pd.concat(outs)[::-1]

