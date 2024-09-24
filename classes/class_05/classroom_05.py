#!/usr/bin/env python
# coding: utf-8

# # Classroom 5 - Training a Named Entity Recognition Model with a LSTM

# The classroom today focuses on using LSTMs to train a named entity recognition model, which is an example of a "many-to-many" recurrent model.
# 
# **Note**: there are a few exercises to solve in this notebook. As it is the first exercise where I am asking you to implement several code chunks by yourself (though with plenty of guidance), this week I am uploading the solutions in a separate notebook. 
# If you get stuck on something for a long time and your want some help to get unstuck, you can take a look at the other notebook -- but try not do do so unless it is really the last resource (i.e., you have tried hard, asked around, googled stuff, etc.)

# ## 1. A very short intro to NER
# Named entity recognition (NER) also known as named entity extraction, and entity identification is the task of tagging an entity is the task of extracting which seeks to extract named entities from unstructured text into predefined categories such as names, medical codes, quantities or similar.
# 
# The most common variant is the [CoNLL-20003](https://www.clips.uantwerpen.be/conll2003/ner/) format which uses the categories, person (PER), organization (ORG) location (LOC) and miscellaneous (MISC), which for example denote cases such nationalies. For example:
# 
# *Hello my name is $Roberta_{PER}$ I live in $Aarhus_{LOC}$ and work at $AU_{ORG}$.*
# 
# For example, let's see how this works with ```spaCy```. NB: you might need to remember to install a ```spaCy``` model:
# 
# ```python -m spacy download en_core_web_sm```

# In[ ]:


import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello my name is Roberta. I live in Denmark and work at Aarhus University, I am Italian and today is Wednesday 25th.")


# In[ ]:


from spacy import displacy
displacy.render(doc, style="ent")


# ## Tagging standards
# There exist different tag standards for NER. The most used one is the BIO-format which frames the task as token classification denoting inside, outside and beginning of a token. 
# 
# Words marked with *O* are not a named entity. Words with NER tags which start with *B-\** indicate the start of a multiword entity (i.e. *B-ORG* for the *Aarhus* in *Aarhus University*), while *I-\** indicate the continuation of a token (e.g. University).
# 
#     B = Beginning
#     I = Inside
#     O = Outside
# 
# <details>
# <summary>Q: What other formats and standards are available? What kinds of entities do they make it possible to tag?</summary>
# <br>
# You can see more examples on the spaCy documentation for their [different models(https://spacy.io/models/en)
# </details>
# 

# In[ ]:


for t in doc:
    if t.ent_type:
        print(t, f"{t.ent_iob_}-{t.ent_type_}")
    else:
        print(t, t.ent_iob_)


# ### Some challenges with NER
# While NER is currently framed as above this formulating does contain some limitations. 
# 
# For instance the entity Aarhus University really refers to both the location Aarhus, the University within Aarhus, thus nested NER (N-NER) argues that it would be more correct to tag it in a nested fashion as \[\[$Aarhus_{LOC}$\] $University$\]$_{ORG}$ (Plank, 2020). 
# 
# Other task also include named entity linking. Which is the task of linking an entity to e.g. a wikipedia entry, thus you have to both know that it is indeed an entity and which entity it is (if it is indeed a defined entity).
# 
# We will be using Bi-LSTMs to train an NER model on a predifined data set which uses IOB tags of the kind we outlined above.

# ## 2. Training in batches
# 
# In previous classes, we discussed stochastic gradient descent on mini-batches as a way to achieve an optimal tradeoff between performance and stability.
# Let's implement batching!
# 
# <details>
# <summary>Reminder: Why might it be a good idea to train on batches, rather than the whole dataset?</summary>
# <br>
# These batches are usually small (something like 32 instances at a time) but they have couple of important effects on training:
# 
# - Batches can be processed in parallel, rather the sequentially. This can result in substantial speed up from computational perspective
# - Similarly, smaller batch sizes make it easier to fit training data into memory
# - Lastly,  smaller batch sizes are noisy, meaning that they have a regularizing effect and thus lead to less overfitting.
# 
# In this notebook, we're going to be using batches of data to train our NER model. To do that, we first have to prepare our batches for training. You can read more about batching in [this blog post](https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/).
# 
# </details>
# 
# 

# In[ ]:


# this allows us to look one step up in the directory
# for importing custom modules from src
import sys
sys.path.append("..")
from src.util import batch
from src.LSTM import LSTMModel
from src.embedding import gensim_to_torch_embedding

# numpy and pytorch
import numpy as np
import torch

# loading data and embeddings
from datasets import load_dataset
import gensim
import gensim.donwloader as api


# We can download the datset using the ```load_dataset()``` function we've already seen. Here we take only the training data.
# 
# When you've downloaded the dataset, you're welcome to save a local copy so that we don't need to constantly download it again everytime the code runs.
# 
# Q: What do the ```train.features``` values refer to?

# In[ ]:


# DATASET
dataset = load_dataset("conllpp")
train = dataset["train"]

# inspect the dataset
train["tokens"][:1]
train["ner_tags"][:1]

# get number of classes
num_classes = train.features["ner_tags"].feature.num_classes


# We then use ```gensim``` to get some pretrained word embeddings for the input layer to the model. 
# 
# In this example, we're going to use a GloVe model pretrained on Wikipedia, with 50 dimensions.
# 
# I've provided a helper function to take the ```gensim``` embeddings and prepare them for ```pytorch```.

# In[120]:


# CONVERTING EMBEDDINGS
model = api.load("glove-wiki-gigaword-50")
# you can also try: model = gensim.models.KeyedVectors.load_word2vec_format("../../../819739/models/english/model.bin", binary=True) -- note that dimensionality is different
embedding_layer, vocab = gensim_to_torch_embedding(model)


# ### Preparing a batch
# 
# The first thing we want to do is to shuffle our dataset before training. 
# 
# Why might it be a good idea to shuffle the data?

# In[ ]:


# shuffle dataset
shuffled_train = dataset["train"].shuffle(seed=1)
validation = dataset["validation"]
test = dataset["test"]


# Next, we want to bundle the shuffled training data into smaller batches of predefined size. I've written a small utility function here to help. 
# 
# <details>
# <summary>Q: Can you explain how the batch() function works?</summary>
# <br>
#  Hint: Check out [this link](https://realpython.com/introduction-to-python-generators/).
# </details>
# 
# 

# In[ ]:


batch_size = 32
batches_tokens = batch(shuffled_train["tokens"], batch_size)
batches_tags = batch(shuffled_train["ner_tags"], batch_size)


# Next, we want to use the ```tokens_to_idx()``` function below on our batches.
# 
# <details>
# <summary>Q: What is this function doing? Why is it doing it?</summary>
# <br>
# We're making everything lowercase and adding a new, arbitrary token called <UNK> to the vocabulary. This <UNK> means "unknown" and is used to replace out-of-vocabulary tokens in the data - i.e. tokens that don't appear in the vocabulary of the pretrained word embeddings.
# </details>
# 

# In[ ]:


def tokens_to_idx(tokens, vocab=model.key_to_index):
    """
    - Write documentation for this function including type hints for each argument and return statement
    - What does the .get method do?
    - Why lowercase?
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]


# We'll check below that everything is working as expected by testing it on a single batch.

# In[ ]:


# sample using only the first batch
batch_tokens = next(batches_tokens)
batch_tags = next(batches_tags)
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens]


# As with document classification, our model needs to take input sequences of a fixed length. To get around this we do a couple of different steps.
# 
# - Find the length of the longest sequence in the batch
# - Pad shorter sequences to the max length using an arbitrary token like <PAD>
# - Give the <PAD> token a new label ```9``` to differentiate it from the other labels

# In[ ]:


# compute length of longest sentence in batch
batch_max_len = max([len(s) for s in batch_tok_idx])


# Q: Can you figure out the logic of what is happening in the next two cells?

# In[ ]:


batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len))
batch_labels = 9 * np.ones((batch_size, batch_max_len))


# In[ ]:


# copy the data to the numpy array
for i in range(batch_size):
    tok_idx = batch_tok_idx[i]
    tags = batch_tags[i]
    size = len(tok_idx)

    batch_input[i][:size] = tok_idx
    batch_labels[i][:size] = tags


# The last step is to convert the arrays into ```pytorch``` tensors, ready for the NN model.

# In[ ]:


# since all data are indices, we convert them to torch LongTensors (integers)
batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(
    batch_labels
)


# With our data now batched and processed, we want to run it through our RNN the same way as when we trained a classifier.
# 
# Q: Why is ```output_dim = num_classes + 1```?

# In[ ]:


# Create model
model = LSTMModel(
    embedding_layer=embedding_layer, output_dim=num_classes + 1, hidden_dim_size=256
)

# Forward pass
X = batch_input
y = model(X)

loss = model.loss_fn(outputs=y, labels=batch_labels)


# ## 3. Creating an LSTM with ```pytorch```

# In the file [LSTM.py](../src/LSTM.py), I've aready created an LSTM for you using ```pytorch```. Take some time to read through the code and make sure you understand how it's built up.
# 
# Some questions for you to discuss in groups:
# 
# - How is an LSTM layer created using ```pytorch```? How does the code compare to the classifier code we used last week?
# - What's going on with that weird bit that says ```@staticmethod```?
#   - [This might help](https://realpython.com/instance-class-and-static-methods-demystified/).
# - On the forward pass, we use ```log_softmax()``` to make output predictions. What is this, and how does it relate to the output from the sigmoid function that we used for classification?
# - How would we make this LSTM model *bidirectional* - i.e. make it a Bi-LSTM? 
#   - Hint: Check the documentation for the LSTM layer on the ```pytorch``` website.

# ## 4. Training the LSTM for named entity recognition

# In this last part of the notebook, we are going to use bits of code we have seen today (related to batching) and code from last week to set up training and evaluation for an LSTM doing named entity recognition. There are a few parts of the code you have to fill: work in groups and note that there is nothing new you need to do! All you are required to do has been done earlier in this notebook, or last week.

# **Task 1**: Package all the code from previous steps into a single function, which prepares a batch of data for the model (we will apply this to all batches in the following chunks)

# In[ ]:


def prepare_batch(tokens, labels, vocab) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare a batch of data for training.

    Args:
        tokens (List[List[str]]): A list of lists of tokens.
        labels (List[List[int]]): A list of lists of labels.
        vocab (dict): A dictionary defining the model's vocabulary

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of tensors containing the tokens and labels.
    """
    batch_size = len(tokens)

    # TODO: convert tokens to vocabulary items using the tokens_to_idx function
    
    # TODO: compute length of longest sentence in batch.

    # TODO: Pad the data and flag padded vs non-padded with 9

    # TODO: copy the data to numpy array

    # TODO: convert data to tensors

    # TODO: return two outputs: batch_input, batch_labels, tensors containing respectively tokens and true labels
    return None 


# Now let's create a function that trains the model. 
# 
# **Questions**:
# What is the last part of the function doing?
# Think back of our first neural networks class: which problem is this trying to prevent?

# In[ ]:


def train_model(model, optimizer, epochs, training, validation, vocab, patience, batch_size):
    """
    A function for training the model.
    """
    best_val_loss = None

    # TODO: apply the prepare_batch function to the validation set, to get inputs and labels

    for epoch in range(epochs):
        batches_processed = 0
        print(f'*** Epoch {epoch+1} ***')
        
        # TODO: pass training['tokens'] and training['ner_tags'] them through the batch() function: call the outputs b_tokens and b_tags
    
        for tokens, tags in zip(b_tokens, b_tags):
            batches_processed += 1
            if batches_processed % 10 == 0:
                print(f'Batch {batches_processed}')
                
            # prepare data
            # TODO: use the prepare_batch function to create inputs and outputs

            # train the model
            #TODO: perform a forward pass
            #TODO: compute the loss using the loss_fn method from our model (take a look at src/LSTM.py)
            #TODO: compute the gradients
            #TODO: update the weights (hint: look at notebook from last week!)

            optimizer.zero_grad()

        #  periodically calculate loss on validation set
        if epoch % 5 == 0:
            #TODO: perform a forward pass on the validation set
            #TODO: compute the loss on the validation set (call it "loss")

            # QUESTION: what is this part of the code doing?
            if best_val_loss is None or loss < best_val_loss:
                best_val_loss = loss
                torch.save(model, 'model.pt')
                patience_ = patience
            else:
                patience_ -= 5
                if patience_ <= 0:
                    break


# Finally, we define a function that runs the whole thing (training and evaluation) end-to-end. Take some time to understand what this function does, and note down any questions you might have.

# In[ ]:


def run(gensim_embedding: str, batch_size: int, epochs: int, learning_rate: float, patience: int = 10, optimizer=0):
    """
    A function that does end-to-end data prepraration, training, and evaluation
    """
    # set a seed to make the results reproducible
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # use the function gensim_to_torch_embeddings to create embedding_layer and vocab
    embeddings = api.load(gensim_embedding)
    embedding_layer, vocab = gensim_to_torch_embedding(embeddings)

    # Preparing data
    # shuffle dataset
    dataset = load_dataset("conllpp")
    train = dataset["train"].shuffle(seed=1)
    test = dataset["test"]
    validation = dataset["validation"]

    # Compute the number of classes for LSTM output (+1 for PAD)
    num_classes = train.features["ner_tags"].feature.num_classes

    # Initialize the model
    lstm = LSTMModel(num_classes + 1, embedding_layer, 20)

    # Initialize optimizer
    if optimizer == 0:
        optimizer = torch.optim.AdamW(lstm.parameters(), lr=learning_rate)
    elif optimizer == 1:
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.RMSprop(lstm.parameters(), lr=learning_rate)

    # train model with given settings
    train_model(lstm, optimizer, epochs, train, validation, vocab, patience, batch_size)

    # Load the best model
    best = torch.load('model.pt')

    # test it on test set
    X, y = prepare_batch(test["tokens"], test["ner_tags"], vocab)
    y_hat = best.predict(X)

    # reformat results by removing pad tokens and flattening
    y_hat_depadded = []
    pos = 0
    for sen in test["ner_tags"]:
        for i in range(pos, pos + len(sen)):
            y_hat_depadded.append(y_hat[i])
        pos += y.shape[1]

    # flatten the test sentences into a single list
    flat_tags = [item for sublist in test["ner_tags"] for item in sublist]
    
    # get actual label
    actual = []
    predicted = []
    ner_dict = {0:'O', 1:'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC', 9: 'NONE'}
    # in theory we would want to exclude label 9 -- but let's keep it in for simplicity
    actual.append([ner_dict.get(k, k) for k in flat_tags])
    predicted.append([ner_dict.get(k, k) for k in y_hat_depadded])

    # calculate f1 and acc (currently the same number?)
    report = classification_report(actual, predicted)
    print(report)

    return None


# In[ ]:


# TODO: use the run function to train and evaluate the model
# hint: start with learning_rate = 0.01


# ### Task:
# - How do results change between bidirectional and unidirectional models?
# - How does the size of the LSTM affect performance?
# - Is the performance of the model balanced for all classes?
# 
# ### Bonus task:
# - If you want to evaluate performance as a function of parameters systematically, try implement all this through scripts, and log results as separate files outputs
# - A good way to monitor training is by using Weights&Biases. Check out their documentation and feel free to experiment: https://docs.wandb.ai/guides/integrations/pytorch
