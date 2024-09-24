#!/usr/bin/env python
# coding: utf-8

# # Clasroom 7 - NER using BERT style models via Huggingface
# 
# In the previous assignment, you were tasked with training a NER model using a (Bi-)LSTM network in Pytorch.
# 
# In the lecture on Tuesday, we looked at transformer architectures and BERT-style models. We ended the lecture by suggesting how pre-trained models can be *finetuned* for specific tasks, such as NER.
# 
# In this notebook, you will see how we can finetune a pretrained BERT-style model via Huggingface for NER.
# 
# The point right now is to see how a *finetuning* pipeline differs from a model training pipeline, and where there are similarities. A secondary goal is to demonstrate the kind of performance increase we get from finetuning pretrained Language Models, compared to training a specific (Bi-)LSTM for the task.
# 
# This notebook has been freely adapted from Huggingface's documentation. There are lots of similar notebooks on their Github repo that you should check out - link [here](https://github.com/huggingface/transformers/tree/main/notebooks).
# 
# The first thing we need to do is to make sure that we have all of the relevant libraries installed:

# In[ ]:


#! pip install datasets transformers seqeval torch


# In[ ]:


import numpy as np
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_metric


# # Fine-tuning a model on a token classification task

# This notebook demonstrates in general how Huggingface can be used for finetuning models on *token classification* tasks.
# 
# The most common token classification tasks are:
# 
# - NER (Named-entity recognition) Classify the entities in the text (person, organization, location...).
# - POS (Part-of-speech tagging) Grammatically classify the tokens (noun, verb, adjective...)
# - Chunk (Chunking) Grammatically classify the tokens and group them into "chunks" that go together
# 
# We will see how to easily load a dataset for these kinds of tasks and use the `Trainer` API to fine-tune a model on it.

# In[ ]:


task = "ner" # Should be one of "ner", "pos" or "chunk"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16


# ## Loading the dataset

# For our example here, we'll use the [CONLLPP dataset](https://huggingface.co/datasets/conllpp), the one that used in the previous class. However, this notebook should work for any of the datasets available in the Huggingface datasets library.

# In[ ]:


datasets = load_dataset("conllpp")


# The `datasets` object itself is [`DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict), which contains one key for the training, validation and test set.

# In[ ]:


datasets


# We can see the training, validation and test sets all have a column for the tokens (the input texts split into words) and one column of labels for each kind of task we introduced before.

# To access an actual element, you need to select a split first, then give an index:

# In[ ]:


datasets["train"][0]


# The labels are already coded as integer ids to be easily usable by our model, but the correspondence with the actual categories is stored in the `features` of the dataset:

# In[ ]:


datasets["train"].features[f"ner_tags"]


# So for the NER tags, 0 corresponds to 'O', 1 to 'B-PER' etc... On top of the 'O' (which means no special entity), there are four labels for NER here, each prefixed with 'B-' (for beginning) or 'I-' (for intermediate), that indicate if the token is the first one for the current group with the label or not:
# - 'PER' for person
# - 'ORG' for organization
# - 'LOC' for location
# - 'MISC' for miscellaneous

# Since the labels are lists of `ClassLabel`, the actual names of the labels are nested in the `feature` attribute of the object above:

# In[ ]:


label_list = datasets["train"].features[f"{task}_tags"].feature.names
label_list


# ## Preprocessing the data

# Before we can feed those texts to our model, we need to preprocess them. Similar to what we did with LSTMs, we need to map words (in this case, tokens) to their number in the model's vocabulary, so that BERT can convert these into token embeddings. This means tokenizing them relative to pretrained vocabulary used by the BERT model that we want to use.
# 
# To do this, we can use a `Tokenizer` class which will tokenize text, get the corresponding IDs in the model vocabulary, and put it in a format the model expects, as well as generate the other inputs that model requires.
# 
# To make this easy for us, Huggingface lets us initialize our tokenizer with the `AutoTokenizer.from_pretrained` method, which will ensure:
# 
# - we get a tokenizer that corresponds to the model architecture we want to use,
# - we download the vocabulary used when pretraining this specific checkpoint.
# 
# That vocabulary will be cached, so it's not downloaded again the next time we run the cell.

# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# The following assertion ensures that our tokenizer is a fast tokenizers from the Tokenizers library. Those fast tokenizers are available for almost all models, and we will need some of the special features they have for our preprocessing.

# You can check which type of models have a fast tokenizer available and which don't on the [big table of models](https://huggingface.co/transformers/index.html#bigtable).

# You can directly call this tokenizer on one sentence.
# 
# **PAUSE**
# - Before running this code, in your groups discuss the following questions:
#   - What kind of result will we see when this cell is run?

# In[ ]:


tokenizer("Hello, this is one sentence!")


# If, as is the case below, your inputs have already been split into words, you should pass the list of words to your tokenzier with the argument `is_split_into_words=True`:

# In[ ]:


tokenizer(["Hello", ",", "this", "is", "one", "sentence", "split", "into", "words", "."], is_split_into_words=True)


# Note that transformers are often pretrained with subword tokenizers, meaning that even if your inputs have been split into words already, each of those words could be split again by the tokenizer. Let's look at an example of that:

# In[ ]:


example = datasets["train"][4]
print(example["tokens"])


# In[ ]:


tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)


# Here the words "Zwingmann" and "sheepmeat" have been split in three subtokens.
# 
# This means that we need to do some processing on our labels as the input ids returned by the tokenizer are longer than the lists of labels our dataset contain, first because some special tokens might be added (we can a `[CLS]` and a `[SEP]` above) and then because of those possible splits of words in multiple tokens:

# In[ ]:


len(example[f"{task}_tags"]), len(tokenized_input["input_ids"])


# Thankfully, the tokenizer returns outputs that have a `word_ids` method which can help us.

# In[ ]:


print(tokenized_input.word_ids())


# As we can see, it returns a list with the same number of elements as our processed input ids, mapping special tokens to `None` and all other tokens to their respective word. This way, we can align the labels with the processed input ids.

# In[ ]:


word_ids = tokenized_input.word_ids()
aligned_labels = [-100 if i is None else example[f"{task}_tags"][i] for i in word_ids]
print(len(aligned_labels), len(tokenized_input["input_ids"]))


# Here we set the labels of all special tokens to -100 (the index that is ignored by PyTorch) and the labels of all other tokens to the label of the word they come from. Another strategy is to set the label only on the first token obtained from a given word, and give a label of -100 to the other subtokens from the same word. We propose the two strategies here, just change the value of the following flag:

# In[ ]:


label_all_tokens = True


# We're now ready to write the function that will preprocess our samples. We feed them to the `tokenizer` with the argument `truncation=True` (to truncate texts that are bigger than the maximum size allowed by the model) and `is_split_into_words=True` (as seen above). Then we align the labels with the token ids using the strategy we picked.
# 
# **PAUSE**
# - In your groups, inspect the function below. Looking back over the previous cells in this notebook, add comments to this function explaining how it works.

# In[ ]:


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# This function works with one or several examples. In the case of several examples, the tokenizer will return a list of lists for each key.
# 
# **PAUSE**
# - Again, what do you expect the output of this function to be? Look at the earlier example of tokenization, and also the function in the previous cell.

# In[ ]:


tokenize_and_align_labels(datasets['train'][:5])


# **Pause**
# - What do you think the code in the following cell does? Why? (Hint: think back to the LSTM notebook.)

# In[ ]:


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


# The method called ```map``` which is defined for ```dataset``` objects. You might be familiar with map from programming in ```R```, which things like map, filter, and lapply are more common than in the 
# 
# This will apply the function on all the elements of all the splits in `dataset`, so our training, validation and testing data will be preprocessed in one single command.

# *A note from the Hugginface documentation*
# 
# The results of ```.map``` are automatically cached by the Datasets library to avoid spending time on this step the next time you run your notebook. The Datasets library is normally smart enough to detect when the function you pass to map has changed (and thus requires to not use the cache data). For instance, it will properly detect if you change the task in the first cell and rerun the notebook. 
# 
# Datasets warns you when it uses cached files, you can pass `load_from_cache_file=False` in the call to `map` to not use the cached files and force the preprocessing to be applied again.
# 
# Note that we passed `batched=True` to encode the texts by batches together. This is to leverage the full benefit of the fast tokenizer we loaded earlier, which will use multi-threading to treat the texts in a batch concurrently.

# ## Fine-tuning the model

# Now that our data is ready, we can download the pretrained model and fine-tune it. Since all our tasks are about token classification, we use the `AutoModelForTokenClassification` class. 
# 
# Like with the tokenizer, the `from_pretrained` method will download and cache the model for us. The only thing we have to specify is the number of labels for our problem (which we can get from the features, as seen before):

# In[ ]:


model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))


# The warning is telling us we are throwing away some weights (the `vocab_transform` and `vocab_layer_norm` layers) and randomly initializing some other (the `pre_classifier` and `classifier` layers). 
# 
# This is to be exepected because we're discarding the head used to pretrain the model on a masked language modeling objective and replacing it with a new head for which we don't have pretrained weights, so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.
# 
# Don't worry if that's a little opaque; we'll dig into it in more detail in coming weeks.

# We then want to initalize a ```Trainer``` class.
# 
# To do this, we have to defined the [`TrainingArguments`](https://huggingface.co/transformers/main_classes/trainer.html#transformers.TrainingArguments), which is a class that contains all the attributes to customize the training. It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional:

# In[ ]:


model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    use_mps_device=True) # you may have to comment this out, if you get a related error


# Here we set the evaluation to be done at the end of each epoch, tweak the learning rate, use the `batch_size` defined at the top of the notebook and customize the number of epochs for training, as well as the weight decay.

# Then we will need a data collator that will batch our processed examples together while applying padding to make them all the same size (each pad will be padded to the length of its longest example). There is a data collator for this task in the Transformers library, that not only pads the inputs, but also the labels:

# In[ ]:


data_collator = DataCollatorForTokenClassification(tokenizer)


# The last thing to define for our `Trainer` is how to compute the metrics from the predictions. Here we will load the [`seqeval`](https://github.com/chakki-works/seqeval) metric (which is commonly used to evaluate results on the CONLL dataset) via the Datasets library.

# In[ ]:


metric = load_metric("seqeval")


# This metric takes list of labels for the predictions and references.
# 
# **PAUSE**
# - What do you think this cell is going to show? 

# In[ ]:


labels = [label_list[i] for i in example[f"{task}_tags"]]
metric.compute(predictions=[labels], references=[labels])


# We need to do a bit of postprocessing to get our computed results in to the format that we need for evaluating the metrics.
# 
# The following function does all this post-processing on the result of `Trainer.evaluate` (which is a namedtuple containing predictions and labels) before applying the metric.
# 
# **PAUSE**
# - Read through this function and add comments explaining how it works and what it's doing.

# In[ ]:


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Note that we drop the precision/recall/f1 computed for each category and only focus on the overall precision/recall/f1/accuracy.
# 
# Then we just need to pass all of this along with our datasets to the `Trainer`:

# In[ ]:


trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# We can now finetune our model by just calling the `train` method.
# 
# **PAUSE**
# - When you run this cell, notice how much longer it takes to run than the LSTM model!
# - Can you make sense of all of the outputs presented to you on the screen?

# In[ ]:


trainer.train()


# The `evaluate` method allows you to evaluate again on the evaluation dataset or on another dataset:

# In[ ]:


trainer.evaluate()


# To get the precision/recall/f1 computed for each category now that we have finished training, we can apply the same function as before on the result of the `predict` method:

# In[ ]:


predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

results = metric.compute(predictions=true_predictions, references=true_labels)
results


# **Questions**
# - Compared to the LSTM approach, which pipeline seems more intuitive? Does one seem preferable to the other?
# - How does the performance of this BERT-style model after 3 epochs compare to the LSTM model?
# - What limitations does this approach have over the model you developed in the assignment?
# - Can you access the number of parameters of this model, and compare it with the number of parameters of the LSTM?
# 
# **Bonus challenge**
# - Can you implement a sentiment classifier using the glue/sst dataset we used in previous classes for sentiment classification?
# - Hint: you will have to use `AutoModelForSequenceClassification` this time 
# - Preprocessing will be much simpler :) 

# 
