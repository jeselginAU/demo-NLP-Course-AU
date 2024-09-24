#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/rbroc/NLP-AU-23/blob/main/nbs/classroom_9.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Class 9 - Text Generation with LLMs, in-context learning, and model fine-tuning
# 
# Last week, we looked into NLG (inference-only) with HuggingFace models. We experimented with smaller models, which can be run on CPU (at least at inference). Some of them were doing a reasonable jobs in some tasks, but, overall, we saw a lot of room for improvement. Today, we focus on three ways to achieve better instruction-based performance:
# - Using large language models in a zero-shot fashion: as these models are prohibitively big, we will use a brand new library (`Petals`) to access them in a distributed fashion, and get a sense for what these models are capable of;
# - In-context learning (one- or few-shot)
# - Instruction fine-tuning
# 
# To begin, let's install [the Petals package](https://github.com/bigscience-workshop/petals) and a few other libraries we will need for this notebook.

# In[ ]:


get_ipython().run_line_magic('pip', 'install py7zr')
get_ipython().run_line_magic('pip', 'install datasets --no-deps')
get_ipython().run_line_magic('pip', 'install git+https://github.com/bigscience-workshop/petals')
# %pip install peft --no-deps


# Let's import the packages and functions we will need throughout the notebook.

# In[ ]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from petals import AutoDistributedModelForCausalLM


# ## Step 1. Using Petals to access very large LMs
# 
# Last week, we worked with fairly small models, whose performance was not always satisfactory. The first (and most trivial ;) ) way to get better performance is to, well, simply use larger models or models trained on larger data.
# 
# As many of these models are prohibitively big (i.e., several billion parameters) we cannot really download them and run them locally. To work with them, we will use a library for inference and fine-tuning of large language models that allows you to use very large LMs without the need to have high-end GPUs. 
# While Petals is not a feasible option for extensive tuning of large language models, it is great for inference and fine-tuning on limited sets of examples -- and it provides us access to models we would otherwise not be able to use.
# The idea behind Petals is that you can join compute resources with other people over the Internet and run large language models such as Llama 2 (70B) or BLOOM-176B right from your desktop computer or Google Colab.
# 
# The syntax of Petals is exactly the same as HuggingFace 🤗 [Transformers](https://github.com/huggingface/transformers) library, so you can re-use all the knowledge you have acquired in previous classes.
# Practically, the way this works is that you will download a small part of the model weights and rely on other computers in the network to run the rest of the model.

# 
# We start with [Stable Beluga 2 (70B)](https://huggingface.co/stabilityai/StableBeluga2), one of the best fine-tuned variants of Llama 2.
# 
# **Task**
# - Take some time to go through the documentation for Stable Beluga: how is the model trained, i.e., which task_datasets is it trained on? Is this data available? If so, can you find them and browse them in the HuggingFace hub? If not, why might this be problematic?
# - How does the training for StableBeluga compare to what we have discussed in our lectures? What does it have in common (in terms of approach to training) which some of the models we have discussed (GPT-3; T5; FlanT5; InstructGPT)?

# In[ ]:


model_name = "petals-team/StableBeluga2" # You could also other models supported from 🤗 Model Hub
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_bos_token=False)
model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
model = model.cuda()


# ### Let's generate!
# 
# Let's try to generate something by calling __`model.generate()`__ method, similar to what we have done last week. Last week we used one of HuggingFace's pipelines (`'textgeneration'`). What the `.generate()` method does is basically the same thing.
# 
# `model.generate()` method runs **greedy** generation by default. But there are several alternative decoding methods. You can **top-p/top-k sampling** or **beam search** to make your generation more creative, less predictable, less repetitive, etc -- just remember to set proper arguments for the 🤗 Transformers [.generate()](https://huggingface.co/blog/how-to-generate) method.
# 
# **Task**
# - Try to compare this to examples from last week's notebook: does StableBeluga (on any other model, if you are experimenting with something else) do better than models we tried out last week? Why may it be so?
# - Prompt StableBeluga to follow different instructions: how does it perform?

# In[ ]:


inputs = tokenizer('This morning, I woke up and', return_tensors="pt")["input_ids"].cuda()
outputs = model.generate(inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))


# Let's try to focus on a specific task, and use StableBeluga for text summarization. To get started, we load a dialogue summarization dataset from HuggingFace datasets, and we try to get a sense for how well StableBeluga summarizes some of the examples.

# In[ ]:


from datasets import load_dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)


# Let's start by visualizing an example from the dataset.

# In[ ]:


index = 23
print('Input:')
print(dataset['test'][index]['dialogue'])
print('Human summary:')
print(dataset['test'][index]['summary'])


# Ok, how does StableBeluga deal with this task? Let's see how well the model does with some basic instructions.

# In[ ]:


def _make_input(dialogue):
  out = f""" Dialogue: "{dialogue}". What was going on?"""
  return out


# In[ ]:


for i, index in enumerate([22]):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(_make_input(dialogue), return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs["input_ids"].cuda(),
            max_new_tokens=50,
        )[0][inputs["input_ids"][0].shape[0]:],
        skip_special_tokens=True
    )

    print(f'Input:\n{dialogue}')
    print(f'Human Summary:\n{summary}')
    print(f'Model Summary:\n{output}\n')


# Task:
# - How would you judge the overall quality of the summaries?
# - There is a suggested prompting template for StableBeluga: https://huggingface.co/petals-team/StableBeluga2. Experiment with prompting and see double-check if model behavior in this task is affected by changing the instructions.

# ### Step 2: One- and few-shot generation

# The reason why StableBeluga is performing this well is arguably a mix of good training protocols *and* size. Let's see if we can get away with using a smaller model with the same result.
# In our lecture, we alked about FlanT5, a model trained on instruction tuning for a wide range of traditional NLG and classification tasks. Let's load a small FlanT5 checkpoint and see how well this model performs.
# 
# Note that Petals does not currently support T5 models.  From now on, we will therefore work with `Transformers`.

# In[ ]:


# del model # optional step to offload StableBeluga


# In[ ]:


model_name_flan = "google/flan-t5-small"
tokenizer_flan = AutoTokenizer.from_pretrained(model_name_flan, use_fast=False)
model_flan = AutoModelForSeq2SeqLM.from_pretrained(model_name_flan) # , torch_dtype=torch.bfloat16
model_flan = model_flan.cuda() # putting the model on our GPU


# As we did with StableBeluga, let's define a function that adds instructions to the input. Note that the template we are using is the suggested FlanT5 template: https://github.com/google-research/FLAN/blob/main/flan/v2/templates.py. You can experiment with different prompts / instructions, to see how that affects the model.

# In[ ]:


def _make_input_flan(dialogue):
  prompt = f"""
      Dialogue:

      {dialogue}

      What was going on in the conversation?

      """
  return prompt


# In[ ]:


for i, index in enumerate([20]):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer_flan(_make_input_flan(dialogue), return_tensors='pt')
    output = tokenizer_flan.decode(
        model_flan.generate(
            inputs["input_ids"].cuda(),
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(f'Input:\n{dialogue}')
    print(f'Human Summary:\n{summary}')
    print(f'Model Summary:\n{output}\n')


# How is our model doing? :) Meh...
# 
# Let's think of a couple of ways to improve it. First, the easiest and most lightweight one: in-context learning: does our model do better if we pass it one or a few examples of what it is supposed to do?
# 
# **Task**
# - Run the same example as before through FlanT5, this time performing one-shot or few-shot generation
# - Are there any performance differences from the zero-shot case?
# - **Hint**: all you need to do, is to prepend a few examples of prompt/completion pairs to your input.

# In[ ]:


def _make_one_shot_example():
  ### Add your own code
  pass


# In[ ]:


# TODO: make inference


# **Bonus task**
# - Try loading another dataset: `samsum` (https://huggingface.co/datasets/samsum). Do you notice any performance differences? If so, do you have a hypothesis about why this is happening? What does it say about generalization and boundary conditions for fine-tuning of LLMs?

# ## Step 3. Fine-tuning a dialogue summarization model
# While in-context learning may help, fine-tuning might be another way to achieve the performance we are aiming to. Let's dive into that. First, let's try to get a feel for how big the model we are training is. the following function goes through the model architecture, and checks the number of parameters.

# In[ ]:


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(model_flan))


# Ok, now we know what we are dealing with. Next step is getting the data in shape for training. The function we define below is simply defining how each example in our dataset should be preprocessed. Then, the last two line apply this function to all examples.

# In[ ]:


def tokenize_function(example):
    start_prompt = 'Summarize this dialogue.\n\n'
    end_prompt = '\n\nSummary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer_flan(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer_flan(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = [
        [(l if l != tokenizer_flan.pad_token_id else -100) for l in label] for label in example['labels']
    ] # ignore padded tokens

    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
# tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)


# In[ ]:


tokenized_datasets


# Now that the data is in place, we need to define how we want to train our model. This is very similar to what we did a couple of classes ago, when we were fine-tuning BERT. As a matter of fact, all fine-tuning through HuggingFace happens with the same classes, which provide useful abstractions to define our training scheme. Some small adjustments may be required for specific tasks, but the overall logic and structure of the code will be the same.
# 
# Alright, let's define our training protocol, and initialize a  `Seq2SeqTrainer`, which is just a special instance of a `Trainer` adapted to sequence-to-sequence (or, as we called them in the lecture, text-to-text) tasks.

# In[ ]:


L_RATE = 1e-4
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="results",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   weight_decay=WEIGHT_DECAY,
   num_train_epochs=NUM_EPOCHS,
   max_steps=100,
   predict_with_generate=True,
   push_to_hub=False
)


trainer = Seq2SeqTrainer(
   model=model_flan,
   args=training_args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["test"],
   tokenizer=tokenizer_flan
)


# Now we are ready to train! Let's train the model, then save it and reload it.

# In[ ]:


trainer.train()


# In[ ]:


trainer.save_model("my_model") # let's save our trained model


# In[ ]:


instruct_model = AutoModelForSeq2SeqLM.from_pretrained("my_model").cuda()


# Finally, let's qualitatively inspect the outputs or our model.
# **Task**
# - Has our model improved from its baseline performance?
# - Do you still see room for improvement?
# - How do you think better performance could be achieved?

# In[ ]:


for i, index in enumerate([50]):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    start_prompt = 'Summarize this dialogue'
    end_prompt = 'Summary:' 
    prompt = start_prompt + dialogue + end_prompt

    inputs = tokenizer_flan(prompt, return_tensors='pt')
    output = tokenizer_flan.decode(
        instruct_model.generate(
            inputs["input_ids"].cuda(),
            max_new_tokens=50,
        )[0],
        skip_special_tokens=True
    )

    print(f'Input:\n{dialogue}')
    print(f'Human Summary:\n{summary}')
    print(f'Model Summary:\n{output}\n')


# ## Final comments
# Today, we have looked into a few additional aspects of NLG:
# - **How to access very large models through `Petals`**. These large models often provide pretty good zero-shot performance, but it is not feasible to run them locally, and fine-tuning on local resources is pretty much impossible. Petals makes it possible to use them for inference, and to also perform some lightweight fine-tuning. Check out the documentation for some examples including fine-tuning of StableBeluga: https://github.com/bigscience-workshop/petals
# - **One- and few-shot inference**: With smaller and more manageable models, you can sometimes achieve good performance with good prompting and in-context learning. This can happen, for example, if the model has been trained on similar tasks, and it just requires some nudging to adapt its behavior to a new task/data;
# - **Instruction fine-tuning**: If in-context learning is not enough, you can fine-tune your model. Note that here we have fine-tuned a relatively small model, and scaling to larger models can become prohibitively resource-intensive.
# 
# Resource limitations are a serious concern when thinking of fine-tuning LLMs. But the development of methods that make it possible to efficiently fine-tune LLMs on reasonably-sized infrastructure is a very active area of research!
# 
# In our last lecture of the course, I will mention a few methods that can help reduce the number of parameters we train in an efficient way, and in ways that do not significantly reduce performance (and that can easily be implemented through existing libraries). This class of methods is called PeFT (parameter-efficient fine-tuning), and it includes methods like **quantization**, **LoRA**, and **prompt tuning**.
# 
# If you are curious about these methods, `Petals` example notebooks implement training with LoRA + quantization (QLoRa). You can also take a look at HuggingFace's PEFT library: https://github.com/huggingface/peft
# 
# Finally, if you really liked the intuition behind RLHF, you check out what is already available in the open-source ecosystem (e.g., https://github.com/CarperAI/trlx). This is a new, exciting, and fast-growing areas.

# In[ ]:




