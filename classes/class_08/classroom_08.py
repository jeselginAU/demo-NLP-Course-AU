#!/usr/bin/env python
# coding: utf-8

# # Classroom 8 - Text Generation with HuggingFace transformers

# In this class, we'll look at how we can easily use HuggingFace Transformers for both text generation and for text-to-text models like T5. As with many tasks, Huggingface makes the basic aspects of text generation extremely simple to implement.
# You can read about HuggingFace's text generation pipelines [here](https://huggingface.co/tasks/text-generation).

# In[ ]:


get_ipython().system('pip install transformers torch')


# ## Text completion 
# AKA: forward/causal/autoregressive language modeling.

# Let's start playing with language generation in HuggingFace. I have created a template for language generation, and the goal for today is to explore how we can use NLG models through HuggingFace, and how some of the notions we have talked about in the lecture (e.g., decoding algorithms, task-specific instructions for NLG tasks etc) are implemented in this framework. As you go through this notebook, make sure you spend some time familiarizing with HuggingFace's documentation and model hub.
# 
# You can find a full range of models which are available for text generation via the Huggingface model zoo [here](https://huggingface.co/models?pipeline_tag=text-generation).
# 
# Try experimenting with some of the following model architectures:
# - GPT-J / GPT2
# - BigScience BLOOM (e.g., BLOOM-560m)
# - T5 / FlanT5 (a multi-task version of T5)
# - Falcon (take a look at https://huggingface.co/tiiuae/falcon-7b and https://huggingface.co/tiiuae/falcon-7b-instruct)
# 
# __Questions__
# - Look at the model documentation: how are each of these models trained?
# - Do some models perform better than others? Do you notice any patterns in their behavior?
# - How much of an impact does scale make on compute?

# In[ ]:


from transformers import pipeline
generator = pipeline('text-generation', model = 'gpt2')


# We give the model a text prompt, define the max length of tokens, and how many examples we want to be generated:

# In[ ]:


outputs = generator("there was a time where ", 
                    max_length=50, do_sample=False)


# The outputs are then returned as a list of dictionaries:

# In[ ]:


outputs


# And we can index specific examples like this:

# In[ ]:


outputs[0].get("generated_text")


# **Question**: try to play with these models and with the inputs you are passing them to highlight the limitations of these models. Can you hack the models to generate linguistically, factually, or ethically debatable behavior?

# ## Decoding parameters

# In our lecture, we have discussed different decoding strategies, and how they may change the characteristics of the text you generate. Our text generation pipeline does greedy decoding if `do_sample` is set to False. What happens if we modulate the decoding strategy? Pick your favorite model, and try to explore what happens if you modify decoding strategies by passing relevant arguments to `generator`.
# 
# NOTE: take a look at this post: https://huggingface.co/blog/how-to-generate to see how these arguments look like, and what strategies they map into. The `textgeneration` pipeline uses `model.generate` under the hood, and the same arguments passed to `model.generate` can be passed to the pipeline.

# In[ ]:


### TODO: add your code here


# ## Text summarization
# Yesterday, we saw how language modeling is the mechanisms that lies behind all natural language generation task. Let's move beyond simple prompt completion, and see how well our models can perform specific language generation tasks.
# 
# Summarization is the prototype text-to-text task, where we are taking an input sequence and trying to train model to accurately produce a summary of the contents. Again, with Huggingface, it's easy to experiment with exisitng text summarization models. Check out their overview [here](https://huggingface.co/tasks/summarization).
# 
# Once more, check out the model zoo for existing models which have been explicitly finetuned for text summarization tasks [here](https://huggingface.co/models?filter=summarization).
# 
# We will use a HuggingFace pipeline for text2text generation. Note that HuggingFace also has a `summarization` pipeline, which is nothing but a subclass of the text2text generation pipeline.
# 
# Here, we start by experimenting with T5 (because we have talked about it in the lecture), but feel free to play with different models. **Note**: you may need to use slightly different instruction prompts for different models.
# 
# __Questions__
# - What do you think of the summaries?
# - Go to the model hub, and try to find some models that are specifically tuned on summarization. How does the performance of models compare based on whether they are fine-tuned on summarization specifically / what they are fine-tuned on (single task? multiple tasks?)?
# - Change your task prefix to perform other text2text generation tasks (e.g., classification, translation, etc)

# In[ ]:


from transformers import pipeline

summarizer = pipeline("text2text-generation", model = "t5-base") 


# We then define a text that we want to summarise. This is the first part of a recent news article in The Guardian - feel free to change this to whatever you want!

# In[ ]:


text = """summarize: Forest conservation and restoration could make a major contribution to tackling the climate crisis as long as greenhouse gas emissions are slashed, according to a study.

By allowing existing trees to grow old in healthy ecosystems and restoring degraded areas, scientists say 226 gigatonnes of carbon could be sequestered, equivalent to nearly 50 years of US emissions for 2022. But they caution that mass monoculture tree-planting and offsetting will not help forests realise their potential.

Humans have cleared about half of Earth’s forests and continue to destroy places such as the Amazon rainforest and the Congo basin that play crucial roles in regulating the planet’s atmosphere.
"""


# As before, generate out summaries using the ```summarizer``` we defined above, which returns a list of dictionaries:

# In[ ]:


summary = summarizer(text, max_length=100)


# And we can get just the summary text:

# In[ ]:


summary[0].get("generated_text") # summary_text


# __Note__
# 
# Here we are only using pretrained models to generate or summarize text. We haven't looked at, for example, how we might train or finetune models on specific tasks.
# 
# If you want to dig into that in a bit more detail, HuggingFace offer many high-quality walkthroughs via [their public Github repo](https://github.com/huggingface/transformers).
# 
# In particular, check out the directory called [Notebooks](https://github.com/huggingface/transformers/tree/main/notebooks) and also the one called [Examples](https://github.com/huggingface/transformers/tree/main/examples). The former are more pedagogical and explain things step-by-step; the latter are more advanced examples of how to fine-tune models effectively. 
# 
# We will mention a couple of additional tricks for efficient model fine-tuning next week!
