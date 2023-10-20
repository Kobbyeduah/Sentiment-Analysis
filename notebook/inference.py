

# Import a module
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer , pipeline , AutoConfig
import numpy as np

import gradio as gr
from scipy.special import softmax
import torch

# Loading requirements from Hugging Face
# HuggingFace path where the fine tuned model is placed
model_path = "Henok21/test_trainer"

# Loading the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

config = AutoConfig.from_pretrained(model_path)

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Creating pipeline
calssifier = pipeline("sentiment-analysis" , model , tokenizer = tokenizer)

# Preparing gradio app
# Preprocessor Function
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Configuring the outputs
config.id2label = {0: 'NEGATIVE', 1: 'NEUTRAL', 2: 'POSITIVE'}
config.label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

# creating a function used for gradio app
# Creating dictionary
dictionary = {}

def sentiment_analysis(text):

    # Encode the text using the tokenizer
    encoded_input = tokenizer(text, return_tensors='pt')
    # Get the output logits from the model
    output = model(**encoded_input)

    # Your code to get the scores for each class
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Convert the numpy array into a list
    scores = scores.tolist()
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    for i in range(len(scores)):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
       
        # Convert the numpy float32 object into a float
        dictionary[l] = float(s)

    # Return the dictionary as the response content
    return dictionary

# Create your interface
demo = gr.Interface(
    fn=sentiment_analysis, 
    inputs="text", 
    outputs="label"
    )

# Launch your interface
demo.launch(debug = True)