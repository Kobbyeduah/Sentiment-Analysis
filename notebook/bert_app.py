# Importing modules
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer , pipeline , AutoConfig
from scipy.special import softmax
import gradio as gr
import numpy as np
import torch

# HuggingFace path where the fine tuned model is placed
model_path = "Henok21/test_trainer"

# Loading the model
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Loading config file
config = AutoConfig.from_pretrained(model_path)

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Creating pipeline
classifier = pipeline("sentiment-analysis" , model , tokenizer = tokenizer)

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
dictionary = {}

def sentiment_analysis(text):

    # Encode the text using the tokenizer
    encoded_input = tokenizer(text, return_tensors='pt')

    # Get the output logits from the model
    output = model(**encoded_input)

    # Get the scores for each class
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Convert the numpy array into a list
    scores = scores.tolist()
    ranking = np.argsort(scores)
    ranking = ranking[::-1]

    for i in range(len(scores)):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
       
        dictionary[l] = float(s)

    return dictionary

# Creating an interface
demo = gr.Interface(
    fn=sentiment_analysis, 
    inputs="text", 
    outputs="label",
    title = "Sentiment Analysis For Covid19 tweet",
    description = "Comment your thought about on the vaccination to covid19. The work is done by fine tunning bert base model."
    )

# Launch your interface
demo.launch(debug = True)