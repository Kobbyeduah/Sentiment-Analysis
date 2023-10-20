# Sentiment-Analysis

## Description

Sentiment analysis is the task of identifying and extracting the emotional tone and attitude of a text, such as positive, negative, or neutral. In this project, I have used the [Hugging Face] library, which provides a collection of state-of-the-art natural language processing models and tools. I have leveraged the power of Hugging Face to perform sentiment analysis on various texts, such as tweets, reviews, comments, etc. Explored the different aspects of sentiment analysis, such as data preprocessing, model selection, evaluation metrics, and visualization. Demonstrated your skills and knowledge in applying Hugging Face to solve a real-world problem using sentiment analysis.

# Installation


### Manual Setup

For manual installation, you need to have [`Python3`](https://www.python.org/) on your system. Then you can clone this repo and being at the repo's `root :: friendly_web_interface_for_ML_models> ...`  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate;
        python -m pip install -q --upgrade pip;
        python -m pip install -qr requirements.txt
  
- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate;
        python -m pip install -q --upgrade pip;
        python -m pip install -qr requirements.txt
The both long command-lines have a same structure, they pipe multiple commands using the symbol **;** but you may manually execute them one after another.

1. **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
2. **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
3. **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
4. **Install the required libraries/packages** listed in the `requirements.txt` file so that it will be allow to import them into the python's scripts and notebooks without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

- Run the demo apps (Make sure you are at the right directory):
        
- Gradio:
    
     gradio  bert_app;
     gradio  roberta_app;
  

