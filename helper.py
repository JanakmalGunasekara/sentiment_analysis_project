import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# 1. Load the trained model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

# 2. Load the stopwords from your local corpora
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

# 3. Load the vocabulary tokens (Must match the order in your notebook)
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):
    # Creating a temporary DataFrame to use pandas string operations
    data = pd.DataFrame([text], columns=['tweet'])
    
    # Standard cleaning steps
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data['tweet'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    
    # IMPORTANT: Return only the string value, not the whole Series
    return data["tweet"].iloc[0]

def vectorizer(sentence):
    # Create a zero vector of the size of your vocabulary
    sentence_vector = np.zeros(len(tokens))
    
    # Split the preprocessed sentence into a list of words
    words = sentence.split()
    
    # Mark 1 if the vocabulary word exists in the input sentence
    for i in range(len(tokens)):
        if tokens[i] in words:
            sentence_vector[i] = 1 
            
    # Convert to a 2D array as required by Scikit-learn models (Shape: 1 x N)
    return np.asarray([sentence_vector], dtype=np.float32)

def get_prediction(vectorized_txt):
    # Get the prediction array from the model
    prediction = model.predict(vectorized_txt)
    
    # Print the raw prediction to the terminal for debugging
    print(f"DEBUG - Model output: {prediction[0]}")
    
    # Based on your notebook: if 1 is Negative and 0 is Positive
    if prediction[0] == 1:
        return 'negative'
    else:
        return 'positive'