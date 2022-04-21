# Importing Necessary modules
from fastapi import FastAPI, Form
import uvicorn
import pickle
from pydantic import BaseModel
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Declaring our FastAPI instance
app = FastAPI()

# Defining path operation for root endpoint
@app.get('/')
def greetings():
    return 'Welcome to my Sentiment Analysis Project!'

model = pickle.load(open("log_final_model.sav", 'rb'))
cVectorizer = pickle.load(open("cVectorizer.sav", 'rb'))
dict_sentiment = pickle.load(open("dictionary.sav", 'rb'))

def clean_up_sentence(sentence:str):
    """
    It takes a sentence, tokenizes it, lemmatizes it, and returns it
    
    :param sentence: The sentence to be processed
    :type sentence: str
    :return: A list of words
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence

def get_text(text:str):
    """
    It takes a string, removes all non-alphabetic characters, converts it to lowercase, and returns the
    result
    
    :param text: The text that you want to clean up
    :type text: str
    :return: A string of text that has been cleaned up.
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = clean_up_sentence(text)
    return text

class request_body(BaseModel):
    text:str

@app.post('/predict')
def predict(text:str = Form(...)):
    """
    The function takes in a text, transforms it into a vector, and then predicts the sentiment of the
    text
    
    :param text: The text to be classified
    :type text: str
    :return: The sentiment of the text.
    """
    text = get_text(text)
    data = cVectorizer.transform([text]).toarray()
    prediction = model.predict(data)[0]
    return dict_sentiment.get(prediction)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
