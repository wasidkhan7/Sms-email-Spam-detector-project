import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk

#nltk.download('punkt_tab') 
#nltk.download('stopwords') 

ps=PorterStemmer()

def transform_text(text):
  # for lower case 
  text=text.lower()  

  # tokenizztion
  text=nltk.word_tokenize(text)  

  # Leaving just alpha numeric  characters
  y=[]
  for i in   text:
    if i.isalnum():   
      y.append(i)

  # removing special char and punctuations
  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation: 
      y.append(i)

  # removing stemming words
  text=y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation: 
      y.append(ps.stem(i))

  return " ".join(y)  


# Loadingfiles
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

# Tile
st.title(" SMS-Email CLassifier: ")

user_input=st.text_area(" Enter The Message; Get to know IF spam OR Ham: ")

if st.button("Predict"):
  
    # 1. preprocessing
    transformed_sms=transform_text(user_input)  # tranforms the text from the 4 steps of preprocessing

    # 2. vectorize
    vector_input=tfidf.transform([transformed_sms])  # it breaks messages into words, tokenize form , helped from "vectorizer.pkl"

    # 3. predict
    result=model.predict(vector_input)[0]

    # 4. display
    if result==0:
        st.header(" NOt Spam ")
    else:
        st.header(" Spam ")
