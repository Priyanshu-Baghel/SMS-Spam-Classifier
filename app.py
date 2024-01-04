import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# loading tfidf and model with help of pickle
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# set title of website
st.title("SMS Spam Classifier")

# take input from user
st.header("Enter the message")
input_message = st.text_area(" ")

# preprocessing of input message

# objet of PorterStemmer class
ps = PorterStemmer()

def transform_sms(sms):
    sms = sms.lower()
    sms = nltk.word_tokenize(sms)

    y = []
    for i in sms:
        if i.isalnum():
            y.append(i)

    sms = y[:]
    y.clear()

    for i in sms:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    sms = y[:]
    y.clear()

    for i in sms:
        y.append(ps.stem(i))

    return " ".join(y)

if st.button('Predict'):

    # preprocessing of the input message
    transformed_message = transform_sms(input_message)

    # vectorize

    vector_input = tfidf.transform([transformed_message])

    # predict
    # Assuming vector_input is your sparse input data
    dense_input = vector_input.toarray()
    result = model.predict(dense_input)[0]

    # Display

    if result == 1:
        st.header("Spam")
    else:
        st.header("Ham")