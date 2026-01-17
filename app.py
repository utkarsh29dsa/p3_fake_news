import streamlit as st
import pandas as pd
import pickle
import re

vectorization = pickle.load(open("vectorizer.pkl", "rb"))
LR  = pickle.load(open("lr_model.pkl", "rb"))
DTC = pickle.load(open("dtc_model.pkl", "rb"))
rfc = pickle.load(open("rfc_model.pkl", "rb"))
gbc = pickle.load(open("gbc_model.pkl", "rb"))

def wordopt(text):
  #Convert into lowercase
  text=text.lower()
  #Remove URLS
  text=re.sub(r'https?://\S+|www\.\S+','', text)
  #Remove HTML tags
  text=re.sub(r'<.*?>','', text)
  #Remove punctuation
  text=re.sub(r'[^\w\s]', '', text)
  #Remove newline characters
  text=re.sub(r'\n', ' ', text)

  return text

def output_label(n):
    if n == 0:
        return "Fake News"
    else:
        return "True News"

def manualtesting(news):
    testing_news = {'text': [news]}
    new_def_test = pd.DataFrame(testing_news)

    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test['text'])

    pred_lr  = LR.predict(new_xv_test)[0]
    pred_dtc = DTC.predict(new_xv_test)[0]
    pred_rfc = rfc.predict(new_xv_test)[0]
    pred_gbc = gbc.predict(new_xv_test)[0]

    return {
        "Logistic Regression": output_label(pred_lr),
        "Decision Tree": output_label(pred_dtc),
        "Random Forest": output_label(pred_rfc),
        "Gradient Boosting": output_label(pred_gbc),
    }

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("Fake News Detection Model")
st.write("Enter a news article below to check whether it is Fake or True.")

news_input = st.text_area("Paste the news text here", height=250)

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        predictions = manualtesting(news_input)

        st.subheader("Model Predictions")
        for model, result in predictions.items():
            st.write(f"{model}: {result}")
