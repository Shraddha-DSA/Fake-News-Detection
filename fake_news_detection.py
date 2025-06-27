import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")
true['label'] = 1
fake['label'] = 0
df = pd.concat([true,fake],ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text
df['text'] = df['title'] + " "+df['text']
df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec,y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy: ",accuracy_score(y_test,y_pred))
print("\n")
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("\n")
print("Classification Report: \n",classification_report(y_test,y_pred))

input("\nPress Enter to exit...")

with open("fake_news_model.pkl","wb") as f:
    pickle.dump(model,f)
with open("vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)