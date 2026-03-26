import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---- YOUR FUNCTION (UNCHANGED) ----
def naive_bayes(X,y):
  X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
  GNB=GaussianNB()

  GNB.fit(X_train,y_train)
  y_pred=GNB.predict(X_test)

  accuracy=accuracy_score(y_test,y_pred)
  confusion=confusion_matrix(y_test,y_pred)

  return accuracy,confusion


# ---- STREAMLIT UI ----
st.title("Naive Bayes Classifier App")

st.write("Upload your dataset (CSV). Make sure the last column is the target.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Select target column
    target_column = st.selectbox("Select Target Column", data.columns)

    if st.button("Run Naive Bayes"):
        X = data.drop(columns=[target_column])
        y = data[target_column]

        accuracy, cm = naive_bayes(X, y)

        st.subheader("Results")
        st.write("Accuracy:", accuracy)
        st.write("Confusion Matrix:")
        st.write(cm)
