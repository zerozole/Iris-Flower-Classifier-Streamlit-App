import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# -----------------------
#  Page Configuration
# -----------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌺",
    layout="centered",
)

st.title("🌺 Iris Flower Classifier")
st.write(
    """
    This interactive app predicts the species of an **Iris flower** based on its sepal and petal measurements.  
    Move the sliders in the sidebar to adjust feature values and see real-time predictions made by a trained **Random Forest** model.
    """
)

# -----------------------
#  Load and Cache Data
# -----------------------
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()

# -----------------------
#  Train the Model
# -----------------------
@st.cache_resource
def train_model(data):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data.iloc[:, :-1], data['species'])
    return model

model = train_model(df)

# -----------------------
#  Sidebar Inputs
# -----------------------
st.sidebar.header("Input Flower Measurements ")
st.sidebar.markdown("Adjust the sliders to change input values:")

sepal_length = st.sidebar.slider(
    "Sepal length (cm)",
    float(df['sepal length (cm)'].min()),
    float(df['sepal length (cm)'].max()),
    float(df['sepal length (cm)'].mean()),
)

sepal_width = st.sidebar.slider(
    "Sepal width (cm)",
    float(df['sepal width (cm)'].min()),
    float(df['sepal width (cm)'].max()),
    float(df['sepal width (cm)'].mean()),
)

petal_length = st.sidebar.slider(
    "Petal length (cm)",
    float(df['petal length (cm)'].min()),
    float(df['petal length (cm)'].max()),
    float(df['petal length (cm)'].mean()),
)

petal_width = st.sidebar.slider(
    "Petal width (cm)",
    float(df['petal width (cm)'].min()),
    float(df['petal width (cm)'].max()),
    float(df['petal width (cm)'].mean()),
)

# -----------------------
#  Make Prediction
# -----------------------
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
probabilities = model.predict_proba(input_data)

predicted_species = target_names[prediction[0]]

# -----------------------
#  Display Results
# -----------------------
st.subheader("🔍 Prediction Result")
st.success(f"**Predicted Species:** {predicted_species}")

st.markdown("#### Prediction Confidence:")
prob_df = pd.DataFrame(probabilities, columns=target_names)
st.bar_chart(prob_df.T)

# -----------------------
#  Show Data
# -----------------------
with st.expander("View Sample of Iris Dataset"):
    st.dataframe(df.sample(10))
