# app.py
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from wine_quality_model import load_data, train_model, evaluate_model

# Load data
data = load_data("winequality-red.csv")

# Check if data is None before using it
if data is not None:
    # Page layout
    st.set_page_config(page_title="Wine Quality Prediction App", page_icon="🍷", layout="wide")

    # Title and subtitle
    st.title("Wine Quality Prediction App")
    st.subheader("Predict the quality of your wine!")

    # Sidebar with user input features
    st.sidebar.header("User Input Features")

    # User input fields
    fixed_acidity = st.sidebar.slider("Fixed Acidity", float(data['fixed acidity'].min()), float(data['fixed acidity'].max()), float(data['fixed acidity'].mean()), key="fixed_acidity")
    volatile_acidity = st.sidebar.slider("Volatile Acidity", float(data['volatile acidity'].min()), float(data['volatile acidity'].max()), float(data['volatile acidity'].mean()), key="volatile_acidity")
    citric_acid = st.sidebar.slider("Citric Acid", float(data['citric acid'].min()), float(data['citric acid'].max()), float(data['citric acid'].mean()), key="citric_acid")

    # Add more input fields for other features
    residual_sugar = st.sidebar.slider("Residual Sugar", float(data['residual sugar'].min()), float(data['residual sugar'].max()), float(data['residual sugar'].mean()), key="residual_sugar")
    chlorides = st.sidebar.slider("Chlorides", float(data['chlorides'].min()), float(data['chlorides'].max()), float(data['chlorides'].mean()), key="chlorides")

    # Submit button
    submit_button = st.sidebar.button("Submit")

    # Display dataset summary
    st.write("Successfully Imported Data!")
    st.write("Dataset Shape:", data.shape)

    # Display distribution of wine quality
    st.subheader("Distribution of Wine Quality")
    fig, ax = plt.subplots()
    ax.hist(data['quality'], bins=range(3, 9), align='left', edgecolor='black')
    st.pyplot(fig)


  # Train and evaluate the model
    X_train, X_test, Y_train, Y_test = train_test_split(
        data.drop(['quality'], axis=1).values,
        data['quality'],
        test_size=0.3,
        random_state=7
    )
    model = train_model(X_train, Y_train)

    # Make predictions
    if submit_button:
        # Use X_test for evaluation
        predictions = evaluate_model(model, X_test)
        st.success(f"The predicted wine quality is: {predictions}")

        # Add more predictions for additional input fields
        input_features = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides]
        additional_prediction = evaluate_model(model, [input_features])
        st.success(f"The second predicted wine quality is: {additional_prediction[0] if additional_prediction else 'None'}")

else:
    st.error("Failed to load data. Check the data loading logic.")
 