import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# Add custom CSS to style the top border and text
st.markdown(
    """
    <style>
    .top-border {
        background-color: purple;   /* Border color */
        color: white;  /* Text color */
        font-size: 30px;  /* Text size - increase for larger text */
        font-family: 'Arial', sans-serif;  /* Font style - change to your preferred font */
        font-weight: bold;  /* Text weight */
        padding: 20px 0;  /* Increase padding to make the border taller */
        text-align: center;  /* Center the text horizontally */
        width: 100%;  /* Full width of the page */
    }
    </style>
    """, unsafe_allow_html=True
)

# Add the top border with text
st.markdown('<div class="top-border"> Bankruptcy Prediction App </div>', unsafe_allow_html=True)

# title
st.title(" Bankruptcy Prevention")

# Display the image at the top with a fixed height and width (rectangular shape)
st.image("https://www.durrettebradshaw.com/wp-content/uploads/2018/08/Bankruptcy.jpg", use_container_width = True, width=800)

st.write("This application explores the Bankruptcy Dataset And Uses Machine Learning Models for Prediction.")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = None
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {}

# Upload Dataset
uploaded_file = st.file_uploader("Upload the Bankruptcy Dataset (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
        # Load the dataset
        data = pd.read_excel(uploaded_file, engine='openpyxl')
        st.write("Dataset Loaded Successfully!")
        st.subheader("Raw Data Preview")
        st.dataframe(data)  # Display the DataFrame

        # Shape of Data
        st.write("Dataset Shape:")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")

        # Ensure it is the Bankruptcy dataset by checking specific columns
        required_columns = {'industrial_risk', 'management_risk', 'financial_flexibility','credibility', 'competitiveness', 'operating_risk', 'class'}
        if not required_columns.issubset(data.columns):
           st.error("This is not the Bankruptcy dataset! Please upload the correct file.")
        else:
            # Data Preprocessing
            st.subheader("Data Preprocessing")

            # Drop Duplicates
            if data.duplicated().sum() > 0:
                st.write("Duplicate rows found. Dropping duplicates...")
                data = data.drop_duplicates()
            st.write(f"Dataset shape after dropping duplicates: {data.shape}")

            # Summary Statistics
            if st.checkbox("Show Summary Statistics"):
                 st.markdown('<h4 style="color: navy;"> Summary Statistics: </h4>', unsafe_allow_html=True)
                 st.write(data.describe())

            # Visualizations
            if st.checkbox("Show Visualizations"):
                 st.markdown('<h4 style="color: navy;">Numerical Feature Distribution: </h4>', unsafe_allow_html=True)
                 numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                 for column in numeric_columns:
                     fig, ax = plt.subplots()
                     sns.histplot(data[column], kde=True, ax=ax)
                     st.write(f"Distribution of {column}")
                     st.pyplot(fig)

                 st.markdown('<h4 style="color: navy;">Categorical Feature Count: </h4>', unsafe_allow_html=True)
                 categorical_columns = data.select_dtypes(include=['object']).columns
                 for column in categorical_columns:
                      fig, ax = plt.subplots()
                      sns.countplot(data=data, x=column, palette = "turbo")
                      st.write(f"Count Plot for {column}")
                      st.pyplot(fig)

                 st.markdown('<h4 style="color: navy;">Correlation Heatmap: </h4>', unsafe_allow_html=True)
                 if len(numeric_columns) > 1:
                      fig, ax = plt.subplots()
                      sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
                      st.pyplot(fig)

            # Renaming the dataset  
            cleaned_data = data.iloc[:,:]

            # Assinging featues and target variables 

            features = data.drop(columns = ['class'])
            target = data['class']
            
            # Splitting the data into features and target variables 
            X = cleaned_data.drop(columns=['class']) # target_column is class
            y = cleaned_data['class']

            # Ensure target variable is binary
            if y.dtype == "object":
                 y = y.map({"bankruptcy": 1, "non-bankruptcy": 0})

            # Standardizing numeric features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Model Selection and Training
            st.header("Model Building")
            model_choice = st.selectbox("Select Model for Training", ("Logistic Regression", "KNN"))

            if model_choice == "Logistic Regression":
                 model = LogisticRegression()
            else:
                 n_neighbors = st.slider("Select number of neighbors (for KNN)", min_value=1, max_value=20, value=5)
                 model = KNeighborsClassifier(n_neighbors=n_neighbors)

            if st.button("Train Model"):
                 model.fit(X_train, y_train)
                 st.session_state['model'] = model  # Save the trained model
                 st.session_state['features'] = features  # Save feature names for validation
                 y_pred = model.predict(X_test)
                 y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                 accuracy = accuracy_score(y_test, y_pred)

                 # Save metrics to session state
                 st.session_state['metrics'] = {
                      'accuracy': accuracy,
                      'confusion_matrix': confusion_matrix(y_test, y_pred),
                      'classification_report': classification_report(y_test, y_pred),
                      'precision': precision_score(y_test, y_pred),
                      'recall': recall_score(y_test, y_pred),
                      'f1_score': f1_score(y_test, y_pred),
                      'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A",
                      'roc_curve': roc_curve(y_test, y_proba) if y_proba is not None else None
                 }
                 st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

                 # Confusion Matrix
                 cm = st.session_state['metrics']['confusion_matrix']
                 st.subheader("Confusion Matrix")
                 fig, ax = plt.subplots()
                 sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                 st.pyplot(fig)

                 # Classification Report
                 st.subheader("Classification Report")
                 st.text(st.session_state['metrics']['classification_report'])

            # Display Performance Metrics
            if st.session_state['metrics']:
                 if st.checkbox("Show Performance Metrics"):
                    metrics = st.session_state['metrics']
                    st.write(f"Precision: {metrics['precision']:.2f}")
                    st.write(f"Recall: {metrics['recall']:.2f}")
                    st.write(f"F1 Score: {metrics['f1_score']:.2f}")
                    st.write(f"ROC AUC Score: {metrics['roc_auc']}")

                    # ROC Curve
                    if metrics['roc_curve'] is not None:
                         fpr, tpr, _ = metrics['roc_curve']
                         fig, ax = plt.subplots()
                         ax.plot(fpr, tpr, label="ROC Curve", color="blue")
                         ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                         ax.set_xlabel("False Positive Rate")
                         ax.set_ylabel("True Positive Rate")
                         ax.set_title("ROC Curve")
                         ax.legend()
                         st.pyplot(fig)

# User Input for Prediction
st.sidebar.header("Make Predictions")
            
if st.session_state.get('model') is not None and st.session_state.get('features') is not None:
     user_input = []
     for feature in st.session_state['features']:
          value = st.sidebar.number_input(f"Enter value for {feature}", value=0.0, step=0.1, format="%.1f")
          user_input.append(value)

     user_input = np.array(user_input).reshape(1, -1)

     if st.sidebar.button("Predict"):
          prediction = st.session_state['model'].predict(user_input)
          prediction_proba = st.session_state['model'].predict_proba(user_input)[0]
          result_prob = prediction_proba[prediction[0]]  # Extract probability of predicted class
          st.sidebar.write("Prediction Result:", "Bankruptcy" if prediction[0] == 1 else "No Bankruptcy")
          st.sidebar.write("Prediction Probability:", f"{result_prob:.2f}")
     else:
          st.sidebar.write("Please train the model before making predictions.")

    

          






                

                

                    
                






