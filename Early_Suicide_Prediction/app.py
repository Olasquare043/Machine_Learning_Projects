import pickle
import streamlit as st # type: ignore
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Set page config
st.set_page_config(
    page_title="Suicide Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("suicide_pipeline.pkl", 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print("Oops!:", e)
pipeline= load_model()

# Title and description
st.title("🧠 Mental Health Risk Assessment")
st.markdown("""
This application uses machine learning to assess suicide risk based on mental health indicators.
**Disclaimer**: This is for informational purposes only and not a substitute for professional medical advice.
""")

# Create sidebar for navigation
st.sidebar.title("Navigation")
page=st.sidebar.radio("Select Page",["Prediction", "About Model", "Batch Prediction"])

# logic for selected page
if page == "Prediction":
    st.header("Individual Risk Assessment")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=10, max_value=80, value=20)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("Stress & Mental State")
        stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
        depression_level = st.selectbox("Depression Level", ["Sometimes", "Often", "Always"])
        anxiety_level = st.selectbox("Anxiety Level", ["Sometimes", "Often", "Always"])
    
    with col2:
        st.subheader("Relationships & Support")
        academic_perf = st.selectbox("Academic Performance", ["Poor", "Average", "Good", "Excellent"])
        health_condition = st.selectbox("Health Condition", ["Abnormal", "Fair", "Normal"])
        relationship = st.selectbox("Relationship Status", ["Single", "In a relationship", "Breakup"])
        
        st.subheader("Support & History")
        family_problem = st.selectbox("Family Problem", ["No", "Financial", "Parental conflict"])
        mental_support = st.selectbox("Mental Support", ["Family", "Friends", "loneliness", "Others"])
        self_harm = st.selectbox("Self Harm History", ["Yes", "No"])
    
    # Create input dataframe
    if st.button("🔍 Assess Risk", key="predict_btn"):
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Stress Level': [stress_level],
            'Academic Performance': [academic_perf],
            'Health Condition': [health_condition],
            'Relationship Condition': [relationship],
            'Family Problem': [family_problem],
            'Depression Level': [depression_level],
            'Anxiety Level': [anxiety_level],
            'Mental Support': [mental_support],
            'Self Harm Story': [self_harm]
        })
        
        # Make prediction
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        
        # Map predictions to labels
        labels = ['Never Thought', 'Thought', 'Attempted']
        predicted_label = labels[prediction]
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Assessment Results")
        
        # Color coding based on risk
        if prediction == 0:
            color = "green"
            emoji = "✅"
        elif prediction == 1:
            color = "orange"
            emoji = "⚠️"
        else:
            color = "red"
            emoji = "🚨"
        
        st.markdown(f"<h2 style='color: {color};'>{emoji} {predicted_label}</h2>", unsafe_allow_html=True)
        
        # Show probabilities
        st.subheader("Risk Probabilities:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Never Thought", f"{probabilities[0]:.1%}")
        with col2:
            st.metric("Thought", f"{probabilities[1]:.1%}")
        with col3:
            st.metric("Attempted", f"{probabilities[2]:.1%}")
        
        # Visualization
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        bars = ax.bar(labels, probabilities, color=['green', 'orange', 'red'], alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title('Risk Assessment Probabilities')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2%}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction >= 1:
            st.warning("Please seek professional mental health support. Here are resources:")
            st.markdown("""
            - **National Suicide Prevention Lifeline**: 1-800-273-8255
            - **Crisis Text Line**: Text HOME to 741741
            - **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/
            """)
        else:
            st.success("Continue maintaining your mental health with regular check-ups and support systems.")

elif page == "About Model":
    st.header("📈 Model Information")
    
    st.subheader("Model Architecture")
    st.write("""
    - **Algorithm**: Random Forest Classifier with 100 estimators
    - **Training Data**: 843 samples
    - **Test Data**: 211 samples
    - **Overall Accuracy**: 85.78%
    """)
    
    st.subheader("Feature Engineering")
    st.write("""
    **Ordinal Encoded Features** (features with natural order):
    - Stress Level: Low → Moderate → High
    - Depression Level: Sometimes → Often → Always
    - Anxiety Level: Sometimes → Often → Always
    - Academic Performance: Poor → Average → Good → Excellent
    - Health Condition: Abnormal → Fair → Normal
    
    **One-Hot Encoded Features** (categorical):
    - Gender, Self Harm Story, Relationship Condition, Family Problem, Mental Support
    """)
    
    st.subheader("Model Performance")
    st.write("""
    **Classification Metrics**:
    - Never Thought: Precision=0.8800, Recall=0.9244, F1=0.9016
    - Thought: Precision=0.8000, Recall=0.7273, F1=0.7619
    - Attempted: Precision=0.8846, Recall=0.8846, F1=0.8846
    """)
    
    st.warning("⚠️ Disclaimer: This model is for educational purposes. Always consult with mental health professionals.")

elif page == "Batch Prediction":
    st.header("📁 Batch Predictions")
    st.write("Upload a CSV file with multiple records for prediction")
    
    uploaded_file = st.file_uploader("Choose CSV file", type='csv')
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.write(f"Uploaded {len(df)} records")
        st.dataframe(df.head())
        
        if st.button("Run Batch Prediction"):
            predictions = pipeline.predict(df)
            probabilities = pipeline.predict_proba(df)
            
            labels = ['Never Thought', 'Thought', 'Attempted']
            df['Prediction'] = [labels[p] for p in predictions]
            df['Confidence'] = probabilities.max(axis=1)
            
            st.subheader("Predictions")
            st.dataframe(df[['Prediction', 'Confidence']])
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )