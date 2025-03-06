import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("gmm_model.pkl", "rb") as file:
    gmm = pickle.load(file)

# Cluster Recommendations
cluster_recommendations = {
    0: '''Safe Zone 🟢:
          Characteristics: Normal weight, regular periods, minimal PCOS symptoms.
          Recommendation: Maintain a balanced diet, continue regular exercise, and focus on overall wellness.''',
    1: '''Low Risk 🟡:
          Characteristics: Mild weight gain, occasional irregular periods, some PCOS symptoms like acne or hair loss.
          Recommendation: Adopt a low-glycemic diet, reduce processed food intake, and include moderate exercise.''',
    2: '''Moderate Risk 🟠:
          Characteristics: Noticeable weight gain, irregular periods, increased symptoms like facial hair growth or skin darkening.
          Recommendation: Focus on weight management, insulin resistance control (low GI diet), and regular physical activity.''',
    3: '''High Risk 🔴:
          Characteristics: Significant PCOS symptoms, frequent period irregularities, high weight gain, and fast food consumption.
          Recommendation: Immediate lifestyle changes—strict low-GI diet, high-fiber foods, strength training, and medical consultation if needed.
          '''
}

# Streamlit UI
st.title("PCOS Health Segmentation & Recommendations")
st.write("Enter your details to get personalized health recommendations.")

# Collect user inputs
Age = st.slider("Age", 15, 50, 25)
Weight = st.number_input("Weight (kg)", min_value=30, max_value=120, value=60)
Height = st.number_input("Height (cm)", min_value=130, max_value=190, value=160)
Period_freq = st.selectbox("Period Frequency", [1, 2, 3, 4, 5])

# Blood Group Selection & One-Hot Encoding
blood_g = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-"])
blood_group_mapping = {
    "A+":  [1, 0, 0, 0, 0, 0],
    "A-":  [0, 1, 0, 0, 0, 0],
    "B+":  [0, 0, 1, 0, 0, 0],
    "B-":  [0, 0, 0, 1, 0, 0],
    "AB+": [0, 0, 0, 0, 1, 0],
    "AB-": [0, 0, 0, 0, 0, 1],
    
}
Blood_Group_Encoded = blood_group_mapping[blood_g]  # One-hot encoding for selected blood group

# Other Features
Weight_gain = st.checkbox("Experiencing weight gain?")
Facial_Hair_Growth = st.checkbox("Increased facial hair growth?")
Skin_Darkening = st.checkbox("Skin darkening in certain areas?")
Fast_Food = st.checkbox("Frequent fast food consumption?")
Hair_Loss = st.checkbox("Are you experiencing hair loss?")
Acne = st.checkbox("Do you have acne?")
Exercise = st.checkbox("Do you exercise regularly?")
PCOS_Diagnosed = st.checkbox("Have you been diagnosed with PCOS/PCOD?")
Mood_Swings = st.checkbox("Do you have frequent mood swings?")
Regular_Periods = st.checkbox("Are your periods regular?")
Period_Duration = st.number_input("For how many days do you have your periods?", min_value=1, max_value=10, value=5)

# Convert boolean inputs to numerical values & normalize
input_data = np.array([
    Age / 50, Weight / 120, Height / 190, Period_freq,
    int(Weight_gain), int(Facial_Hair_Growth), int(Skin_Darkening), int(Fast_Food),
    int(Hair_Loss), int(Acne), int(Exercise), int(PCOS_Diagnosed), int(Mood_Swings),
    int(Regular_Periods), Period_Duration / 10  # Normalize Period Duration
] + Blood_Group_Encoded).reshape(1, -1)  # Append One-Hot Encoded Blood Group

# Predict cluster
if st.button("Get Recommendations"):
    cluster_label = gmm.predict(input_data)[0]
    st.subheader(f"You belong to Cluster {cluster_label}")
    st.write(cluster_recommendations.get(cluster_label, "No recommendation available for this cluster."))

