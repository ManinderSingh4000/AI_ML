import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    # In a real scenario, you'd load data from a CSV file or database
    # For this example, we'll create some sample data
    data = pd.DataFrame({
        'student_id': range(1, 101),
        'cgpa': np.random.uniform(5.0, 10.0, 100),
        'internships': np.random.randint(0, 3, 100),
        'projects': np.random.randint(1, 6, 100),
        'placed': np.random.choice([0, 1], 100, p=[0.3, 0.7])
    })
    return data

# Train the model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

# Main app
def main():
    st.title("Student Placement System")
    
    # Load data
    data = load_data()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Add Student", "Predict Placement"])
    
    if page == "Dashboard":
        st.header("Student Placement Dashboard")
        st.write(data)
        
        st.subheader("Placement Statistics")
        placement_rate = data['placed'].mean() * 100
        st.write(f"Placement Rate: {placement_rate:.2f}%")
        
        st.subheader("CGPA Distribution")
        st.histogram_chart(data['cgpa'])
        
    elif page == "Add Student":
        st.header("Add New Student")
        
        new_student_id = st.number_input("Student ID", min_value=1, step=1)
        new_cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, step=0.1)
        new_internships = st.number_input("Number of Internships", min_value=0, step=1)
        new_projects = st.number_input("Number of Projects", min_value=0, step=1)
        
        if st.button("Add Student"):
            new_student = pd.DataFrame({
                'student_id': [new_student_id],
                'cgpa': [new_cgpa],
                'internships': [new_internships],
                'projects': [new_projects],
                'placed': [0]  # Initially set as not placed
            })
            data = pd.concat([data, new_student], ignore_index=True)
            st.success("Student added successfully!")
            st.write(new_student)
        
    elif page == "Predict Placement":
        st.header("Predict Student Placement")
        
        # Train the model
        X = data[['cgpa', 'internships', 'projects']]
        y = data['placed']
        model, scaler, accuracy = train_model(X, y)
        
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Input for prediction
        pred_cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, step=0.1)
        pred_internships = st.number_input("Number of Internships", min_value=0, step=1)
        pred_projects = st.number_input("Number of Projects", min_value=0, step=1)
        
        if st.button("Predict Placement"):
            input_data = np.array([[pred_cgpa, pred_internships, pred_projects]])
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)
            
            if prediction[0] == 1:
                st.success("The student is likely to be placed!")
            else:
                st.warning("The student might need to improve their profile for better placement chances.")

if __name__ == "__main__":
    main()

