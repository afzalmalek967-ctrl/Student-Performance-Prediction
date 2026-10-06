import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("dataset/student_data.csv")

# Input features
X = data[
    [
        "attendance",
        "study_hours",
        "internal_marks",
        "assignment_marks",
        "previous_marks",
        "practical_marks"
    ]
]

# Target
y = data["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create ML model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "model/student_performance_model.pkl")


# Prediction
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Training Completed!")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Test prediction
student = pd.DataFrame([{
    "attendance": 85,
    "study_hours": 4,
    "internal_marks": 72,
    "assignment_marks": 80,
    "previous_marks": 68,
    "practical_marks": 75
}])

prediction = model.predict(student)

print("\nSample Student:")
print("Attendance: 85%")
print("Study Hours: 4")
print("Internal Marks: 72")
print("Assignment Marks: 80")
print("Previous Marks: 68")
print("Practical Marks: 75")

print("\nPredicted Final Marks:", round(prediction[0], 2))