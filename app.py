from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("model/student_performance_model.pkl")

# Load dataset
dataset = pd.read_csv("dataset/student_data.csv")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Student details
    student_name = request.form["student_name"]
    enrollment = request.form["enrollment"]

    # Performance details
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    internal_marks = float(request.form["internal_marks"])
    assignment_marks = float(request.form["assignment_marks"])
    previous_marks = float(request.form["previous_marks"])
    practical_marks = float(request.form["practical_marks"])

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "attendance": attendance,
        "study_hours": study_hours,
        "internal_marks": internal_marks,
        "assignment_marks": assignment_marks,
        "previous_marks": previous_marks,
        "practical_marks": practical_marks
    }])

    # Prediction
    prediction = model.predict(input_data)

    predicted_marks = round(prediction[0], 2)

    # Performance category
    if predicted_marks >= 80:
        performance = "Excellent"
    elif predicted_marks >= 70:
        performance = "Good"
    elif predicted_marks >= 60:
        performance = "Average"
    else:
        performance = "Needs Improvement"

    # Pass / Fail
    if predicted_marks >= 40:
        result = "PASS"
    else:
        result = "FAIL"

    # Recommendation
    if predicted_marks >= 80:
        recommendation = "Excellent performance! Keep up the good work."
    elif predicted_marks >= 70:
        recommendation = "Good performance. Focus on improving weak areas."
    elif predicted_marks >= 60:
        recommendation = "Average performance. Increase study hours and practice."
    else:
        recommendation = "Performance needs improvement. Focus on attendance, study and assignments."

    return render_template(
        "result.html",
        prediction=predicted_marks,
        performance=performance,
        result=result,
        recommendation=recommendation,
        student_name=student_name,
        enrollment=enrollment,
        attendance=attendance,
        study_hours=study_hours,
        internal_marks=internal_marks,
        assignment_marks=assignment_marks,
        previous_marks=previous_marks,
        practical_marks=practical_marks
    )


@app.route("/dataset")
def dataset_analysis():

    total_students = len(dataset)

    average_marks = round(dataset["final_marks"].mean(), 2)

    highest_marks = round(dataset["final_marks"].max(), 2)

    lowest_marks = round(dataset["final_marks"].min(), 2)

    return render_template(
        "dataset.html",
        total_students=total_students,
        average_marks=average_marks,
        highest_marks=highest_marks,
        lowest_marks=lowest_marks
    )


if __name__ == "__main__":
    app.run(debug=True)