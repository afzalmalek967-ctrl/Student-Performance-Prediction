import pandas as pd
import random

random.seed(42)

students = []

for i in range(300):

    attendance = random.randint(50, 100)
    study_hours = round(random.uniform(1, 8), 1)
    internal_marks = random.randint(40, 95)
    assignment_marks = random.randint(45, 100)
    previous_marks = random.randint(40, 95)
    practical_marks = random.randint(45, 100)

    # Calculate final marks with some realistic variation
    final_marks = (
        attendance * 0.10
        + study_hours * 2
        + internal_marks * 0.25
        + assignment_marks * 0.15
        + previous_marks * 0.20
        + practical_marks * 0.15
    )

    # Add small random variation
    final_marks += random.uniform(-4, 4)

    # Keep marks between 35 and 95
    final_marks = max(35, min(95, final_marks))

    students.append([
        attendance,
        study_hours,
        internal_marks,
        assignment_marks,
        previous_marks,
        practical_marks,
        round(final_marks, 2)
    ])


# Create DataFrame

data = pd.DataFrame(
    students,
    columns=[
        "attendance",
        "study_hours",
        "internal_marks",
        "assignment_marks",
        "previous_marks",
        "practical_marks",
        "final_marks"
    ]
)


# Save dataset

data.to_csv(
    "dataset/student_data.csv",
    index=False
)


print("Dataset created successfully!")
print("Total students:", len(data))

print("\nFirst 10 records:")
print(data.head(10))