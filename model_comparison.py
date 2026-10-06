import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------
# Linear Regression
# --------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_prediction
)

linear_r2 = r2_score(
    y_test,
    linear_prediction
)


# --------------------------------
# Random Forest
# --------------------------------

random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train,
    y_train
)

random_forest_prediction = random_forest_model.predict(
    X_test
)

random_forest_mae = mean_absolute_error(
    y_test,
    random_forest_prediction
)

random_forest_r2 = r2_score(
    y_test,
    random_forest_prediction
)


# --------------------------------
# Display Results
# --------------------------------

print("\n====================================")
print("       MODEL COMPARISON")
print("====================================")

print("\nLinear Regression")
print("-------------------------")
print("MAE :", round(linear_mae, 2))
print("R2  :", round(linear_r2, 2))


print("\nRandom Forest Regression")
print("-------------------------")
print("MAE :", round(random_forest_mae, 2))
print("R2  :", round(random_forest_r2, 2))


# --------------------------------
# Select Best Model
# --------------------------------

if random_forest_r2 > linear_r2:

    print("\nBEST MODEL: Random Forest Regression")

else:

    print("\nBEST MODEL: Linear Regression")