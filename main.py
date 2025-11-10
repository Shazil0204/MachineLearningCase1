import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CSV_PATH = "Salary_dataset.csv"

def evaluate(model, X_tr, y_tr, X_te, y_te, name="Model"):
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    mse_tr = mean_squared_error(y_tr, y_pred_tr)
    mse_te = mean_squared_error(y_te, y_pred_te)
    r2_tr = r2_score(y_tr, y_pred_tr)
    r2_te = r2_score(y_te, y_pred_te)
    print(f"\n=== {name} ===")
    print(f"Train -> MSE: {mse_tr:.3f}, R^2: {r2_tr:.3f}")
    print(f"Test -> MSE: {mse_te:.3f}, R^2: {r2_te:.3f}")

# def main():
#     df = pd.read_csv(CSV_PATH)
    
#     # Clean column names by stripping whitespace
#     df.columns = df.columns.str.strip()
    
#     print(df.head())
#     print("Column names:", df.columns.tolist())

#     plt.figure()
#     plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85)
#     plt.title("Years of Experience vs. Salary")
#     plt.xlabel("Years of Experience")
#     plt.ylabel("Salary")
#     plt.tight_layout()

#     X = df[["YearsExperience"]]
#     y = df["Salary"]

#     model = LinearRegression().fit(X, y)

#     intercept = float(model.intercept_)
#     slope = float(model.coef_[0])

#     print(f"Model: ŷ = {slope:.3f} * Years of Experience + {intercept:.3f}")
#     print(f"Interpretation: For each additional year of experience: +{slope:.2f} in salary.")
#     print(f"Intercept (0 years): {intercept:.2f}")

#     years = 4.5
#     pred = float(model.predict(pd.DataFrame({'YearsExperience':[years]}))[0])
#     print(f"Expected salary for {years} years of experience: {pred:.2f}")

#     x_min, x_max = df["YearsExperience"].min(), df["YearsExperience"].max()
#     X_line = pd.DataFrame({"YearsExperience": np.linspace(x_min, x_max, 200)})
#     y_line = model.predict(X_line)

#     plt.figure()
#     plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85, label="Data")
#     plt.plot(X_line["YearsExperience"], y_line, label="Linear Regression")
#     plt.title("Linear Regression Fit")
#     plt.xlabel("Years of Experience")
#     plt.ylabel("Salary")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     joblib.dump(model, 'salary_prediction_model.joblib')
#     print("Model saved")

def main():
    df = pd.read_csv(CSV_PATH)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()

    print(df.head())

    X = df[["YearsExperience"]]
    y = df["Salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    evaluate(linear, X_train, y_train, X_test, y_test, "Linear Regression")

    # Visualizer
    x_min = X["YearsExperience"].min()
    x_max = X["YearsExperience"].max()

    X_line = pd.DataFrame({"YearsExperience": np.linspace(x_min, x_max, 100)})
    y_line = linear.predict(X_line)

    plt.figure()
    plt.scatter(X_train, y_train, label="Training data")
    plt.scatter(X_test, y_test, label="Test data", alpha=0.7)
    plt.plot(X_line, y_line, color="red", label="Prediction")
    plt.title("Linear Regression Fit")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()

    joblib.dump(linear, 'salary_prediction_model.joblib')
    print("Model saved")

if __name__ == "__main__":
    main()
