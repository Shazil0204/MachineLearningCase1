import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

CSV_PATH = "Salary_dataset.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    print(df.head())
    print("Column names:", df.columns.tolist())

    plt.figure()
    plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85)
    plt.title("Years of Experience vs. Salary")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.tight_layout()

    X = df[["YearsExperience"]]
    y = df["Salary"]

    model = LinearRegression().fit(X, y)

    intercept = float(model.intercept_)
    slope = float(model.coef_[0])

    print(f"Model: ŷ = {slope:.3f} * Years of Experience + {intercept:.3f}")
    print(f"Interpretation: For each additional year of experience: +{slope:.2f} in salary.")
    print(f"Intercept (0 years): {intercept:.2f}")

    years = 4.5
    pred = float(model.predict(pd.DataFrame({'YearsExperience':[years]}))[0])
    print(f"Expected salary for {years} years of experience: {pred:.2f}")

    x_min, x_max = df["YearsExperience"].min(), df["YearsExperience"].max()
    X_line = pd.DataFrame({"YearsExperience": np.linspace(x_min, x_max, 200)})
    y_line = model.predict(X_line)

    plt.figure()
    plt.scatter(df["YearsExperience"], df["Salary"], alpha=0.85, label="Data")
    plt.plot(X_line["YearsExperience"], y_line, label="Linear Regression")
    plt.title("Linear Regression Fit")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.tight_layout()
    plt.show()

    joblib.dump(model, 'salary_prediction_model.joblib')
    print("Model saved")

if __name__ == "__main__":
    main()
