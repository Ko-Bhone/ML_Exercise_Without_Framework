import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.population = None
        self.profit = None
        self.X = None
        self.y = None
        self.theta = None
        self.m = 0
        self.n = 0
        self.cost_history = []
        self.mean = None
        self.std = None

    # Load Dataset
    def load_data(self):
        self.df = pd.read_csv(self.file_path, header=None, names=["Population", "Profit"])
        print("\n" + "=" * 60)
        print("DATASET LOADED")
        print("=" * 60)
        print("\nFirst 5 Rows")
        print(self.df.head())
        print("\nLast 5 Rows")
        print(self.df.tail())
        print("\nDataset Shape")
        print(self.df.shape)
        print("\nColumn Names")
        print(self.df.columns)
        print("\nData Types")
        print(self.df.dtypes)
        print("\nDataset Information")
        self.df.info()
        print("\nStatistical Summary")
        print(self.df.describe())
        print("\nMissing Values")
        print(self.df.isnull().sum())
        print("\nDuplicate Rows")
        print(self.df.duplicated().sum())

    # Data Cleaning
    def clean_data(self):
        print("\n" + "=" * 60)
        print("DATA CLEANING")
        print("=" * 60)
        # Remove Duplicate Rows
        duplicate_count = self.df.duplicated().sum()
        if duplicate_count > 0:
            self.df.drop_duplicates(inplace=True)
            print(f"Removed Duplicate Rows : {duplicate_count}")
        else:
            print("Duplicate Rows : None")
        # Remove Missing Values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            self.df.dropna(inplace=True)
            print(f"Removed Missing Values : {missing_count}")
        else:
            print("Missing Values : None")
        # Convert Data Type
        self.df["Population"] = pd.to_numeric(self.df["Population"], errors="coerce")
        self.df["Profit"] = pd.to_numeric(self.df["Profit"], errors="coerce")
        # Remove invalid values after conversion
        self.df.dropna(inplace=True)
        # Reset Index
        self.df.reset_index(drop=True, inplace=True)
        # Outlier Detection (IQR Method)
        print("\nChecking Outliers...\n")
        for column in self.df.columns:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - (1.5 * IQR)
            upper_limit = Q3 + (1.5 * IQR)
            outliers = self.df[(self.df[column] < lower_limit) | (self.df[column] > upper_limit)]
            print(f"{column}")
            print(f"Outliers : {len(outliers)}")
        print("\nData Cleaning Completed.")
        print("\nDataset Shape After Cleaning")
        print(self.df.shape)
        print("\nFirst 5 Rows")
        print(self.df.head())

    # Prepare Data
    def prepare_data(self):
        print("\n" + "=" * 60)
        print("PREPARING DATA")
        print("=" * 60)
        # Extract Features and Target
        self.population = self.df["Population"].to_numpy().reshape(-1, 1)
        self.profit = self.df["Profit"].to_numpy().reshape(-1, 1)
        # Number of Samples
        self.m = self.population.shape[0]
        # Number of Features
        self.n = self.population.shape[1]
        # Add Bias Column
        # X Shape => (m, 2)
        bias = np.ones((self.m, 1))
        self.X = np.hstack((bias, self.population))

        # Target Vector
        # y Shape => (m, 1)
        self.y = self.profit
        # Initialize Theta
        # theta Shape => (2, 1)
        self.theta = np.zeros((self.X.shape[1], 1))
        # Display Shapes
        print(f"Number of Samples  : {self.m}")
        print(f"Number of Features : {self.n}")
        print("\nMatrix Shapes")
        print(f"X Shape     : {self.X.shape}")
        print(f"y Shape     : {self.y.shape}")
        print(f"Theta Shape : {self.theta.shape}")
        print("\nFirst 5 Rows of X")
        print(self.X[:5])
        print("\nFirst 5 Rows of y")
        print(self.y[:5])
        print("\nData Preparation Completed.")

    # Complete Dataset Pipeline
    def dataset_pipeline(self):
        self.load_data()
        self.clean_data()
        self.prepare_data()

    # Compute Cost Function
    def compute_cost(self):
        predictions = self.X @ self.theta
        errors = predictions - self.y
        cost = (1 / (2 * self.m)) * np.sum(errors ** 2)
        return cost

    # Compute Gradient
    def compute_gradient(self):
        predictions = self.X @ self.theta
        errors = predictions - self.y
        gradient = (1 / self.m) * (self.X.T @ errors)
        return gradient

    # Gradient Descent
    def gradient_descent(self, learning_rate=0.01, iterations=1500, tolerance=1e-8):
        print("\n" + "=" * 60)
        print("GRADIENT DESCENT")
        print("=" * 60)
        previous_cost = self.compute_cost()
        for iteration in range(iterations):
            gradient = self.compute_gradient()
            self.theta = self.theta - learning_rate * gradient
            current_cost = self.compute_cost()
            self.cost_history.append(current_cost)
            if iteration % 100 == 0:
                print(
                    f"Iteration : {iteration:4d} | "
                    f"Cost : {current_cost:.6f}"
                )

            # Early Stopping
            if abs(previous_cost - current_cost) < tolerance:
                print("\nEarly Stopping...")
                print(f"Stopped at Iteration : {iteration}")
                break
            previous_cost = current_cost
        print("\nTraining Finished.")
        print(f"Final Cost : {self.compute_cost():.6f}")

        return self.theta

    # Train Model
    def train(self, learning_rate=0.01, iterations=1500):
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        initial_cost = self.compute_cost()
        print(f"Initial Cost : {initial_cost:.6f}")
        self.gradient_descent(learning_rate=learning_rate, iterations=iterations)
        print("\nTheta")
        print(self.theta)

    # Predict
    def predict(self, population):
        population = np.array(population, dtype=float)
        if population.ndim == 0:
            population = population.reshape(1, 1)
        elif population.ndim == 1:
            population = population.reshape(-1, 1)
        bias = np.ones((population.shape[0], 1))
        X_new = np.hstack((bias, population))
        # Prediction
        predictions = X_new @ self.theta
        return predictions * 10000

    # Model Evaluation
    def evaluate(self):
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        predictions = self.X @ self.theta
        errors = predictions - self.y
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"MAE  : {mae:.6f}")
        print(f"MSE  : {mse:.6f}")
        print(f"RMSE : {rmse:.6f}")
        print(f"R²   : {r2:.6f}")
        print("\nTraining Summary")
        print("-" * 40)
        print(f"Samples      : {self.m}")
        print(f"Features     : {self.n}")
        print(f"Final Cost   : {self.compute_cost():.6f}")
        print(f"Theta 0      : {self.theta[0,0]:.6f}")
        print(f"Theta 1      : {self.theta[1,0]:.6f}")
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    # Save Model
    def save_model(self, file_name="theta.npy"):
        np.save(file_name, self.theta)
        print(f"\nModel saved successfully -> {file_name}")

    # Load Model
    def load_model(self, file_name="theta.npy"):
        self.theta = np.load(file_name)
        print(f"\nModel loaded successfully <- {file_name}")

    def plot_training_data(self):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.population, self.profit, color="red", marker="o", label="Training Data")
        plt.title("Training Data")
        plt.xlabel("Population (10,000s)")
        plt.ylabel("Profit ($10,000)")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot Regression Line
    def plot_regression_line(self):
        predictions = self.X @ self.theta
        plt.figure(figsize=(8, 5))
        plt.scatter(self.population, self.profit, color="red", label="Training Data")
        plt.plot(self.population, predictions, color="blue", linewidth=2, label="Regression Line")
        plt.title("Linear Regression")
        plt.xlabel("Population (10,000s)")
        plt.ylabel("Profit ($10,000)")
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot Cost History
    def plot_cost_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.cost_history) + 1), self.cost_history, color="blue")
        plt.title("Cost Function History")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.grid(True)
        plt.show()

    # Plot Actual vs Predicted
    def plot_actual_vs_predicted(self):
        predictions = self.X @ self.theta
        plt.figure(figsize=(6, 6))
        plt.scatter(self.y, predictions, color="green")
        min_value = min(self.y.min(), predictions.min())
        max_value = max(self.y.max(), predictions.max())
        plt.plot([min_value, max_value], [min_value, max_value], "r--")
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True)
        plt.show()

    # Plot Residuals
    def plot_residuals(self):
        predictions = self.X @ self.theta
        residuals = self.y - predictions
        plt.figure(figsize=(8, 5))
        plt.scatter(predictions, residuals, color="purple")
        plt.axhline(y=0, color="red", linestyle="--")
        plt.title("Residual Plot")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.show()

# Main
if __name__ == "__main__":
    FILE_PATH = "C:/Users/User/Desktop/Machine learning exercise/data1/data/ex1data1.txt"
    model = LinearRegressionModel(FILE_PATH)
    model.dataset_pipeline()
    model.train(learning_rate=0.01, iterations=1500)
    model.evaluate()
    population = 3.5
    prediction = model.predict(population)
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)
    print(f"Population : {population * 10000:,.0f}")
    print(f"Predicted Profit : ${prediction[0][0]:,.2f}")
    model.save_model()
    model.plot_training_data()
    model.plot_regression_line()
    model.plot_cost_history()
    model.plot_actual_vs_predicted()
    model.plot_residuals()