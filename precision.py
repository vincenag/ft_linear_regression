import numpy as np
import pandas as pd

class ModelEvaluator:
    def __init__(self, dataset_file, thetas_file):
        self.dataset_file = dataset_file
        self.thetas_file = thetas_file
        self.load_thetas()
        self.load_dataset()

    def load_thetas(self):
        with open(self.thetas_file, "r") as file:
            theta_values = file.readlines()
            self.theta0 = float(theta_values[0].split("=")[1].strip())
            self.theta1 = float(theta_values[1].split("=")[1].strip())

    def load_dataset(self):
        df = pd.read_csv(self.dataset_file)
        self.mileage = df["km"].values
        self.actual_price = df["price"].values
    
    def calculate_errors(self):
        predicted_price = self.theta0 + self.theta1 * self.mileage
        errors = predicted_price - self.actual_price 
        # R2 score (Coefficient of Determination) --> 1.0 Perfect Fit, 0.0 No explanatory power
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((self.actual_price - np.mean(self.actual_price)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        print(f"R² Score: {r2:.4f}")

        if r2 >= 0.9:
            print("The precision of your model is Excellent!")
        elif r2 >= 0.75:
            print("The precision of your model is Good.")
        elif r2 >= 0.5:
            print("The precision of your model is Moderate.")
        else:
            print("The precision of your model is Poor. Consider improving it.")

if __name__ == "__main__":
    evaluator = ModelEvaluator("data.csv", "coefficients.txt")
    evaluator.calculate_errors()