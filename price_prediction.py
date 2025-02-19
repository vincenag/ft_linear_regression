import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CarPricePredictor:
    def __init__(self, thetas_file = "coefficients.txt"):
        self.theta0, self.theta1 = self.loadThetas(thetas_file)

    def loadThetas(self, thetas_file):
        with open(thetas_file, "r") as file:
            lines = file.readlines()
        theta0 = float(lines[0].split("=")[1].strip())
        theta1 = float(lines[1].split("=")[1].strip())
        return theta0, theta1
    
    def predictPrice(self, mileage):
        return self.theta0 + self.theta1 * mileage
    
    def dataPlot(self, data_file):
        df = pd.read_csv(data_file)
        plt.figure(figsize=(10,5))
        plt.scatter(df["km"], df["price"], color="b", marker="o", label= "Price by mileage")

        # Generate regression line
        x_values = np.linspace(0, df["km"].max() * 1.2, 100)
        y_values = self.theta0 + self.theta1 * x_values

        # Plot the regression line
        plt.plot(x_values, y_values, color="r", label="Regression Line")
        plt.xlim(0, df["km"].max() * 1.2)  

        plt.xlabel("Mileage [Km]")
        plt.ylabel("Price [$]")
        plt.title("Price by mileage")
        plt.grid(True)
        plt.show()
            

if __name__ == "__main__":

    predict = CarPricePredictor()
    mileage = float(input("Enter a car mileage: "))
    predictedPrice = predict.predictPrice(mileage)
    print (f"Estimated price: {predictedPrice:.2f}")
    predict.dataPlot("data.csv")