import numpy as np

class LinearRegTrainer:
    def __init__(self, dataset_file, learning_rate, iterations, tolerance):
        self.dataset_file = dataset_file
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.tolerance = tolerance

    def load_dataset(self):
        data = np.loadtxt(self.dataset_file, delimiter = ",", skiprows=1)
        self.mileage = data[:, 0]
        self.price = data[:,1]
        self.m = len(self.mileage)

        #Normalize features to avoid overflow
        self.mileage_mean = np.mean(self.mileage)
        self.mileage_std = np.std(self.mileage)
        self.price_mean = np.mean(self.price)
        self.price_std = np.std(self.price)

        self.mileage = (self.mileage - self.mileage_mean) / self.mileage_std
        self.price = (self.price - self.price_mean) / self.price_std

    def train_model(self):
        theta0= 0
        theta1= 0
        previous_cost = float("inf")

        for i in range(self.iterations):
            predictions = theta0 + theta1 * self.mileage
            error = predictions - self.price

            grad_theta0 = (1 / self.m) * np.sum(error)
            grad_theta1 = (1 / self.m) * np.sum(error * self.mileage)

            theta0 -= self.learning_rate * grad_theta0
            theta1 -= self.learning_rate * grad_theta1

            cost = (1/(2 * self.m) * np.sum(error ** 2))

            if abs(previous_cost - cost) < self.tolerance:
                print(f"Early stop at iteration{i}: Cost change is below tolerance.")
                break

            previous_cost = cost

        self.save_thetas(theta0, theta1)

    def save_thetas(self, theta0, theta1):
        #Denormalize
        theta1_orig = theta1 * (self.price_std / self.mileage_std)
        theta0_orig = (theta0 * self.price_std) + self.price_mean - (theta1_orig * self.mileage_mean)

        with open("coefficients.txt", "w") as file:
            file.write(f"theta0 = {theta0_orig}\n")
            file.write(f"theta1 = {theta1_orig}\n")
        print("Model parameters saved to coefficients.txt")


if __name__ == "__main__":
    trainer = LinearRegTrainer("data.csv", 0.001, 1000, 1e-6)

    trainer.load_dataset()

    trainer.train_model()


