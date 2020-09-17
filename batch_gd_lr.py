from math import fabs
import time


class LinearRegressor:
    def __init__(self, x, y, alpha=0.01, decay=0.9, b0=0, b1=0, max_epochs=1000, beta=0.9):
        """
            x: input feature
            y: result / target
            alpha: learning rate, default is 0.01
            b0, b1: linear regression coefficient.
        """
        self.i = 0
        self.x = x
        self.y = y
        self.alpha0 = self.alpha = alpha
        self.beta = beta
        self.b0 = b0
        self.b1 = b1
        self.max_epochs = max_epochs
        self.decay = decay
        self.b0_momentum = 0
        self.b1_momentum = 0
        if len(x) != len(y):
            raise TypeError("x and y should have same number of rows.")

    def predict(model, x):
        """Predicts the value of prediction based on
           current value of regression coefficients when input is x"""
        # Y = b0 + b1 * X
        return model.b0 + model.b1 * x

    def cost_derivative(model, i, idx):
        x, y, b0, b1 = model.x, model.y, model.b0, model.b1
        predict = model.predict
        # index = 1

        # for xi, yi in zip(x, y):
        #     print(f"Prediction for {index}: {predict(xi)}")
        #     index += 1

        return (predict(x[idx]) - y[idx]) if idx == 0 else (predict(x[idx]) - y[idx]) * x[idx]

    def update_coeff(model, i, idx):
        cost_derivative = model.cost_derivative
        mse0 = 0
        mse1 = 0
        if i == 0:
            # print("Updating b0")
            mse0 = cost_derivative(i, idx)
            model.b0_momentum = model.beta * model.b0_momentum + (1 - model.beta) * mse0
            model.b0 -= model.alpha * model.b0_momentum
            return mse0
        elif i == 1:
            # print("Updating b1")
            mse1 = cost_derivative(i, idx)
            model.b1_momentum = model.beta * model.b1_momentum + (1 - model.beta) * mse1
            model.b1 -= model.alpha * model.b1_momentum
            return mse1

        return 0

    def stop_iteration(model):
        model.i += 1
        # print(f"Epoch {model.i}")
        if model.i == model.max_epochs:
            return True
        else:
            return False

    def fit(model):
        update_coeff = model.update_coeff
        model.i = 0
        while True:
            try:
                if model.stop_iteration():
                    break
                else:
                    for idx in range(len(model.x)):
                        mse0 = update_coeff(0, idx)
                        mse1 = update_coeff(1, idx)

                        if model.i % 10000 == 0 or model.i == 1:
                            print(f"{(model.i / model.max_epochs)*100:.10f}% complete.")
                            print(f"MSE of iteration {model.i} is:.")
                            print(f"MSE0: {mse0}")
                            print(f"MSE1: {mse1}")
                            print(f"B0: {model.b0}.")
                            print(f"B1: {model.b1}.")
                            print(f"Learning rate: {model.alpha}")

                        model.alpha = (1 / (1 + model.decay * model.i)) * model.alpha0
            except KeyboardInterrupt:
                break


if __name__ == '__main__':
    x_values = [i for i in range(12)]
    y_values = [2 * i + 3 for i in range(12)]

    linearRegressor = LinearRegressor(
        x=x_values,
        y=y_values,
        alpha=0.003,  # 0.035 current best
        b0=1,
        b1=1,
        max_epochs=10000000,
        decay=0.000000000001,  # 0.001 current best
    )
    start_time = time.time()
    linearRegressor.fit()
    end_time = time.time()

    print(f"Training complete in {((end_time - start_time)/60):.5f} minutes. Press enter to continue.")
    input()

    for xi, yi in zip(x_values, y_values):
        print(f"X is {xi}")
        print(f"Y is {yi}")
        print(f"Y_hat is {linearRegressor.predict(xi)}")
        print()

    print(linearRegressor.predict(12))

    # expects 2 * 12 + 3 = 27
