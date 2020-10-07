import numpy as np
import pandas as pd


def hypothesis(input_matrix, weights):
    return input_matrix.dot(weights.T)


def cost_function(input_matrix, predicted_output, expected_output, num_examples):
    cost = input_matrix.T.dot(predicted_output - expected_output) / num_examples
    return cost


def gradient_descent(input_matrix, predicted_output, expected_output, num_examples, learning_rate, weights):
    cost = cost_function(input_matrix, predicted_output, expected_output, num_examples)
    weights = weights - learning_rate * cost
    return weights


def regularized_cost_function(input_matrix, predicted_output, expected_output, num_examples, regularization_term, weights):
    regularization_filter = np.ones(num_examples)
    regularization_filter[0] = 0
    cost = (input_matrix.T.dot(predicted_output - expected_output) / num_examples) + ((regularization_term * weights * regularization_filter) / num_examples)
    return cost


def regularized_gradient_descent(input_matrix, predicted_output, expected_output, num_examples, learning_rate, weights, regularization_term):
    cost = regularized_cost_function(input_matrix, predicted_output, expected_output, num_examples, regularization_term, weights)
    weights = weights - learning_rate * cost
    return weights


def test_1():
    # extract values from dataset
    test_dataset = pd.read_csv("test_data.csv")

    # prepare features for linear regression
    num_rows, num_cols = test_dataset.shape
    bias = pd.DataFrame([1] * num_rows, columns=["Bias"])

    input_features = test_dataset.get("X")
    expected_outputs = test_dataset.get("y")

    X = pd.concat([bias, input_features], axis=1)
    y = expected_outputs

    X = np.asarray(X)
    y = np.asarray(y)
    theta = np.zeros(num_cols)

    # training
    max_epochs = 10000
    alpha = 0.002
    lambda_ = 1

    current_epoch = 0
    while current_epoch < max_epochs:
        current_epoch += 1
        y_hat = hypothesis(X, theta)
        theta = regularized_gradient_descent(X, y_hat, y, num_rows, alpha, theta, lambda_)
        if not current_epoch % 1000:
            print(f"Cost: {regularized_cost_function(X, y_hat, y, num_rows, lambda_, theta)}")
            print(f"Weights: {theta}")

    print(f"Final weights: {theta}")


def test_2():
    # extract values from dataset
    test_dataset = pd.read_csv("C:/Users/lucer/Desktop/Datasets/Bike Sharing - UCI Machine Learning Repo/hour.csv")

    test_dataset.pop("casual")
    test_dataset.pop("registered")
    test_dataset.pop("dteday")

    # prepare features for linear regression
    num_rows, num_cols = test_dataset.shape
    bias = pd.DataFrame([1] * num_rows, columns=["Bias"], dtype=float)

    input_features = test_dataset.drop("cnt", axis=1)
    expected_outputs = test_dataset.get("cnt")

    X = pd.concat([bias, input_features], axis=1)
    y = expected_outputs

    X = np.asarray(X)
    y = np.asarray(y)
    theta = np.zeros(num_cols)

    # training
    max_epochs = 100000
    alpha = 0.003

    current_epoch = 0
    while current_epoch < max_epochs:
        current_epoch += 1
        y_hat = hypothesis(X, theta)
        theta = gradient_descent(X, y_hat, y, num_rows, alpha, theta)
        if not current_epoch % 1000:
            print(f"Cost: {cost_function(X, y_hat, y, num_rows)}")
            print(f"Weights: {theta}")

    print(f"Final weights: {theta}")


if __name__ == '__main__':
    test_1()
