# Kinga Kwoka
# Olga Krupa

from random import uniform
import numpy as np
from matplotlib import pyplot as plt


def j(x):
    """
    Calculates the value of a given function J:[-5,5]->R
    """
    return np.sin(x*np.sqrt(1+1))+np.cos(x*np.sqrt(8+1))


def plot_graph(theta1, theta2):
    """
    Plots the given function (in blue)
    and the approximation based on a 100
    random points in range [-5,5](in red).
    Additionally calculates and prints the error on the generated set.
    """
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, j(x))
    errors = 0
    for i in range(100):
        x, d = generate_training_pair(-5, 5)
        value = approximate_function(x, theta1, theta2)[0]
        errors += (value - d)**2
        plt.scatter(x, value, color="red")
    print(f'MSE = {errors/100}')


def y(x, theta_line):
    """
    x - input vector for neuron
    theta_line - weights for neuron
    Calculates the sum for one neuron.
    """
    sum = 0
    for i in range(len(x)):
        sum += theta_line[i]*x[i]
    return round(sum, 3)


def sig(value):
    """
    Calculates the value of a sigmoidal
    activation function arctg(x)
    """
    result = np.arctan(value)
    return result


def der_arctan(value):
    """
    Calculates the value of a derivative
    of a sigmoidal activation function arctg(x)"""
    v = 1 + np.power(value, 2)
    result = 1/v
    return result


def y1_vector(theta, x):
    """
    x - input vector with prreviously added 1 (y0=[x, 1])
    theta- theta' matrix with weights for sigmoidal neurons
    Calculates the output vector y1 of hidden layer and extends it with a 1.
    """
    y1 = []
    for line in theta:
        y1.append(sig(y(x, line)))
    y1.append(1)
    return y1


def y2_vector(theta, x):
    """
    x - input vector with prreviously added 1 (y1 = [y, 1])
    theta - theta"
    Calculates the output vector y2 of output layer.
    """
    y2 = []
    for line in theta:
        y2.append((y(x, line)))
    return y2


def generate_theta(n, type):
    """
    n - number of hidden layer neurons
    type - {1, 2} marks theta' or theta''
    Generates the initial theta' or theta'' matrix.
    """
    if type == 1:
        theta = []
        for i in range(n):
            theta.append([round(uniform(-1, 1), 3), round(uniform(-1, 1), 3)])
    elif type == 2:
        theta = []
        line = []
        for i in range(n+1):
            line.append(0)
        theta.append(line)
    return theta


def generate_training_pair(min, max):
    """
    Generates a pair (x,d) such that
    x is in [-5, 5] and d = J(x).
    """
    x = uniform(min, max)
    value = j(x)
    return x, value


def recalculate_theta2(theta, approx_val, d_value, y1_value):
    """
    Recalculates theta''
    """
    beta = 0.009
    for i in range(len(theta)):
        error = (d_value - approx_val[i])
        for j in range(len(theta[i])):
            element = theta[i][j]
            element = element - 2*beta*error*y1_value[j]
            theta[i][j] = round(element, 10)
    return theta


def sum_der_y2(approx_val, theta, j, d_value):
    """
    Calculates the sum of partial derivative in recalculating theta'
    """
    sum = 0
    for i in range(len(theta)):
        sum += 2*(approx_val[i]-d_value)*theta[i][j]
    if np.isnan(sum):
        # if the numbers get too small
        sum = 0
    return sum


def recalculate_theta1(theta1, theta2, approx_val, d_value, y0_value):
    """
    Recalculates theta'
    """
    beta = 0.05
    for j in range(len(theta1)):
        for i in range(len(theta1[j])):
            element = theta1[j][i]
            z = (2*beta*y0_value[i]
                 * der_arctan(y(y0_value, theta1[j]))
                 * sum_der_y2(approx_val, theta2, j, d_value))
            element = element - z
            theta1[j][i] = round(element, 10)
    return theta1


def approximate_function(x, theta1, theta2):
    """
    x - input value
    Approximates the value of J function
    """
    y_0 = [x, 1]
    y_1 = y1_vector(theta1, y_0)
    y_2 = y2_vector(theta2, y_1)[0]
    return y_2, y_1


def calculate_error(validating_set, theta1, theta2):
    """
    Calculates the error on a validating set
    """
    error_sum = 0
    for pair in validating_set:
        approx = approximate_function(pair[0], theta1, theta2)
        error = abs(approx[0]-pair[1])
        if error < 0.0000000000000001:
            error = 0
        error_sum += error
    error = error_sum/len(validating_set)
    return error


def run_learning():
    """
    Sets the parameters and constants.
    Runs the learning process.
    """
    t = 1
    error = 1000
    n = 20
    theta1 = generate_theta(n, 1)
    theta2 = generate_theta(n, 2)
    validating_set = []
    for i in range(100):
        validating_set.append(generate_training_pair(-5, 5))
    while t < 250 and error > 0.5:
        x, d = generate_training_pair(-5, 5)
        approx, y1 = approximate_function(x, theta1, theta2)
        theta2 = recalculate_theta2(theta2, [approx], d, y1)
        theta1 = recalculate_theta1(theta1, theta2, [approx], d, [x, 1])
        error = calculate_error(validating_set, theta1, theta2)
        t += 1
    print(f't = {t}')
    return theta1, theta2


if __name__ == "__main__":
    theta1, theta2 = run_learning()
    plot_graph(theta1, theta2)
    plt.show()
