from itertools import count
from os import sep
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []
    data = pd.read_csv(datapath, sep = ' ', header= None)
    data.columns = ["x", "y"]

    for d in degrees:
        temp1 = feature_matrix(np.array(data["x"]), d)
        temp2 = least_squares(temp1, np.array(data["y"]))
        paramFits.append(temp2)
    
    visualize(np.array(data["x"]),np.array(data["y"]),paramFits)

    # fill in
    # read the input file, assuming it has two columns, where each row is of the form [x y] as
    # in poly.txt.
    # iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.

    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    X = [[number ** degree for degree in range(d, -1, -1)] for number in x]
    return X


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    least_sqrs = np.linalg.inv(X.T @ X) @ X.T @ y
    B = least_sqrs.tolist()
    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    return B

def visualize(x, y, paramFits):


    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of different degree models')
    plt.scatter(x,y,s=1)
    x.sort()
    for n in paramFits:
        deg_num = len(n) - 1
        feat_mtrx = np.array(feature_matrix(x, deg_num))
        y_hat = feat_mtrx @ n
        plt.plot(x, y_hat, label= 'degree = '+str(deg_num))

    plt.legend(fontsize = 8)
    plt.show()

if __name__ == "__main__":
    datapath = "poly.txt"
    degrees = [2, 4]

    paramFits = main(datapath, degrees)
    print(paramFits)
    # 2
    degrees = [1, 2, 3, 4, 5]
    paramFits = main(datapath, degrees)
    print(paramFits)

    # 4
    # Best fit is degree 5
    x = 2
    deg_num = 5
    feat_mtrx = [x ** degree for degree in range(deg_num, -1, -1)]
    y = np.matmul( feat_mtrx,  paramFits[4])
    print(y)
