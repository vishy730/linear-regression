
import pandas as pd
from numpy import *

def compute_error_for_line_given_points(c,m, df):
    total_error = 0
    for i in range(0, len(df)):
        # get x value
        x = df[i][0]
        # get y vaue
        y = df[i][8]
        # get the difference, square it and add it to the total
        
        total_error += (y - (m * x + c)) **2
    return total_error/float(len(df))

def gradient_descent_number(df, starting_c, starting_m, learning_rate, num_iterations):
    c = starting_c
    m = starting_m
    for i in range(num_iterations):
        c,m = step_gradient(c, m, array(df), learning_rate)
    return [c, m]

def step_gradient(current_c, current_m, df, learning_rate1):
    c_gradient = 0
    m_gradient = 0
    N = float(len(df))
    for i in range(0, len(df)):
        x = df[i][0]
        y = df[i][8]
        c_gradient += -(2/N) * (y- ((current_m*x) + current_c))
        m_gradient += -(2/N) * x * (y-((current_m*x) + current_c))
    new_b = current_c - (learning_rate1 * c_gradient)
    new_m = current_m - (learning_rate1 * m_gradient)
    return [new_b, new_m]


def run():
    # read data
    df = genfromtxt('testData.csv', delimiter=',',dtype=None, names=True)
    print(df)

    # defining hyperparameters
    learning_rate = 0.0001
    #y=mx + c where m is the slope and c is the y-intercept
    initial_m = 0
    initial_c = 0
    num_iterations = 100

    # training the model
    print('starting gradient descent at c = {0}, m = {0}, error = {2}'.format(initial_c, initial_m,
                                                                              compute_error_for_line_given_points
                                                                              (initial_c, initial_m, df)))
    [c, m] = gradient_descent_number(df, initial_c, initial_m, learning_rate, num_iterations)

    print('ending gradient descent at c = {1}, m = {2}, error = {3}'.format(num_iterations, c, m,
                                                                              compute_error_for_line_given_points
                                                                              (c, m, df)))

if __name__ == '__main__':
    run()

# from sklearn import linear_model
# import pandas as pd
# from numpy import *
#
# df = pd.read_csv('testData.csv')
# points = genfromtxt('testData.csv', delimiter=',',dtype=None, names=True)
# print(points)
# for i in range(0,len(points)):
#     x = points[i][1]
#     print(x)


# reg = linear_model.LinearRegression()
# x = points['Age']
# y = points['Mortgage']
# print(x)
# print(y)
# reg.fit(x,y)
# print(reg.coef_)