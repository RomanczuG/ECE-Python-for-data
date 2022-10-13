import math as m
from os import stat
import numpy as np
import scipy.stats as stats

# import or paste dataset here

data = [3, -3, 3, 12, 15, -16, 17, 19, 23, -24, 32]


null_hypothesis = 0.75
confidence = 0.9
alpha = (1 - confidence) / 2

sample_size = len(data)
sample_mean = np.mean(data)
std_deviation = np.std(data, ddof=1)
std_error = std_deviation / (np.sqrt(sample_size))
t_score = stats.ttest_1samp(data, alpha)[0]
p_value = stats.ttest_1samp(data, alpha)[1]
range_value1 = sample_mean - t_score * std_error
range_value2 = sample_mean + t_score * std_error
# code for question 1
print('\nProblem 1 Answers:')
# code below this line

print("Sample size is: " + str(sample_size))
print("Sample mean is: " + str(sample_mean))
print("Standard error is: " + str(std_error))
print("t_score is: " + str(t_score))
print("p-value is: " + str(p_value))
print("The range of confidence from " + str(range_value1) + ' to ' + str(range_value2))

# code for question 2
print('\nProblem 2 Answers:')
# code below this line
null_hypothesis = 0.75
confidence = 0.95
alpha = (1 - confidence) / 2

sample_size = len(data)
sample_mean = np.mean(data)
std_deviation = np.std(data, ddof=1)
std_error = std_deviation / (np.sqrt(sample_size))
t_score = stats.ttest_1samp(data, alpha)[0]
p_value = stats.ttest_1samp(data, alpha)[1]
range_value1 = sample_mean - t_score * std_error
range_value2 = sample_mean + t_score * std_error

print("Sample size is: " + str(sample_size))
print("Sample mean is: " + str(sample_mean))
print("Standard error is: " + str(std_error))
print("t_score is: " + str(t_score))
print("p-value is: " + str(p_value))
print("The range of confidence from " + str(range_value1) + ' to ' + str(range_value2))

# code for question 3
print('\nProblem 3 Answers:')
# code below this line

std_deviation = 15.836
std_error = std_deviation / (np.sqrt(sample_size))
z_score =  (sample_mean - null_hypothesis) / std_error
p_value = 2* stats.norm.cdf(-abs(z_score))
range_value1 = sample_mean - z_score * std_error
range_value2 = sample_mean + z_score * std_error

print("Standard error is: " + str(std_error))
print("Standard score is: " + str(z_score))
print("p-value is: " + str(p_value))
print("The range of confidence from " + str(range_value1) + ' to ' + str(range_value2))

# code for question 4
print('\nProblem 4 Answers:')
# code below this line




sample_size = len(data)
sample_mean = np.mean(data)
std_error = std_deviation / (np.sqrt(sample_size))


t_score = sample_mean/std_error
p_value = stats.t.cdf(-abs(t_score), sample_size-1)
confidence = 1 - p_value

print("Level of confidence is " + str(confidence))
