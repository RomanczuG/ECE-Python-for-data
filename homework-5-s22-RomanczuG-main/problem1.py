import numpy as np
import math as m
import scipy.stats as stats
from scipy.stats import norm
from scipy.stats import t


# import or paste dataset here
myFile = open('engagement_0.txt')
#myFile = open('/Users/mateusz/Desktop/Classes/ECE20875/Assigments/homework-5-s22-RomanczuG/engagement_0.txt')

data0 = myFile.readlines()
myFile.close()

myFile = open('engagement_1.txt')
#myFile = open('/Users/mateusz/Desktop/Classes/ECE20875/Assigments/homework-5-s22-RomanczuG/engagement_0.txt')

data1 = myFile.readlines()
myFile.close()

data0 = [float(x) for x in data0]
data1 = [float(x) for x in data1]


null_hypothesis1 = 0.75

# code for question 2
print('\nProblem 2 Answers:')
# code below this line
sample_size1 = len(data1)
sample_mean1 = np.mean(data1)
std_deviation1 = np.std(data1, ddof=1)
std_error1 = std_deviation1 / (np.sqrt(sample_size1))
z_score =  (sample_mean1 - null_hypothesis1) / std_error1
p_value = 2* stats.norm.cdf(-abs(z_score))

print("Sample size is: " + str(sample_size1))
print("Sample mean is: " + str(sample_mean1))
print("Standard error is: " + str(std_error1))
print("Standard score is: " + str(z_score))
print("p-value is: " + str(p_value))

if p_value < 0.1:    
    print("90% confidence interval, reject the null hypothesis")
else:
    print("90% confidence interval, accept the null hypothesis")
    
if p_value < 0.05:    
    print("95% confidence interval, reject the null hypothesis")
else:
    print("95% confidence interval, accept the null hypothesis")
    
if p_value < 0.01:   
    print("99% confidence interval, reject the null hypothesis")
else:
    print("99% confidence interval, accept the null hypothesis")


# code for question 3
print('\nProblem 3 Answers:')
# code below this line
alpha = 0.05
stdr_alpha = norm.ppf(alpha)

SE_alpha1 = (sample_mean1 - null_hypothesis1) / stdr_alpha
size_alpha1 = (std_deviation1 / SE_alpha1) ** 2

print("Standard error: " + str(SE_alpha1))
print("Minimum sample size: " + str(size_alpha1))


# code for question 5
print('\nProblem 5 Answers:')
# code below this line

null_hypothesis0 = 0.75

sample_size0 = len(data0)
sample_mean0 = np.mean(data0)
std_deviation0 = np.std(data0, ddof=1)
std_error0 = std_deviation0 / (np.sqrt(sample_size0))
z_score =  (sample_mean0 - null_hypothesis0) / std_error0
p_value = 2* stats.norm.cdf(-abs(z_score))

print("Sample size is: " + str(sample_size0))
print("Sample mean is: " + str(sample_mean0))
print("Standard error is: " + str(std_error0))
print("Standard score is: " + str(z_score))
print("p-value is: " + str(p_value))

if p_value < 0.1:    
    print("90% confidence interval, reject the null hypothesis")
else:
    print("90% confidence interval, accept the null hypothesis")
    
if p_value < 0.05:    
    print("95% confidence interval, reject the null hypothesis")
else:
    print("95% confidence interval, accept the null hypothesis")
    
if p_value < 0.01:   
    print("99% confidence interval, reject the null hypothesis")
else:
    print("99% confidence interval, accept the null hypothesis")
