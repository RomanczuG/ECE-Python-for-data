#!/usr/bin/python3
import math
number = 100
# Your code should be below this line
test1 = (5*number**2 + 4)
test2 = (5*number**2 - 4)

root1 = int(math.sqrt(test1) + 0.5)
root2 = int(math.sqrt(test2) + 0.5)

if (number > 0) and (number%2 == 0) and ((root1) ** 2 == test1 or (root2) ** 2 == test2):
    print("Yes")

else:
    print("No")