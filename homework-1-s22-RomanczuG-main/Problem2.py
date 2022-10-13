#!/usr/bin/python3
import math
n = 21
# Your code should be below this line

month = 1
year = 2022


if not(n <= 31 and n >= 1):
    print("Not valid")

else:
    if month == 1:
        month = 13
        year = year -1
    
    if month == 2:
        month = 14
        year = year -1

    m = month
    K = year%100
    J = int(year/100)
    
    day = int( n + 13 * (m+1) / 5 + K + K/4 + J/4 + 5*J )
    day = day % 7
    if day <= 6 and day >= 0:
        if day <= 1:
            print("Weekend")
        else:
            print("Weekday")

