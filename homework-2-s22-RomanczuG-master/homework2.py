def histogram(data, n, b, h):

    
    if b == h:
        print('b and h are the same value')
        return []
    elif n <= 0:
        return []
    else:
        if b > h:
            temp = h
            h = b
            b = temp
        hist = [0] * n
        w = (h-b)/n
        

        for i in range(0,n):
            min = b + i*w 
            max = b + (i + 1) * w
            for number in data:
                if i == 0:
                    if min < number < max:
                        hist[i] = hist[i] + 1
                else:
                    if min <= number < max:
                        hist[i] = hist[i] + 1

        return hist
        
        

    # data is a list
    # n is an integer
    # b and h are floats
    
    # Write your code here

    # return the variable storing the histogram
    # Output should be a list
    return histogram


def happybirthday(name_to_day, name_to_month, name_to_year):
    #name_to_day, name_to_month and name_to_year are dictionaries
    
    # Write your code here

    # return the variable storing name_to_all
    # Output should be a dictionary
    month_to_all = {}
    for name, month in name_to_month.items():
        
        month_to_all[month] = (name,(name_to_day[name], name_to_year[name], 2022 - name_to_year[name]))
    return month_to_all
