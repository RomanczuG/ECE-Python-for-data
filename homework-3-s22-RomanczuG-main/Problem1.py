from itertools import count
import numpy as np
import matplotlib.pyplot as plt


def norm_histogram(hist):
    """
    takes a histogram of counts and creates a histogram of probabilities
    :param hist: a numpy ndarray object
    :return: list
    """
    
    counts=sum(hist)
    list=[]

    for num in hist:
        prob=num/counts
        list.append(prob)

    return list


def compute_j(histo, width):
    """
    takes histogram of counts, uses norm_histogram to convert to probabilties, it then calculates compute_j for one bin width
    :param histo: list
    :param width: float
    :return: float
    """
    sum = 0
    m = 0
    
    hist = norm_histogram(histo)
    hist_sq = [0] * len(hist)
    for i in range(0,len(hist)):

        hist_sq[i] = hist[i] ** 2
        sum += (hist_sq[i])
        m += histo[i]

    J = 2/ ((m - 1) * width) - (((m * 1)/((m - 1) * width)) * (sum))
    return J

    # counts=sum(histo)
    # prob = norm_histogram(histo)

    # temp_func = lambda i: i*i

    # sum_prob = sum(map(temp_func, prob))

    # J=((2/((counts-1)*width))-((counts+1)/((counts-1)*width)))*sum_prob

    # return J


def sweep_n(data, minimum, maximum, min_bins, max_bins):
    """
    find the optimal bin
    calculate compute_j for a full sweep [min_bins to max_bins]
    please make sure max_bins is included in your sweep
    :param data: list
    :param minimum: int
    :param maximum: int
    :param min_bins: int
    :param max_bins: int
    :return: list
    """
    J_list=[]

    for bins in range(min_bins,max_bins+1):
        hist_vals=plt.hist(data,bins,(minimum,maximum))
        J_comp=compute_j(hist_vals[0],((maximum-minimum)/bins))
        J_list.append(J_comp)


    return J_list


def find_min(l):
    """
    takes a list of numbers and returns the mean of the three smallest number in that list and their index.
    return as a tuple i.e. (the_mean_of_the_3_smallest_values,[list_of_the_3_smallest_values])
    For example:
        A list(l) is [14,27,15,49,23,41,147]
        The you should return ((14+15+23)/3,[0,2,4])

    :param l: list
    :return: tuple
    """

    minimumIndex = []
    minimumSum = 0

    for i in range(0,3) :
        minimum=min(l)
        minimumSum += minimum
        minimumIndex.append(l.index(minimum))
        l[l.index(minimum)] = max(l)

    average = minimumSum/3
    return (average,minimumIndex)



if __name__ == "__main__":
    data = np.loadtxt("input.txt")  # reads data from input.txt
    lo = min(data)
    hi = max(data)
    bin_l = 1
    bin_h = 100
    js = sweep_n(data, lo, hi, bin_l, bin_h)
    """
    the values bin_l and bin_h represent the lower and higher bound of the range of bins.
    They will change when we test your code and you should be mindful of that.
    """
    print(find_min(js))
