def exponent_map(n):
    """
    Returns a new function f. f should take in a list L and computes x^n over every element x in L, returning the new list after exponentiation
    :param n: exponent
    :return: function
    """
    numbers=[1,2,3,4]
    squares=list(map(lambda x:pow(x,2),numbers))
    print(squares)
    return lambda L : list(map(lambda x : pow(x,n), L))


def compose_map(fun1, fun2, L):
    """
    Returns a new list r where each element in r is fun2(fun1(i)) for the
    corresponding element i in L
    :param fun1: function
    :param fun2: function
    :param L: list
    :return: list
    """
    return list(map(lambda x : fun1( fun2( x ) ), L))
    # Fill in
    pass


def compose(fun1, fun2):
    # Fill in
    """
    Returns a new function f. f should take a single input i, and return
    fun1(fun2(i))
    :param fun1: function
    :param fun2: function
    :return: function
    """
    def ret_fun(i):
        return fun1(fun2(i))
        # Fill in
        pass

    return ret_fun


def repeater(fun, num_repeats):
    """
    Returns a new function f. f should take a single input , and returnx
    fun applied to x num_repeats times. In other words, if num_repeats is 1, f
    should return fun(x). If num_repeats is 2, f should return fun(fun(x)). If
    num_repeats is 0, f should return x.
    :param fun: function
    :param num_repeats: int
    :return: function
    """
    def ret_fun(x):
        if num_repeats == 0:
            return x
        
        for _ in range (num_repeats-1):
            x = fun(x)

        return x
            
        # Fill in
        pass

    return ret_fun


if __name__ == '__main__':

    def test1(x):
        return x * 2

    def test2(x):
        return x - 3

    data = [2, 5, -10, -7, -7, -3, -1, 9, 8, -6]
    
    f1 = exponent_map(2)
    print(f1(data))

    print(compose_map(test1, test2, data))
    print(compose_map(test2, test1, data))

    f2 = compose(test1, test2)

    print(f2(4))

    print(list(map(f2, data)))

    f3 = compose(test2, test1)

    print(f3(4))

    print(list(map(f3, data)))

    z = repeater(test1, 0)
    once = repeater(test1, 1)
    twice = repeater(test1, 2)
    thrice = repeater(test1, 3)

    print("repeat 0 times: {}".format(z(5)))
    print("repeat 1 time: {}".format(once(5)))
    print("repeat 2 times: {}".format(twice(5)))
    print("repeat 3 times: {}".format(thrice(5)))
