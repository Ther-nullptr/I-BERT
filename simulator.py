import math

def linear_function_e(x, x_0):
    return math.exp(x_0) * x + (1 - x_0) * math.exp(x_0)

def linear_function_2(x, x_0):
    return (2 ** x_0) * math.log(2) * x + (1 - x_0 * math.log(2)) * (2 ** x_0)

if __name__ == '__main__':
    print(linear_function_e(15, 5.5))
    print(linear_function_2(20, 5.5))