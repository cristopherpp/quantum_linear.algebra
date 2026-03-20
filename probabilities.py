import random
import math

def bernoulli(p, n):
    return [1 if random.random() < p else 0 for _ in range(n)]

def normal(mu, sigma, n):
    return [random.gaus(mu, sigma) for _ in range(n)]

def mean(data):
    return sum(data) / len(data)

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def std_dev(data):
    return math.sqrt(variance(data))

def probability(condition, data):
    return sum(1 for x in data if condition(x)) / len(data)

# use or probability
data = bernoulli(0.7, 1000)
print(probability(lambda x: x == 1, data))

def conditional_probability(cond_A, cond_B, data):
    filtered =  [x for x in data if cond_B(x)]
    if len(filtered) == 0:
        return 0
    return sum(1 for x in filtered if cond_A(x)) / len(filtered)

def linear_regresion(x, y):
    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    num = sum((x[i] - mean_x)*(y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x)**2 for i in range(n))

    slope = num / den
    intercept = mean_y - slope * mean_x

    return slope, intercept


