import random
import math

# RANDOM DATA GENERATION

def bernoulli(p, n):
    """
    Generate n Bernoulli(p) observations.

    Each value is:
    - 1 with probability p
    - 0 with probability 1 - p

    Example:
        bernoulli(0.7, 5) -> [1, 0, 1, 1, 0]
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    if n <= 0:
        raise ValueError("n must be positive.")

    return [1 if random.random() < p else 0 for _ in range(n)]

def normal(mu, sigma, n):
    """
    Generate n observations from a Normal(mu, sigma) distribution.

    mu    = mean of the distribution
    sigma = standard deviation of the distribution

    random.gaus(mu, sigma) gives one Gaussian random value.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if n <= 0:
        raise ValueError("n must be positive.")

    return [random.gaus(mu, sigma) for _ in range(n)]

# BASIC DESCRIPTIVE STATISTICS

def mean(data):
    """
    Compute the arithmetic mean of a dataset.

    Formula:
        mean = (x1 + x2 + ... + xn) / n
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    return sum(data) / len(data)


def population_variance(data):
    """
    Compute the population variance.

    Use this when your dataset is the entire population.

    Formula:
        Var(X) = (1/n) * sum((xi - mean)^2)

    Important:
    This is NOT the usual variance used when estimating from a sample.
    """
    if len(data) == 0:
         raise ValueError("data must not be empty.")

    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def sample_variance(data):
    """
    Compute the sample variance.

    Use this when your dataset is a sample from a larger population.

    Formula:
        s^2 = (1/(n-1)) * sum((xi - mean)^2)

    Why n - 1?
    Because this corrects the bias when estimating population variance
    from sample data. This is Bessel's correction.
    """
    if len(data) < 2:
        raise ValueError("sample variance requires at least 2 data points.")

    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def population_std_dev(data):
    """
    Population standard deviation = sqrt(population variance)
    """
    return math.sqrt(population_variance(data))


def sample_std_dev(data):
    """
    Sample standard deviation = sqrt(sample variance)
    """
    return math.sqrt(sample_variance(data))


def minimum(data):
    """
    Return the smallest value in the dataset.
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    return min(data)


def maximum(data):
    """
    Return the largest value in the dataset.
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    return max(data)


def data_range(data):
    """
    Range = max(data) - min(data)

    This is a very rough measure of spread.
    """
    return maximum(data) - minimum(data)

# EMPIRICAL PROBABILITY
#
def probability(condition, data):
    """
    Estimate probability from data.

    condition: a function that returns True or False for each observation
    data: dataset

    Example:
        probability(lambda x: x == 1, [1, 0, 1, 1]) -> 0.75

    Interpretation:
    This computes relative frequency:
        P(A) ≈ count(A) / n
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    return sum(1 for x in data if condition(x)) / len(data)


def conditional_probability(cond_A, cond_B, data):
    """
    Estimate conditional probability P(A | B) from data.

    Steps:
    1. Keep only the data points where B is true
    2. Among those, count how many also satisfy A

    Formula:
        P(A | B) = P(A and B) / P(B)

    Example:
        P(x > 5 | x is even)
    """
    if len(data) == 0:
        raise ValueError("data must not be empty.")

    filtered = [x for x in data if cond_B(x)]

    if len(filtered) == 0:
        return 0.0

    return sum(1 for x in filtered if cond_A(x)) / len(filtered)

# COVARIANCE AND CORRELATION

def covariance(x, y, sample=True):
    """
    Compute covariance between two datasets x and y.

    Covariance measures whether x and y move together.

    Positive covariance:
        when x increases, y tends to increase

    Negative covariance:
        when x increases, y tends to decrease

    sample=True  -> divide by (n - 1)
    sample=False -> divide by n
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) == 0:
        raise ValueError("x and y must not be empty.")
    if sample and len(x) < 2:
        raise ValueError("sample covariance requires at least 2 points.")

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))

    if sample:
        return numerator / (len(x) - 1)
    else:
        return numerator / len(x)


def correlation(x, y):
    """
    Compute Pearson correlation between x and y.

    Formula:
        corr(x, y) = cov(x, y) / (std_x * std_y)

    This standardizes covariance so the result lies between -1 and 1.
    """
    std_x = sample_std_dev(x)
    std_y = sample_std_dev(y)

    if std_x == 0 or std_y == 0:
        raise ValueError("correlation is undefined when one variable has zero variance.")

    return covariance(x, y, sample=True) / (std_x * std_y)

# LINEAR REGRESSION

def linear_regression(x, y):
    """
    Fit a simple linear regression line:
        y = slope * x + intercept

    This uses ordinary least squares (OLS).

    slope tells you how much y changes when x increases by 1.
    intercept is the predicted value of y when x = 0.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("linear regression requires at least 2 points.")

    mean_x = mean(x)
    mean_y = mean(y)

    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
    den = sum((x[i] - mean_x) ** 2 for i in range(len(x)))

    if den == 0:
        raise ValueError("cannot fit regression when all x values are identical.")

    slope = num / den
    intercept = mean_y - slope * mean_x

    return slope, intercept


def predict(x_value, slope, intercept):
    """
    Predict y from a fitted regression line.

    Formula:
        y_hat = slope * x + intercept
    """
    return slope * x_value + intercept


def residuals(x, y, slope, intercept):
    """
    Compute residuals:
        residual = observed_y - predicted_y

    Residuals tell you the prediction error for each point.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    return [y[i] - predict(x[i], slope, intercept) for i in range(len(x))]


def r_squared(x, y, slope, intercept):
    """
    Compute R^2 for a simple linear regression model.

    R^2 measures how much of the variance in y is explained by the model.

    Formula:
        R^2 = 1 - SS_res / SS_tot
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(y) == 0:
        raise ValueError("data must not be empty.")

    y_mean = mean(y)
    ss_res = sum((y[i] - predict(x[i], slope, intercept)) ** 2 for i in range(len(y)))
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)

    if ss_tot == 0:
        return 1.0

    return 1 - (ss_res / ss_tot)

# STANDARDIZATION

def z_scores(data):
    """
    Convert data into z-scores.

    Formula:
        z = (x - mean) / sample_std_dev

    Interpretation:
    A z-score tells you how many standard deviations a value is
    above or below the mean.
    """
    s = sample_std_dev(data)
    if s == 0:
        raise ValueError("z-scores are undefined when standard deviation is zero.")

    m = mean(data)
    return [(x - m) / s for x in data]

# EXAMPLE USAGE

if __name__ == "__main__":
    # Simulate Bernoulli data
    data = bernoulli(0.7, 1000)

    # Estimate P(X = 1)
    print("Estimated P(X = 1):", probability(lambda x: x == 1, data))

    # Simulate paired data for regression
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 6]

    slope, intercept = linear_regression(x, y)
    print("Slope:", slope)
    print("Intercept:", intercept)
    print("Prediction at x = 10:", predict(10, slope, intercept))
    print("Correlation:", correlation(x, y))
    print("R^2:", r_squared(x, y, slope, intercept))
