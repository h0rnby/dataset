from math import exp, floor
from warnings import warn

import numpy as np


def mvlnorm_generate_costdur(num_samples, log_mean_dur=2.191054, log_var_dur=0.246245, log_mean_cost=6.642006,
                             log_var_cost=1.555780, cov=0.374572):
    """
    Generate data using a multi-variate log-normal distribution. The default values for means, variances,
    and covariance are derived from the (log) data extracted from the 2016 IIP. The MV log-normal distribution is
    simulated by using a MV normal distribution and exponentiating the resulting output.

    The multivariate normal distribution is generated using the log_mean values and a covariance matrix specified by
    [log_var_dur, cov]
    [cov, log_var_cost]

    :param log_mean_dur: Mean of the log-scaled project durations
    :param log_var_dur: Variance of the log-scaled project durations
    :param log_mean_cost: Mean of the log-scaled total project costs
    :param log_var_cost: Variance of the log-scaled total project costs
    :param cov: Covariance between log-scaled project durations and costs
    :param num_samples: The number of duration, cost samples to generate

    :return: A (num_samples x 2) numpy array of duration, cost samples rounded to the nearest integer
    """
    cov_mat = [[log_var_dur, cov], [cov, log_var_cost]]

    data = np.random.multivariate_normal([log_mean_dur, log_mean_cost], cov_mat, num_samples)
    data = np.rint(np.exp(data)).astype(int)

    return data


def fuzzy_weibull_cost_distribution(total_cost, duration, shape=1.589, scale=0.71, std_shape=2, std_scale=0.3):
    """
    Generate the cost distribution using a (fuzzy) Weibull distribution. The distribution is considered fuzzy as the
    shape and scale parameters are randomly drawn from Gaussian distributions. The cumulative distribution function is
    used to generate costs from the generated Weibull distribution.

    :param total_cost: The total cost of the project, to be broken down into yearly costs.
    :param duration: The duration of the project.
    :param shape: The mean of the shape parameter.
    :param scale: The mean of the scale parameter.
    :param std_shape: The standard deviation of the shape parameter.
    :param std_scale: The standard deviation of the scale parameter.

    :return: A yearly cost array.
    """

    # generate a shape and scale parameter using Gaussian distributions
    fuzzy_shape = np.random.normal(shape, std_shape)
    while fuzzy_shape < 0.01:
        fuzzy_shape = np.random.normal(shape, std_shape)

    fuzzy_scale = max(np.random.normal(scale, std_scale), 0.1)

    costs = np.zeros(duration, dtype=float)

    # calculate the yearly cost, maintaining a cumulative record to ensure the entire cost is distributed
    cumulative_cost = 0
    prev_sample = 0
    for t in range(duration - 1):
        # estimate the value of the Weibull distribution given the completion percentage (i.e., cumulative distribution)
        sample = _weibull_estimate((t + 1) / duration, fuzzy_shape, fuzzy_scale)
        # use the difference between subsequent samples as the total proportion of cost in the current year
        cost = floor((sample - prev_sample) * total_cost)
        costs[t] = cost
        cumulative_cost += cost
        prev_sample = sample

    # ensure the total cost is accounted for
    costs[duration - 1] = total_cost - cumulative_cost

    return costs


def _weibull_estimate(cdf_value, shape=1.589, scale=0.71):
    """
    Estimate the value of the Weibull distribution, given a value of the CDF. That is, given a cdf_value in the range
    [0, 1], representing the cumulative completion percentage, calculate the sample value from the Weibull distribution
    that corresponds to the supplied CDF value.

    :param cdf_value: The CDF value to invert.
    :param shape: The shape parameter for the Weibull distribution.
    :param scale: The scale parameter for the Weibull distribution.

    :return: The sample value of the Weibull distribution corresponding to the CDF value provided.
    """
    if cdf_value < 0:
        warn("time < 0 in _weibull_estimate. Returning 0.")
        return 0
    elif cdf_value >= 1:
        return 1

    num = 1 - exp(-(cdf_value / scale) ** shape)
    denom = 1 - exp(-(1 / scale) ** shape)

    return num / denom
