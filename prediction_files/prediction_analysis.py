import numpy as np


def normalized_rms_error(prediction, actual):
    """
    Normalized RMS error in prediction is
    $$ e_j = \frac{||\bf{x}_j^{pred} - \bf{\hat{x}}_j||}{\sqrt{\langle ||\bf{\hat{x}}_j||^2 \rangle}} $$
    where $||...||$ is the Euclidean norm, $\langle ... \rangle$ is an average over prediction interval
    (Wikner et al. 2021, p. 14).
    """
    numerator = np.linalg.norm(prediction - actual, axis=0)
    denominator = np.sqrt(np.mean(np.linalg.norm(actual, axis=0) ** 2))
    return numerator / denominator


def valid_time(prediction, actual, prediction_times, threshold=0.9):
    """
    Valid time is the time at which normalized RMS error first exceeds a threshold $\kappa = 0.9$
    (Wikner et al. 2021, p. 14).
    """
    err = normalized_rms_error(prediction, actual)
    index = np.argmax(err > threshold)
    if index == 0:
        # raise Exception('Error never exceeds threshold')
        print("Error never exceeds threshold")
        return np.nan
    else:
        return prediction_times[index]


def find_peaks(series, times=None):
    peak_indices = []
    peak_values = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i - 1] and series[i] > series[i + 1]:
            peak_indices.append(i)
            peak_values.append(series[i])
    if times is not None:
        peak_times = list(map(lambda i: times[i], peak_indices))
        return np.array([peak_values, peak_times])
    else:
        return np.array([peak_values, peak_indices])

