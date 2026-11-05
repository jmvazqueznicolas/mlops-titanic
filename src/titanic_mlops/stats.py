"""A metric is an estimator: report it with an interval, compare models on paired data."""
import math

from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test on the passengers where the two models disagree."""
    b = int(sum(a and not c for a, c in zip(correct_a, correct_b)))   # A right, B wrong
    c = int(sum(c and not a for a, c in zip(correct_a, correct_b)))   # B right, A wrong
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return b, c, p
