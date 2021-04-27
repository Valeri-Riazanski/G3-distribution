import math
from pandas import read_csv
import scipy.stats as stats
from scipy.stats import weibull_min


def r_norm(t):
    a = t[0]
    s = t[1]
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    num = len(array[:, 0])
    for i in range(num):
        x.append(array[i, 0])
        femp.append(array[i, 1])
    summa = 0.0
    y = stats.norm.cdf(x, loc=a, scale=s)
    for i in range(len(x)):
        summa += abs(femp[i] - y[i])
    return summa / num * 100


def r_g3(t):
    k = t[0]
    n = t[1]
    teta = t[2]
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    num = len(array[:, 0])
    for i in range(num):
        x.append(array[i, 0])
        x[i] = x[i] ** (1 / n)
        femp.append(array[i, 1])
    summa = 0.0
    y = stats.gamma.cdf(x, a=k, loc=0, scale=teta)
    for i in range(len(x)):
        summa += abs(femp[i] - y[i])
    return summa / num * 100


def g3(t):
    k = t[0]
    n = t[1]
    teta = t[2]
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    for i in range(len(array[:, 0])):
        x.append(array[i, 0])
        x[i] = x[i] ** (1 / n)
    y = stats.gamma.cdf(x, a=k, loc=0, scale=teta)
    return y


def r_g3s(t):
    k = t[0]
    m = t[1]
    alfa = (t[2] * math.sqrt(2)) ** (1 / k)
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    num = len(array[:, 0])
    for i in range(num):
        x.append(array[i, 0])
        femp.append(array[i, 1])
    x2 = []
    x3 = []
    num = len(x)
    for i in range(num):
        if x[i] <= m:
            x2.append(abs(x[i] - m) ** (1 / k))
        else:
            x3.append(abs(x[i] - m) ** (1 / k))
    y2 = stats.gamma.cdf(x2, a=k, loc=0, scale=alfa).tolist()
    z2 = []
    z2.extend(y2)
    for i in range(len(y2)):
        z2[i] = (1 - z2[i]) / 2
    y3 = stats.gamma.cdf(x3, a=k, loc=0, scale=alfa).tolist()
    z3 = []
    z3.extend(y3)
    for i in range(len(y3)):
        z3[i] = (1 + z3[i]) / 2
    z2.extend(z3)
    summa = 0.0
    n_2 = len(z2)
    for i in range(n_2):
        summa += abs(femp[i] - z2[i])
    return summa / n_2 * 100


def g3s(t):
    k = t[0]
    m = t[1]
    alfa = (t[2] * math.sqrt(2)) ** (1 / k)
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    for i in range(len(array[:, 0])):
        x.append(array[i, 0])
        femp.append(array[i, 1])
    x2 = []
    x3 = []
    num = len(x)
    for i in range(num):
        if x[i] <= m:
            x2.append(abs(x[i] - m) ** (1 / k))
        else:
            x3.append(abs(x[i] - m) ** (1 / k))
    y2 = stats.gamma.cdf(x2, a=k, loc=0, scale=alfa).tolist()
    z2 = []
    z2.extend(y2)
    for i in range(len(y2)):
        z2[i] = (1 - z2[i]) / 2
    y3 = stats.gamma.cdf(x3, a=k, loc=0, scale=alfa).tolist()
    z3 = []
    z3.extend(y3)
    for i in range(len(y3)):
        z3[i] = (1 + z3[i]) / 2
    z2.extend(z3)
    return z2


def r_weibull(t):
    a = t[0]
    b = t[1]
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    num = len(array[:, 0])
    for i in range(num):
        x.append(array[i, 0])
        femp.append(array[i, 1])
    y = weibull_min.cdf(x, a, loc=0, scale=b)
    summa = 0.0
    for i in range(num):
        summa += abs(femp[i] - y[i])
    return summa / num * 100


def r_gamma(t):
    k = t[0]
    alfa = t[1]
    file_name = 'data/empiric_fun.csv'
    dataset = read_csv(file_name)
    array = dataset.values
    x = []
    femp = []
    num = len(array[:, 0])
    for i in range(num):
        x.append(array[i, 0])
        femp.append(array[i, 1])
    y = stats.gamma.cdf(x, a=k, loc=0, scale=alfa)
    summa = 0.0
    for i in range(num):
        summa += abs(femp[i] - y[i])
    return summa / num * 100
