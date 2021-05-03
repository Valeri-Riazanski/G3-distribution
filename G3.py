import math

import scipy.special
from pandas import read_csv
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize, fsolve
from matplotlib import pyplot as plt
from scipy.stats import weibull_min
from math import log, exp

from emp import r_norm, r_g3s, r_g3, g3, g3s, r_weibull, r_gamma

# file_name = 'data/loss.csv'
# file_name = 'data/select.csv'
file_name = 'data/voltage.csv'
file_name_full = 'data/voltage_full.csv'
# file_name = 'data/capture.csv'
dataset = read_csv(file_name)
data_full = read_csv(file_name_full)
array = dataset.values
array_full = data_full.values
x = []
x_full = []
femp = []
for i in range(len(array[:, 0])):
    x.append(array[i, 0])
    femp.append(array[i, 1])
for i in range(len(array_full[:, 0])):
    x_full.append(array_full[i, 0])

x0 = np.array([150, 30])
res_norm = minimize(r_norm, x0, method='Nelder-Mead')
print('norm_CDF', 'fun=', res_norm.fun, 'x=', res_norm.x[0], res_norm.x[1])

plt.plot(x, femp)
plt.ylim([0, 1.1])
plt.xlim([x[0] - 0.1 * x[0], x[-1] + 0.1 * x[-1]])
y1 = stats.norm.cdf(x, loc=res_norm.x[0], scale=res_norm.x[1])
# plt.plot(x, y1)

k = 0.5
m = res_norm.x[0]
alfa = res_norm.x[1]
x0 = np.array([k, m, alfa])
res_g3s = minimize(r_g3s, x0, method='Nelder-Mead')
print('g3s_CDF', 'fun=', res_g3s.fun, 'x=', res_g3s.x[0], res_g3s.x[1], res_g3s.x[2])


# t = [res_g3s.x[0], res_g3s.x[1], res_g3s.x[2]]
# plt.plot(x, g3s(t))


# оценки параметров a, b по методу МП для распределения Weibulla
def equations(u):
    e, d = u
    len_x = len(x_full)
    p = 0.0
    q = 0.0
    r = 0.0
    for j in range(len_x):
        x_e = x_full[j] ** e
        l_x = log(x_full[j])
        p += x_e
        q += x_e * l_x
        r += l_x
    p = p / len_x
    q = q / len_x
    r = r / len_x
    eq1 = d ** e - p
    eq2 = 1 / e - q / p + r
    return np.array([eq1, eq2])


a0 = np.array([1, 1])
[a_w, b_w] = fsolve(equations, a0)
print('оценка МП параметров Вейбулла a = ', a_w, 'b = ', b_w)
# y_w_omp = weibull_min.cdf(x, a_w, loc=0, scale=b_w)
# plt.plot(x, y_w_omp)


x0 = np.array([3, 3])
res_weibull = minimize(r_weibull, x0, method='Nelder-Mead')
print('weibull_CdF', 'fun=', res_weibull.fun, 'x=', res_weibull.x[0], res_weibull.x[1])
t = [res_weibull.x[0], res_weibull.x[1]]
# y_weibull = weibull_min.cdf(x, t[0], loc=0, scale=t[1])
# plt.plot(x, y_weibull)

# оценки параметров k, alfa по методу МП для гамма-распределения
a = 0
b = 0
num = len(x_full)
for i in range(num):
    a += x_full[i]
    b += math.log(x_full[i])
a = a / num
b = b / num
k_gamma = 1 / (2 * (math.log(a) - b))
alfa_gamma = a / k_gamma
print('оценка МП параметров гамма-распределения, k = ', k_gamma, 'alfa = ', alfa_gamma)
# y_gamma_mp = stats.gamma.cdf(x, a=k_gamma, loc=0, scale=alfa_gamma)
# plt.plot(x, y_gamma_mp)

x0 = np.array([k_gamma, alfa_gamma])
res_gamma = minimize(r_gamma, x0, method='Nelder-Mead', options={'maxiter': 5000})
print('gamma_CDF', 'fun =', res_gamma.fun, 'x=', res_gamma.x[0], res_gamma.x[1])
t = [res_gamma.x[0], res_gamma.x[1]]
# y_gamma = stats.gamma.cdf(x, a=t[0], loc=0, scale=t[1])
# plt.plot(x, y_gamma)


# оценки параметров k, n, alfa по методу МП для G-распределения
def equations_g(u):
    kk, nn, aa = u
    len_x = len(x_full)
    p = 0.0
    q = 0.0
    r = 0.0
    for j in range(len_x):
        x_a = (x_full[j] / aa) ** (1 / nn)
        l_x = log(x_full[j] / aa)
        p += x_a
        q += x_a * l_x
        r += l_x
    psi_k = scipy.special.digamma(kk)
    eq1 = nn * len_x * psi_k - r
    eq2 = nn * len_x * (1 + kk * psi_k) - q
    eq3 = kk * len_x - p
    return np.array([eq1, eq2, eq3])


kna0 = np.array([k_gamma, 0.93, alfa_gamma])
[k_G, n_G, a_G] = fsolve(equations_g, kna0)
print('оценка МП параметров G-распределения k = ', k_G, ' n = ', n_G, ' alfa = ', a_G)
t_g_mp = [k_G, n_G, a_G]
plt.plot(x, g3(t_g_mp))


x0 = np.array([res_gamma.x[0], 1.3, res_gamma.x[1]])
res_g3 = minimize(r_g3, x0, method='Nelder-Mead', options={'maxiter': 5000})
print('g3_CDF', 'fun = ', res_g3.fun, 'x= ', res_g3.x[0], res_g3.x[1], res_g3.x[2], res_g3.status)
# print(res_g3)
t_g = [res_g3.x[0], res_g3.x[1], res_g3.x[2]]
plt.plot(x, g3(t_g))

plt.show()
