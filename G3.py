from pandas import read_csv
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from scipy.stats import weibull_min

from emp import r_norm, r_g3s, r_g3, g3, g3s, r_weibull, r_gamma


# file_name = 'data/loss.csv'
file_name = 'data/select.csv'
# file_name = 'data/voltage.csv'
# file_name = 'data/capture.csv'
dataset = read_csv(file_name)
array = dataset.values
x = []
femp = []
for i in range(len(array[:, 0])):
    x.append(array[i, 0])
    femp.append(array[i, 1])
x0 = np.array([1.3, 1.0])
res_norm = minimize(r_norm, x0, method='Nelder-Mead')
print('norm_CDF', res_norm)

plt.plot(x, femp)
plt.ylim([0, 1.1])
plt.xlim([x[0] - 0.1 * x[0], x[-1] + 0.1 * x[-1]])
y1 = stats.norm.cdf(x, loc=res_norm.x[0], scale=res_norm.x[1])
plt.plot(x, y1)

k = 0.5
m = res_norm.x[0]
alfa = res_norm.x[1]
x0 = np.array([k, m, alfa])
res_g3s = minimize(r_g3s, x0, method='Nelder-Mead')
print('g3s_CDF', res_g3s)
t = [res_g3s.x[0], res_g3s.x[1], res_g3s.x[2]]
plt.plot(x, g3s(t))


x0 = np.array([2, 3])
res_weibull = minimize(r_weibull, x0, method='Nelder-Mead')
print('weibull_CdF', res_weibull)
t = [res_weibull.x[0], res_weibull.x[1]]
y_weibull = weibull_min.cdf(x, t[0], loc=0, scale=t[1])
plt.plot(x, y_weibull)

x0 = np.array([1, 1])
res_gamma = minimize(r_gamma, x0, method='Nelder-Mead', options={'maxiter': 2000})
print('gamma_CDF', res_gamma)
t = [res_gamma.x[0], res_gamma.x[1]]
y_gamma = stats.gamma.cdf(x, a=t[0], loc=0, scale=t[1])
plt.plot(x, y_gamma)


x0 = np.array([res_gamma.x[0], 1, res_gamma.x[1]])
res_g3 = minimize(r_g3, x0, method='Nelder-Mead', options={'maxiter': 2000})
print('g3_CDF', res_g3)
t = [res_g3.x[0], res_g3.x[1], res_g3.x[2]]
plt.plot(x, g3(t))

plt.show()
