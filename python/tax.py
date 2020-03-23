# https://www.freee.co.jp/kb/kb-payroll/the-deduction-for-employment-income/
# see: 

import numpy as np

def calc_kyuyoshotokukouzyo(kyuyo):
    tmp = kyuyo * 0.4 -10
    return max(tmp, 55)

def calc_shotokuzei(kyuyo):
    tax = (kyuyo - calc_kyuyoshotokukouzyo(kyuyo)) * 0.05
    return tax

def calc_hokenryo(x):
    if x < 1300000:
        return 0
    lst_tau = np.array([98000, 104000, 110000, 118000, 126000])
    lst_val = np.array([16104, 17934, 19032, 20130, 21594])
    n = len(lst_val)

    ave = x/12.0
    isLower = ave < lst_tau
    for i in range(n):
        if isLower[i]:
            return lst_val[i] * 12.0
    raise NotImplementedError()

def calc_juminzei(x, shotokukouzyo=380000):
    tax_common = 3500 + 1500
    tax_depedent = (x - shotokukouzyo) * 0.1
    return tax_common + tax_depedent

support = False
if support:
    x = 1200000
    tax = calc_shotokuzei(x) + calc_hokenryo(x)
    money = x - tax + 0.5 * (26 + 53)
    print(money)
else:
    x = 1440000
    tax = calc_shotokuzei(x) + calc_hokenryo(x)
    money = x - tax + 0.5 * (26 + 53)
    print(money)





