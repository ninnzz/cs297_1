"""
Nino Eclarin
CS 297 HW
"""

import os
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def get_x(X ,slope, intercept):
    return np.dot(X, slope) + intercept


def main():
    _ts = 0.4
    _rs = 1234
    trains = []
    tests = []
    regs = []

    filename = 'Example1.csv'
    df = pd.read_csv(filename)

    # Transform/reshape the data, then split
    X = np.array(df['X1']).reshape(-1, 1)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=_ts, random_state=_rs)

    regs.append([LinearRegression().fit(X_train, y_train), 'LinReg'])
    regs.append([Ridge(alpha=0.5).fit(X_train, y_train), 'Ridge Alpha 0.5'])
    regs.append([Ridge(alpha=50).fit(X_train, y_train), 'Ridge Alpha 50'])
    regs.append([Lasso(alpha=0.5).fit(X_train, y_train), 'Lasso Alpha 0.5'])
    regs.append([Lasso(alpha=50).fit(X_train, y_train), 'Ridge Alpha 50'])

    plt.rcParams.update({'font.size': 30})
    fig, axes = plt.subplots(ncols=2, figsize=(60, 25))

    # plt.tight_layout()
    
    axes[0].scatter(X_train, y_train, s=150)
    axes[1].scatter(X_test, y_test, s=150)

    for algo in regs:
        axes[0].plot(
            X_train, 
            get_x(X_train, algo[0].coef_, algo[0].intercept_), 
            label=algo[1]
        )

        axes[1].plot(
            X_test, 
            get_x(X_test, algo[0].coef_, algo[0].intercept_), 
            label=algo[1]
        )
        
    axes[0].set_title('TRAINING DATA')
    axes[1].set_title('TEST DATA')
    axes[0].legend(prop={'size': 20})
    axes[1].legend(prop={'size': 20})

    plt.savefig('NinzExample1.png')

if __name__ == '__main__':
    main()