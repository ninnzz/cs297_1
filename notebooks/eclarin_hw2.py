"""
Nino Eclarin
CS 297 HW2
Use python 3.6+
"""
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, RidgeCV

FILE_NAME = '../data/Sample_Data.xlsx'
MODEL_COUNT = 5


def print_result(r):
    print('Model used: {}'.format(r['model_name']))
    print('Prediction score on the test set: {}'.format(r['score']))
    print('Mean squared error: {}'.format(r['mse']))


def analysis(X_tr, y_tr, X_te, y_te):
    results = []

    models = [
        {'model': LinearRegression(), 'name': 'Linear Regression'},
        {'model': Ridge(alpha=0.05), 'name': 'Ridge'},
        {'model': Lasso(alpha=0.05), 'name': 'Lasso'},
        {'model': KNeighborsRegressor(n_neighbors=3), 'name': 'KNR'},
        {'model': RidgeCV(alphas=[0.01, 0.1, 10, 100], cv=5), 'name': 'RidgeCV'}
    ]

    for item in models:
        item['model'].fit(X_tr, y_tr)
        y_pre = item['model'].predict(X_te)
        results.append({
            'score': item['model'].score(X_te, y_te),
            'mse': mean_squared_error(y_te, y_pre),
            'model_name': item['name']
        })
        
    return results


def main(show_details=True):
    feature_size = 10
    df = pd.read_excel(FILE_NAME)

    # Pre-process file
    # No generalization assumption
    # was made for the features
    df_x = df.iloc[:, 0:feature_size]
    Y = df['Y']
    df_label = df['Label']
    labels = df.Label.unique()


    components = 1
    xes_len = len(df_x.columns)

    while components < xes_len:
        pca = PCA(n_components=components).fit(df_x)
        variance = pca.explained_variance_ratio_
        if sum(variance) >= 0.95:
            break
        
        components += 1
    else:
        # Maybe its not needed
        print('WARNING: The variance did not reach 95%')
        exit(1)
    
    print('====================== STEP 1 ======================')
    print('{} components are needed to reach 95%'.format(components))

    new_x = pca.transform(df_x)

    # Prepares a new dataframe with PCA
    # Makes it easier to group code later
    _tmp = {'X{}'.format(i + 1): new_x[:, i] for i in range(components)}
    pca_df = pd.DataFrame(_tmp)
    pca_df['Label'] = df_label
    pca_df['Y'] = Y


    # Apply models to ungrouped dataset
    X_train, X_test, y_train, y_test = train_test_split(new_x, Y, test_size=0.3, random_state=1234)
    all_results = analysis(X_train, y_train, X_test, y_test)

    print('\n\n')
    print('====================== STEP 2 ======================')
    print('==================== UNGROUPED =====================')
    for result in all_results:
        print_result(result)


    groupings = []
    grouped_result = []

    for i in range(MODEL_COUNT):
        grouped_result.append({
            'score': 0,
            'mse': 0,
            'model_name': None
        })


    print('\n')
    print('====================== STEP 3 ======================')    
    for l in labels:
        l2 = pca_df[pca_df['Label'] == l]
        print('\n+++++++++++++ {} +++++++++++++'.format(l))
        print('Number of instances for group "{}": {}'.format(l, len(l2)))
        _new_x = l2.iloc[:,0:components]
        _Y = l2['Y']
        _df_label = l2['Label']
        
        _X_train, _X_test, _y_train, _y_test = train_test_split(_new_x, _Y, test_size=0.3, random_state=1234)
        
        _all_results = analysis(_X_train, _y_train, _X_test, _y_test)

        for i, result in enumerate(_all_results):

            if show_details:
                print_result(result)

            grouped_result[i]['score'] += result['score']
            grouped_result[i]['mse'] += result['mse']
            grouped_result[i]['model_name'] = result['model_name']

    print('\n')
    print('====================== STEP 4 ======================')
    for i, result in enumerate(grouped_result):
        print('Model used: {}'.format(result['model_name']))
        print('MSE(Sum) of grouped data: {}'.format(result['mse']))
        print('MSE of ungrouped whole data: {}'.format(all_results[i]['mse']))
        diff = ((all_results[i]['mse'] - result['mse']) / all_results[i]['mse']) * 100
        print('Difference: ', all_results[i]['mse'] - result['mse'])

        if diff > 0:
            print('Accuracy improved by: {}%'.format(round(abs(diff), 2)))
        elif diff < 0:
            print('Accuracy decreased by: {}%'.format(round(abs(diff), 2)))
        else:
            print('Overall MSE stayed the same')

        print('\n')


if __name__ == '__main__':
    main()