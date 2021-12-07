"""
Lang Qin, Jasmine Xie, Wenjin Lyu
CSE 163 A
Final Project/Machine Learning Module

This file includes methods to obtain regression models by machine learning
to predict future infomration.
Two machine learning models used: LinearRegression, kNeighborsClassifier
"""
import math
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def basic(data):
    """
    machine learning with all features
    returns the RMSE of linear regression
    """
    data = data.drop(columns='HDI for year')
    features = data.loc[:, ['year', 'population', 'suicides/100k pop',
                            'gdp_per_capita ($)']]
    labels = data['suicides_no']

    model = LinearRegression()
    xtrain, xtest, ytrain, ytest = \
        train_test_split(features, labels, test_size=0.2)
    model.fit(xtrain, ytrain)
    yprediction = model.predict(xtest)
    return math.sqrt(mean_squared_error(ytest, yprediction))


def advanced(data):
    """
    find the optimal regression model by KNeighborsClassifier
    returns RMSE of the optimal model
    """
    # select data of US
    data = data[data['country'] == 'United States']

    # regression by machine learning
    x = data[['gdp_per_capita']]
    y = data[['suicides_no']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.50)

    # test k value
    rmse_values = []
    for k in range(1, 16):
        model = neighbors.KNeighborsRegressor(n_neighbors=k)
        # fit the model
        model.fit(x_train, y_train)
        # make prediction on test set
        prediction = model.predict(x_test)
        # compute RMSE
        error = math.sqrt(mean_squared_error(y_test, prediction))
        # store rmse values and corresponding k value
        rmse_values.append(error)

    # plot the graph between k-value and RMSE
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(range(1, 16), rmse_values)
    ax.set_title('RMSE by k value')
    ax.set_xlabel("K value")
    ax.set_ylabel("RMSE")
    fig.savefig('RMSE.png')

    # With plots, we clearly see at k=3, we reach minimum RMSE. Thus,
    # we conclude the optimal value for k will be around k=3.

    # test the model with test data
    model = neighbors.KNeighborsRegressor(n_neighbors=3)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    test_error = math.sqrt(mean_squared_error(y_test, prediction))
    # predict the suicide number of US in 2016 and 2018 according to
    # GDP per captia with model
    real_data = [44965, 48344]
    pre_data = model.predict([[54795.5], [62997]])
    print('\nPrdicted SuicideNumber in 2016:')
    print('    ' + str(pre_data[0][0]))
    print('Prdicted SuicideNumber in 2018:')
    print('    ' + str(pre_data[1][0]) + '\n')
    error = math.sqrt(mean_squared_error(real_data, pre_data))

    return (test_error, error)
