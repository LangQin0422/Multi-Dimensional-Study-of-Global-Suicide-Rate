"""
Lang Qin, Jasmine Xie, Wenjin Lyu
CSE 163 A
Final Project/Main

This file is the main program to perform data cleaning, visulization, and
machine learning.
This program also contains preview of the dataset and possible correlation
between columns from the dataset.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import visual
import machine_learning


def clean_data(df):
    """
    returns four special dataset:
        annual_per_country: dataset contains suicide information per country
                            per year with annual gdp per capita
        annual_sex_country: dataset contains suicide information per sex per
                            country overtime
        annual_per_sex: dataset contains global suicide information per sex
                        overtime
        annual_per_age: dataset contains global suicide information per year
    """
    # find annual suicide rate per country with GDP
    annual_per_country = df.groupby(['country', 'ye' +
                                     'ar'])['suicides_no'].sum().reset_index()
    temp = df[(df['sex'] == 'male') &
              (df['age'] == '15-24 years')].reset_index()
    annual_per_country['gdp_per_capita'] = temp['gdp_per_capita ($)']

    # find annual suicide rate per sex per country
    annual_sex_country = df.groupby(['year', 'country',
                                     'sex'])['suicides_no'].sum().reset_index()

    # find annual suicide rate per sex
    annual_per_sex = df.groupby(['year',
                                 'sex'])['suicides_no'].sum().reset_index()

    # find annual suicide rate per age range
    annual_per_age = df.groupby(['year',
                                 'age'])['suicides_no'].sum().reset_index()

    return (annual_per_country, annual_sex_country,
            annual_per_sex, annual_per_age)


def preview(df):
    """
    previews the missing data and find possible correlations between columns
    plots heat map of correlations between columns
    """
    # specify the colours - yellow is missing; blue is not.
    cols = df.columns[:30]
    colours = ['#000099', '#ffff00']
    fig_1, ax1 = plt.subplots(1, figsize=(10, 10))
    sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours), ax=ax1,
                cbar=False)
    plt.xticks(rotation=-45)
    ax1.set_title('Specify Missing Data')
    fig_1.savefig('preview.png')

    # find possible correlation between all columns in the dataset
    find_correlation(df, 'preview_corr.png')


def find_correlation(data, name):
    """
    plots heat map of correlation between coloums of the dataset
    """
    fig_1, ax1 = plt.subplots(1, figsize=(10, 10))
    corr = data.corr()
    sns.heatmap(ax=ax1, data=corr, xticklabels=corr.columns,
                yticklabels=corr.columns, fmt='.4f', annot=True)
    plt.xticks(rotation=-45)
    ax1.set_title('Possible Correlation Map')
    fig_1.savefig(name)


def main():
    # clean data
    df = pd.read_csv('master.csv')
    annual_per_country, annual_sex_country, annual_per_sex, \
        annual_per_age = clean_data(df)
    preview(df)
    find_correlation(annual_per_country, 'country.png')

    # visualization
    visual.plot_geo_data_sum(annual_per_country)
    visual.plot_suicide_overtime(annual_per_country)
    visual.plot_agegroup_overtime(annual_per_age)
    visual.plot_US_suicide_per_sex_overtime(annual_sex_country)
    visual.plot_sex_overtime(annual_per_sex)
    visual.plot_US_suicide_GDP(annual_per_country)
    visual.plot_global_suicide_overtime(annual_per_country)
    visual.plot_suicide_top10(annual_per_country)

    # machine learning
    # Linear Regression
    print('RMSE by Linear Regression for Suicide Number:')
    print('    ' + str(machine_learning.basic(df)))
    print()
    # kNeiborrsClassifier
    test_error, error_2016 = machine_learning.advanced(annual_per_country)
    print('RMSE by KNeighborsClassifier between Suicide' +
          'Number and GDP per Capita with test data:')
    print('    ' + str(test_error))
    print('RMSE by KNeighborsClassifier between Suicide' +
          'Number and GDP per Capita at 2016 and 2018:')
    print('    ' + str(error_2016))


if __name__ == '__main__':
    main()
