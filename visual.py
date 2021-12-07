"""
Lang Qin, Jasmine Xie, Wenjin Lyu
CSE 163 A
Final Project/Visualization Module

This program provides methods to plot graphs of possible correlation between
suicide number and sex, time, age, country, and gdp_per_captia.
"""
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def merge_data(df):
    """
    merge geopandas with pandas
    """
    shape = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # manipulate the country names
    country_data = list(df['country'].unique())
    country_geo = list(shape['name'])
    country_diff = [country for country in country_data
                    if country not in country_geo]
    # print the country with different names
    print('\nCountries with different name:')
    print(country_diff)
    print()
    # convert the names
    names = {'United States': 'United States of America',
             'Russian Federation': 'Russia',
             'Republic of Korea': 'Korea',
             'Czech Republic': 'Czech Rep.',
             'Bosnia and Herzegovina': 'Bosnia and Herz.',
             'Dominica': 'Dominican Rep.'}
    temp = pd.DataFrame(df['country'].replace(names))
    df['country'] = temp
    # merge two dataset
    df['country'] = temp
    merged_df = shape.merge(df, left_on='name',
                            right_on='country', how='left')
    return merged_df


def plot_geo_data_sum(data):
    """
    plot the map of number of suicide people over time
    """
    df = merge_data(data.groupby('country')['suicides_no'].sum().reset_index())
    fig, ax = plt.subplots(1)
    df.plot(ax=ax, color='grey')
    df.plot(column='suicides_no', legend=True, ax=ax)
    ax.set_title('Map of Suicide number per Country over Time')
    fig.savefig('mapovertime.png')


def plot_suicide_overtime(data):
    """
    plot the linechart of suicide number per country over time
    """
    fig, ax = plt.subplots(1, figsize=(20, 30))
    sns.lineplot(data=data, x='year', y='suicides_no', hue='country', ax=ax)
    ax.set_title('Suicide Number of Countries over Time', fontsize=30)
    fig.savefig('numberovertime.png')


def plot_suicide_top10(data):
    """
    plot the top 10 countries with suicide number
    """
    data = data.groupby('country')['suicides_no'].sum()\
        .nlargest(10).reset_index()
    fig, ax = plt.subplots(1, figsize=(20, 20))
    sns.barplot(x='country', y='suicides_no', data=data)
    ax.set_title('TOP 10 Coutries by Suicide Number', fontsize=30)
    fig.savefig('top10.png')


def plot_agegroup_overtime(data):
    """
    plot the global suicide number per age group over time
    """
    fig, ax = plt.subplots(1, figsize=(30, 10))
    ax.set_title('The distribution of suicides by age groups', fontsize=30)
    sns.set(font_scale=2)
    sns.barplot(data=data, y='suicides_no', x='year', hue='age',
                ax=ax, palette='deep')
    plt.ylabel('Count of suicides')
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig('age.png')


def plot_US_suicide_per_sex_overtime(data):
    """
    plot the suicide number in US per sex
    """
    data = data[data['country'] == 'United States']
    fig, ax = plt.subplots(1, figsize=(15, 15))
    sns.lineplot(data=data, x='year', y='suicides_no', hue='sex', ax=ax)
    ax.set_title('Suicide Number of US per Sex over Time')
    fig.savefig('USSex.png')


def plot_sex_overtime(data):
    """
    plot global suicide number per sex overtime
    """
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.lineplot(data=data, x='year', y='suicides_no', hue='sex', ax=ax)
    ax.set_title('Suicide Number Sex over Time', fontsize=30)
    fig.savefig('sexovertime.png')


def plot_US_suicide_GDP(data):
    """
    plot Suicide Number VS GDP in US over Time
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    data = data[data['country'] == 'United States']
    sns.lineplot(data=data, x='year', y='suicides_no', ax=ax)
    sns.lineplot(data=data, x='year', y='gdp_per_capita', ax=ax)
    plt.legend(labels=['suicides_no', 'gdp_per_capita'])
    ax.set_title('Suicide Number VS GDP in US over Time')
    fig.savefig('USGDP.png')


def plot_global_suicide_overtime(data):
    """
    plot global suicide number over time
    """
    data = data.groupby('year')['suicides_no'].sum().reset_index()
    data = data[data['year'] < 2016]
    fig, ax = plt.subplots(1, figsize=(20, 10))
    sns.lineplot(data=data, x='year', y='suicides_no', ax=ax)
    ax.set_title('Global Suicide Number over Time', fontsize=30)
    fig.savefig('overtime.png')
