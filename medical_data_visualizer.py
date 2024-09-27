import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Load the dataset
df = pd.read_csv('medical_examination.csv')

# 2 - Add the 'overweight' column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3 - Normalize cholesterol and glucose data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4 - Function to draw the categorical plot
def draw_cat_plot():
    # 5 - Melt the dataframe for categorical plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group the data by cardio, variable, and value, and count occurrences
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7 - Draw the catplot (returns a FacetGrid object)
    grid = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

    # 8 - Extract the Figure object from the FacetGrid
    fig = grid.fig

    # 9 - Save the figure
    fig.savefig('catplot.png')
    return fig

# 10 - Function to draw the heatmap
def draw_heat_map():
    # 11 - Clean the data by filtering out incorrect segments
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Remove 'BMI' column, as the test doesn't expect it
    df_heat = df_heat.drop(columns=['BMI'])

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15 - Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', ax=ax, cmap='coolwarm')

    # 16 - Save the figure
    fig.savefig('heatmap.png')
    return fig
