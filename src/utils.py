import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


config_dir = '../config/config.yaml'

def config_load():
    """
    Load configuration data from a YAML file.

    Parameters
    ----------
    config_dir : str
        The path to the YAML configuration file to be loaded.

    Returns
    -------
    dict
        Configuration data loaded from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the specified YAML file does not exist.
    yaml.YAMLError
        If there is an error while parsing the YAML file.

    Example
    -------
    >>> config = config_load('config.yaml')
    """
    try:
        with open(config_dir, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as fnf_error:
        raise FileNotFoundError(f"File not found: {config_dir}") from fnf_error
    except yaml.YAMLError as yaml_error:
        raise yaml.YAMLError(f"Error parsing YAML file: {config_dir}") from yaml_error

    return config

def pickle_load(file_path):
    """
    Load data from a binary pickle file using joblib.

    Parameters
    ----------
    file_path : str
        The path to the pickle file from which data will be loaded.

    Returns
    -------
    Any
        The data loaded from the pickle file. The returned object can be of any type, 
        depending on what was saved in the pickle file.

    Example
    -------
    >>> data = pickle_load('data.pkl')
    """
    return joblib.load(file_path)

def pickle_dump(data, file_path):
    """
    Save data to a binary pickle file using joblib.

    Parameters
    ----------
    data : Any
        The data to be saved in the pickle file. This can be any Python object that is serializable.
    file_path : str
        The path to the pickle file where the data will be saved. If the file does not exist, it will be created.
        If the file already exists, it will be overwritten.

    Returns
    -------
    None

    Example
    -------
    >>> data = {'key': 'value'}
    >>> pickle_dump(data, 'data.pkl')
    """
    joblib.dump(data, file_path)

def annotate_text(data, axis, prop=False, prec=False, fontsize=12):
    """
    Annotates the bar plot with counts or proportions.

    Parameters:
    - data: pandas DataFrame
        The DataFrame containing the data to be plotted.
    - axis: matplotlib.axes._subplots.AxesSubplot
        The axes object of the plot on which annotations will be added.
    - prop: bool, optional, default: False
        Indicates whether to annotate with proportions (True) or counts (False).
    - prec: bool, optional, default: False
        Indicates whether precision (decimal) or integer annotations are required.
    - fontsize: int, optional, default: 12
        The font size of the annotation text.

    Notes:
    - This function calculates the total number of observations in the DataFrame to determine 
      proportions if `prop` is set to True. Otherwise, it uses the counts directly.
    - Annotations are placed at the center and above each bar.
    - Only bars with a positive height are annotated.
    """
    total = float(len(data))  # Total number of observations

    for p in axis.patches:
        height = p.get_height()

        if prop:
            value = height / total * 100
            value = str(round(value, 2)) + '%'
        else:
            if prec:
                value = round(height, 2)
            else:
                value = round(height)
            value = str(value)

        if height > 0:
            axis.annotate(f'{value}',
                          (p.get_x() + p.get_width() / 2., p.get_y() + height),
                          ha='center', va='center',
                          xytext=(0, 5),
                          textcoords='offset points',
                          fontsize=fontsize,
                          color='black')

def check_missing_values(data):
    """
    Check for missing values in a DataFrame and return a DataFrame with the count and percentage of missing values for each column.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to check for missing values.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:
        - 'count': The number of missing values in each column.
        - '%rate': The percentage of missing values in each column, as a string with a '%' symbol.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None],
    ...     'B': [None, None, 3],
    ...     'C': [4, 5, 6]
    ... })
    >>> check_missing_values(data)
       count %rate
    B      2    66%
    A      1    33%
    C      0     0%
    """
    loan_null = data.isna().sum().sort_values(ascending=False).to_frame(name='count')
    loan_null['%rate'] = (loan_null['count'] / len(data) * 100).astype(int).astype(str) + '%'
    return loan_null

def stacked_bar_proportion(data, col, subtitle, order=[], xy=(0.07, 0.935), rot=0):
    """
    Plot the proportion of good and bad borrowers for a specified categorical column using a stacked bar chart.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    col : str
        The name of the categorical column to be analyzed.
    subtitle : str
        The subtitle to be displayed below the main title of the plot.
    order : list, optional
        The order in which the categories should be displayed. Default is an empty list, 
        which means the categories will be displayed in their natural order.
    xy : tuple, optional
        The coordinates for placing the main title. Default is (0.07, 0.935).
    rot : int, optional
        The rotation angle of the x-axis labels. Default is 0.

    Returns
    -------
    None
        This function does not return any value. It displays a bar plot.

    Example
    -------
    >>> plot_proportion(loan_df, 'emp_length', 'This is a subtitle', 
                        order=['< 1 year', '1 year', '2 years', '3 years', 
                               '4 years', '5 years', '6 years', '7 years', 
                               '8 years', '9 years', '10+ years'], 
                        rot=45)
    """
    x1, y1 = xy
    x2, y2 = x1, y1 - 0.04

    if order:
        df = data.unstack().reindex(order).stack()
    else:
        df = data

    fig, ax = plt.subplots(figsize=(10, 6))
    df.unstack().plot(kind='bar', stacked=True, width=0.8, ax=ax)
    plt.xlabel(f"{col.replace('_',' ').title()}")
    plt.ylabel("% of Customers")
    plt.xticks(rotation=rot)
    plt.figtext(x1, y1, f"Percentage of Customers by Borrower Status and {col.replace('_',' ').title()}", fontsize=16, fontweight='bold')
    plt.figtext(x2, y2, subtitle, fontsize=12, style='italic')

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., p.get_y() + 3.),
                        ha='center', va='center',
                        xytext=(0, 0),
                        textcoords='offset points',
                        fontsize=12,
                        color='black')

    plt.legend(['Bad', 'Good'], title='Borrower Status', loc=2, bbox_to_anchor=(0.905, 1.145))
    plt.tight_layout()
    plt.show()

def pie_proportion(data, col, subtitle='', fontsize=12):
    """
    Plot the proportion of categories within a specified column as a pie chart.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    col : str
        The name of the categorical column to be analyzed.
    subtitle : str, optional
        The subtitle to be displayed below the main title of the plot. Default is an empty string.
    fontsize : int, optional
        The font size for the percentage labels in the pie chart. Default is 12.

    Returns
    -------
    None
        This function does not return any value. It displays a pie chart.

    Example
    -------
    >>> pie_proportion(loan, 'grade', 'Distribution of Loan Grades', fontsize=14)
    """
    df = data[col].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(df, labels=None,
                                       autopct='%1.1f%%', startangle=90)

    plt.figtext(0, 1.015, f"Proportion of {col.replace('_', ' ').title()} with Good and Bad Borrowers", fontsize=16, fontweight='bold')
    plt.figtext(0, 0.975, subtitle, fontsize=11, style='italic')

    plt.axis('equal')
    for autotext in autotexts:
        autotext.set_fontsize(fontsize)

    plt.legend(wedges, df.index, title=f"{col.replace('_', ' ').title()}",
               bbox_to_anchor=(0.90, 0.60),
               fontsize=10)

    plt.tight_layout()
    plt.show()

def top_stacked_bar(data, col, ntop=10, xpad=50):
    """
    Plot a stacked horizontal bar chart showing the top categories of a specified column 
    with the proportion of good and bad borrowers.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the data to plot.
    col : str
        The name of the column to plot.
    ntop : int, optional, default=10
        The number of top categories to plot.
    xpad : int, optional, default=50
        Padding for the text annotations within the bars.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    top_titles = data[data['good_borrower'] == 1][col].value_counts().nlargest(ntop).index
    data[col].value_counts().loc[top_titles].plot(kind='barh', width=0.8, ax=ax)
    data[data['good_borrower'] == 1][col].value_counts().nlargest(ntop).plot(kind='barh', color='darkorange', width=0.8, ax=ax)
    ax.set_title(f"Top {ntop} {col.replace('_',' ').title()} with Good and Bad Borrowers", fontdict={'fontsize': 14, 'weight': 'bold'})
    ax.set_xlabel("Count")
    ax.set_ylabel(f"{col.replace('_',' ').title()}")
    
    for i, value in enumerate(data[col].value_counts().loc[top_titles]):
        ax.text(xpad, i, top_titles[i], va='center', ha='left', fontsize=8, color='black', weight='bold')
    
    ax.set_yticklabels([])
    
    plt.legend(['Bad', 'Good'], title='Borrower Status')
    plt.tight_layout()
    plt.show()

def date_columns(df, column):
    """
    Convert a date column in text format to the number of months since a specific date.

    This function will:
    1. Convert the date column from text format (e.g., "Jan-20") to datetime type.
    2. Calculate the number of months since a specific date (in this case, August 1, 2020).
    3. Add a new column named 'mths_since_' followed by the original column name, which stores the number of months that have elapsed.
    4. Handle cases where the number of months calculated is negative by replacing it with the maximum value in the column.
    5. Drop the original date column from the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the date column.
    column : str
        Name of the column in the DataFrame that contains the dates in text format to be converted.

    Returns
    -------
    None
        This function modifies the DataFrame in place.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = pd.DataFrame({
    ...     'date_column': ['Jan-20', 'Feb-20', 'Mar-20']
    ... })
    >>> date_columns(data, 'date_column')
    >>> data
       mths_since_date_column
    0                      7
    1                      6
    2                      5
    """
    def calculate_months_diff(date1, date2):
        return (date1.year - date2.year) * 12 + date1.month - date2.month

    today_date = pd.to_datetime('2020-08-01')
    df[column] = pd.to_datetime(df[column], format="%b-%y")
    df['mths_since_' + column] = df[column].apply(lambda x: calculate_months_diff(today_date, x))
    df['mths_since_' + column] = df['mths_since_' + column].apply(lambda x: df['mths_since_' + column].max() if x < 0 else x)
    df.drop(columns=[column], inplace=True)

def dist_mean_numerical(data, col, subtitle, order=[], xy=(0.07, 1.035), prec=False, rot=0):
    """
    Plot the distribution and average of a numerical column for good and bad borrowers.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data to be plotted.
    col : str
        The name of the numerical column to be analyzed.
    subtitle : str
        The subtitle to be displayed below the main title of the plot.
    order : list, optional
        The order in which the categories should be displayed. Default is an empty list.
    xy : tuple, optional
        The coordinates for placing the main title. Default is (0.07, 1.035).
    - prec: bool, optional, default: False
        Indicates whether precision (decimal) or integer annotations are required.
    rot : int, optional
        The rotation angle of the x-axis labels in the bar plot. Default is 0.

    Returns
    -------
    None
        This function does not return any value. It displays a histogram and a bar plot.

    Example
    -------
    >>> dist_mean_numerical(loan_df, 'loan_amnt', 'Subtitle for loan amount distribution')
    """
    
    x1, y1 = xy
    x2, y2 = x1, y1 - 0.04

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.hist(x=col, data=data[data['good_borrower'] == 1], bins=30, color='#FF8C00')
    ax1.hist(x=col, data=data[data['good_borrower'] == 0], bins=30, color='#1f77b4')
    
    data.groupby(['good_borrower'])[col].mean().plot(kind='bar', ax=ax2, color=['#1f77b4', '#FF8C00'], width=0.8)
    annotate_text(data, ax2, prec=prec)
    ax2.set_xlabel(None)
    ax2.set_xticklabels(['Bad', 'Good'], rotation=rot)
    ax2.yaxis.set_visible(False)

    plt.figtext(x1, y1, f"Distribution and Average of {col.replace('_',' ').title()} by Good and Bad Category", fontsize=16, fontweight='bold')
    plt.figtext(x2, y2, subtitle, fontsize=12, style='italic')

    fig.legend(['Good', 'Bad'], loc='upper center', bbox_to_anchor=(0.945, 1.085))
    plt.tight_layout()
    plt.show()

def iv_woe(data, target, bins=10, show_woe=False, show_iv=False):
    
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    cols = data.columns
    
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)
        
        if show_woe == True:
            print(d)
        if show_iv == True:
            print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
            
    return newDF, woeDF

def woe_discrete(df, discrete_variable_name, good_bad_variable_df):
    """
    Calculate Weight of Evidence (WoE) and Information Value (IV) for a discrete variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the discrete variable and the good/bad variable.
    discrete_variable_name : str
        The name of the discrete variable column for which to calculate WoE and IV.
    good_bad_variable_df : pandas.Series or pandas.DataFrame
        The Series or DataFrame containing the binary good/bad variable (0 = bad, 1 = good).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - discrete_variable_name: The unique values of the discrete variable.
        - n_obs: The number of observations for each unique value.
        - prop_good: The proportion of good outcomes for each unique value.
        - prop_n_obs: The proportion of total observations for each unique value.
        - n_good: The number of good outcomes for each unique value.
        - n_bad: The number of bad outcomes for each unique value.
        - prop_n_good: The proportion of total good outcomes for each unique value.
        - prop_n_bad: The proportion of total bad outcomes for each unique value.
        - WoE: The Weight of Evidence for each unique value.
        - IV: The Information Value for the discrete variable (same value for all rows).

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'feature': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'B'],
    ...     'target': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    ... })
    >>> result = woe_discrete(data, 'feature', data['target'])
    >>> print(result)
      feature  n_obs  prop_good  prop_n_obs  n_good  n_bad  prop_n_good  prop_n_bad       WoE        IV
    0       A      5        0.6        0.5     3.0    2.0          0.5         0.4  0.223144  0.022314
    1       B      5        0.4        0.5     2.0    3.0          0.5         0.6 -0.405465  0.081093
    """
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

def plot_by_woe(df_WoE, rotation_of_x_axis_labels=0):
    """
    Plot the Weight of Evidence (WoE) for a given DataFrame.

    Parameters
    ----------
    df_WoE : pandas.DataFrame
        A DataFrame containing the WoE values. It should have at least two columns:
        - The first column contains the categories or bins of the discrete variable.
        - The 'WoE' column contains the WoE values for each category or bin.
    rotation_of_x_axis_labels : int, optional
        The rotation angle for the x-axis labels, by default 0.

    Returns
    -------
    None
        This function creates a plot and does not return any value.

    Example
    -------
    >>> import pandas as pd
    >>> data = {'feature': ['A', 'B', 'C'], 'WoE': [0.2, -0.3, 0.1]}
    >>> df_WoE = pd.DataFrame(data)
    >>> plot_by_woe(df_WoE)
    """
    x = np.array(df_WoE.iloc[:, 0].apply(str))
    y = df_WoE['WoE']
    plt.figure(figsize=(18, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='k')
    plt.xlabel(df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(f'Weight of Evidence by {df_WoE.columns[0]}')
    plt.xticks(rotation=rotation_of_x_axis_labels)
    plt.show()

def woe_ordered_continuous(df, discrete_variable_name, good_bad_variable_df):
    """
    Calculate Weight of Evidence (WoE) for an ordered continuous variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the ordered continuous variable and the target variable (good/bad indicator).
    discrete_variable_name : str
        The name of the ordered continuous variable column in the DataFrame `df`.
    good_bad_variable_df : pandas.Series
        A Series or DataFrame containing the target variable (good/bad indicator).

    Returns
    -------
    pandas.DataFrame
        DataFrame with WoE values calculated for the ordered continuous variable.

    Notes
    -----
    - WoE (Weight of Evidence) is a measure of the separation of good and bad outcomes within each category of the variable.
    - IV (Information Value) quantifies the predictive power of a variable. Higher IV indicates higher predictive power.

    Example
    -------
    >>> # Example usage:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'continuous_var': [1.5, 2.1, 3.2, 4.5, 5.0],
    ...                    'target': [0, 1, 0, 1, 1]})
    >>> target = df['target']
    >>> woe_df = woe_ordered_continuous(df, 'continuous_var', target)
    >>> print(woe_df)
       continuous_var  n_obs  prop_good  prop_n_obs  n_good  n_bad  prop_n_good  prop_n_bad       WoE  diff_prop_good  diff_WoE        IV
    0             1.5      1        0.0         0.2     0.0    1.0          0.0         0.2 -1.098612             NaN       NaN  0.130039
    1             2.1      1        1.0         0.2     1.0    0.0          0.2         0.0  1.098612             1.0  2.197225  0.130039
    2             3.2      1        0.0         0.2     0.0    1.0          0.0         0.2 -1.098612             1.0  2.197225  0.130039
    3             4.5      1        1.0         0.2     1.0    0.0          0.2         0.0  1.098612             1.0  2.197225  0.130039
    4             5.0      1        1.0         0.2     1.0    0.0          0.2         0.0  1.098612             0.0  0.000000  0.130039
    """

    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis=1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

def training_model(model, X, y, result):
    """
    Trains the given model using the provided features and target variable, performs cross-validation, 
    and records the performance metrics into the result DataFrame.

    Parameters:
    model (estimator object implementing 'fit'): The machine learning model to be trained.
    X (DataFrame): The feature set used for training the model.
    y (Series): The target variable used for training the model.
    result (DataFrame): The DataFrame to store the performance metrics of the model. Default is df_model.

    Returns:
    DataFrame: Updated DataFrame with the training performance metrics of the model.
    """
    model_name = type(model).__name__ + ' - Train'
    model.fit(X, y)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

    result.loc[model_name, 'accuracy'] = cv_results['test_accuracy'].mean()
    result.loc[model_name, 'precision'] = cv_results['test_precision'].mean()
    result.loc[model_name, 'recall'] = cv_results['test_recall'].mean()
    result.loc[model_name, 'f1_score'] = cv_results['test_f1'].mean()
    result.loc[model_name, 'roc_auc_score'] = cv_results['test_roc_auc'].mean()
    
    return result

def evaluation_model(model, X, y, result):
    """
    Evaluates the given model using the provided features and target variable, 
    and records the performance metrics into the result DataFrame.

    Parameters:
    model (estimator object implementing 'predict' and 'predict_proba'): The trained machine learning model to be evaluated.
    X (DataFrame): The feature set used for evaluating the model.
    y (Series): The target variable used for evaluating the model.
    result (DataFrame): The DataFrame to store the performance metrics of the model. Default is df_model.

    Returns:
    DataFrame: Updated DataFrame with the validation performance metrics of the model.
    """
    model_name = type(model).__name__ + ' - Validation'
    y_pred = model.predict(X)

    result.loc[model_name, 'accuracy'] = accuracy_score(y, y_pred)
    result.loc[model_name, 'precision'] = precision_score(y, y_pred)
    result.loc[model_name, 'recall'] = recall_score(y, y_pred)
    result.loc[model_name, 'f1_score'] = f1_score(y, y_pred)
    result.loc[model_name, 'roc_auc_score'] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    
    return result
