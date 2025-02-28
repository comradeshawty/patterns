import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def pearson_correlation_scatterplot(df: pd.DataFrame, cols: list, 
                                    title: str = "Pearson Correlation Scatterplot", 
                                    xlabel: str = None, ylabel: str = None):
    """
    Creates a scatterplot for two specified columns from the DataFrame, 
    calculates the Pearson correlation coefficient, and displays the plot.
    
    Parameters:
      df : pd.DataFrame
          The input DataFrame containing the data.
      cols : list
          List of two column names (as strings) for which the scatterplot and correlation
          coefficient will be computed.
      title : str, optional
          The title for the scatterplot. Default is "Pearson Correlation Scatterplot".
      xlabel : str, optional
          Label for the x-axis. If None, the first column name from cols is used.
      ylabel : str, optional
          Label for the y-axis. If None, the second column name from cols is used.
    
    Returns:
      corr_coef : float
          The computed Pearson correlation coefficient.
      p_value : float
          The p-value for testing non-correlation.
    """
    if len(cols) != 2:
        raise ValueError("Exactly two columns must be provided for Pearson correlation.")
    
    x_col, y_col = cols
    x = df[x_col]
    y = df[y_col]
    
    # Calculate the Pearson correlation coefficient and p-value.
    corr_coef, p_value = pearsonr(x, y)
    
    # Setup axis labels if not provided.
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
        
    # Create scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, c='blue', edgecolors='w', s=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Display the correlation coefficient on the plot.
    plt.text(0.05, 0.95, f"Pearson r = {corr_coef:.3f}\np-value = {p_value:.3g}", 
             transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return corr_coef, p_value

from scipy.stats import spearmanr
def spearman_correlation_scatterplot(df: pd.DataFrame, cols: list, 
                                       title: str = "Spearman Rank Correlation Scatterplot", 
                                       xlabel: str = None, ylabel: str = None):
    """
    Creates a scatterplot for two specified columns from the DataFrame, 
    calculates the Spearman's rank correlation coefficient, and displays the plot.
    
    Parameters:
      df : pd.DataFrame
          The input DataFrame containing the data.
      cols : list
          List of two column names (as strings) for which the scatterplot and correlation
          coefficient will be computed.
      title : str, optional
          The title for the scatterplot. Default is "Spearman Rank Correlation Scatterplot".
      xlabel : str, optional
          Label for the x-axis. If None, the first column name from cols is used.
      ylabel : str, optional
          Label for the y-axis. If None, the second column name from cols is used.
    
    Returns:
      corr_coef : float
          The computed Spearman rank correlation coefficient.
      p_value : float
          The p-value for testing non-correlation.
    """
    if len(cols) != 2:
        raise ValueError("Exactly two columns must be provided for Spearman correlation.")
    
    x_col, y_col = cols
    x = df[x_col]
    y = df[y_col]
    
    # Calculate Spearman's rank correlation coefficient and corresponding p-value.
    corr_coef, p_value = spearmanr(x, y)
    
    # Setup axis labels if not provided.
    if xlabel is None:
        xlabel = x_col
    if ylabel is None:
        ylabel = y_col
        
    # Create scatter plot.
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7, c='green', edgecolors='w', s=100)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Display the correlation coefficient on the plot.
    plt.text(0.05, 0.95, f"Spearman r = {corr_coef:.3f}\np-value = {p_value:.3g}", 
             transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.5))
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return corr_coef, p_value

def plot_visit_counts(mp):
    """Plots violin + strip plot for Median RAW_VISIT_COUNTS across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="adjusted_raw_visit_counts", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="adjusted_raw_visit_counts", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of Median Visit Counts by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Median Visit Counts")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_dwell_time(mp):
    """Plots violin + strip plot for Weighted Median Dwell Time across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="MEDIAN_DWELL", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="MEDIAN_DWELL", palette="tab20", size=4, jitter=True, alpha=0.8)
    plt.title("Distribution of Weighted Median Dwell Time by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Weighted Median Dwell Time")
    plt.xticks(rotation=45, ha="right")
    plt.show()
def plot_distance_from_home(mp):
    """Plots violin + strip plot for Weighted Median Distance from Home across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="median_distance_from_home", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="median_distance_from_home", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of Median Distance from Home by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Median Distance from Home")
    plt.xticks(rotation=45, ha="right")
    plt.show()
def plot_seg(mp):
    """Plots violin + strip plot for income segregation across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp.dropna(subset='income_segregation'), x="place_category", y="income_segregation", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp.dropna(subset='income_segregation'), x="place_category", y="income_segregation", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of income_segregation by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("income_segregation")
    plt.xticks(rotation=45, ha="right")
    plt.show()
def plot_racial_entropy(mp):
    """Plots violin + strip plot for income segregation across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp.dropna(subset='local_entropy'), x="place_category", y="local_entropy", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp.dropna(subset='local_entropy'), x="place_category", y="local_entropy", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of racial entropy by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("local_entropy")
    plt.xticks(rotation=45, ha="right")
    plt.show()
