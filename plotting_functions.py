import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
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
