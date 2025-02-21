import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import geopandas as gpd
def plot_visit_counts(mp):
    """Plots violin + strip plot for Median RAW_VISIT_COUNTS across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="Median RAW_VISIT_COUNTS", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="Median RAW_VISIT_COUNTS", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of Median Visit Counts by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Median Visit Counts")
    plt.xticks(rotation=45, ha="right")
    plt.show()

def plot_dwell_time(mp):
    """Plots violin + strip plot for Weighted Median Dwell Time across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="Weighted Median MEDIAN_DWELL", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="Weighted Median MEDIAN_DWELL", palette="tab20", size=4, jitter=True, alpha=0.8)
    plt.title("Distribution of Weighted Median Dwell Time by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Weighted Median Dwell Time")
    plt.xticks(rotation=45, ha="right")
    plt.show()
def plot_distance_from_home(mp):
    """Plots violin + strip plot for Weighted Median Distance from Home across place categories."""
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=mp, x="place_category", y="Weighted Median DISTANCE_FROM_HOME", palette="tab20", inner=None, alpha=0.7)
    sns.stripplot(data=mp, x="place_category", y="Weighted Median DISTANCE_FROM_HOME", palette="tab20", size=4, jitter=True, alpha=0.8)

    plt.title("Distribution of Weighted Median Distance from Home by Place Category", fontsize=14)
    plt.xlabel("Place Category")
    plt.ylabel("Weighted Median Distance from Home")
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

def plot_daily_trends(mp):
    """
    Plots daily visit trends across place categories with a distinct color for each category.
    """
    day_columns = ['visit_count_monday', 'visit_count_tuesday', 'visit_count_wednesday',
                   'visit_count_thursday', 'visit_count_friday', 'visit_count_saturday', 'visit_count_sunday']

    day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    mp[day_columns] = mp[day_columns].div(mp[day_columns].sum(axis=1), axis=0)

    category_trends = mp.groupby('place_category')[day_columns].mean()

    # Define a distinct color palette
    unique_categories = category_trends.index
    colors = sns.color_palette("husl", len(unique_categories))  # "husl" gives visually distinct colors
    color_dict = dict(zip(unique_categories, colors))

    # Line plot for daily trends
    plt.figure(figsize=(14, 7))
    for category in category_trends.index:
        plt.plot(day_labels, category_trends.loc[category], marker='o', label=category, color=color_dict[category])

    plt.title("Visit Trends by Day of the Week Across Place Categories", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Average Visit Proportion", fontsize=12)
    plt.legend(title="Place Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return category_trends

def plot_hourly_trends(mp):
    """
    Plots hourly visit trends across place categories with distinct colors.
    """
    hour_columns = [f'hour_{hour}' for hour in range(24)]
    hour_labels = [f'{hour}:00' for hour in range(24)]

    mp[hour_columns] = mp[hour_columns].div(mp[hour_columns].sum(axis=1), axis=0)

    category_hourly_trends = mp.groupby('place_category')[hour_columns].mean()

    # Define a distinct color palette
    unique_categories = category_hourly_trends.index
    colors = sns.color_palette("husl", len(unique_categories))  # "husl" gives visually distinct colors
    color_dict = dict(zip(unique_categories, colors))

    # Line plot for hourly trends
    plt.figure(figsize=(14, 7))
    for category in category_hourly_trends.index:
        plt.plot(hour_labels, category_hourly_trends.loc[category], marker='o', label=category, color=color_dict[category])

    plt.title("Visit Trends by Hour of the Day Across Place Categories", fontsize=14)
    plt.xlabel("Hour of the Day", fontsize=12)
    plt.ylabel("Average Visit Proportion", fontsize=12)
    plt.legend(title="Place Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    return category_hourly_trends

def plot_daily_trends_heatmap(mp):
    """
    Plots a heatmap for visit count proportions by day of the week across place categories.
    """
    day_columns = ['visit_count_monday', 'visit_count_tuesday', 'visit_count_wednesday',
                   'visit_count_thursday', 'visit_count_friday', 'visit_count_saturday', 'visit_count_sunday']

    mp[day_columns] = mp[day_columns].div(mp[day_columns].sum(axis=1), axis=0)
    category_trends = mp.groupby('place_category')[day_columns].mean()

    plt.figure(figsize=(12, 6))
    sns.heatmap(category_trends, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

    plt.title("Daily Visit Trends Heatmap by Place Category", fontsize=14)
    plt.xlabel("Day of the Week", fontsize=12)
    plt.ylabel("Place Category", fontsize=12)
    plt.xticks(ticks=range(7), labels=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45)
    plt.show()

def plot_hourly_trends_heatmap(mp):
    """
    Plots a heatmap for visit count proportions by hour of the day across place categories.
    """
    hour_columns = [f'hour_{hour}' for hour in range(24)]

    mp[hour_columns] = mp[hour_columns].div(mp[hour_columns].sum(axis=1), axis=0)
    category_hourly_trends = mp.groupby('place_category')[hour_columns].mean()

    plt.figure(figsize=(12, 6))
    sns.heatmap(category_hourly_trends, cmap="coolwarm", annot=False, linewidths=0.5)

    plt.title("Hourly Visit Trends Heatmap by Place Category", fontsize=14)
    plt.xlabel("Hour of the Day", fontsize=12)
    plt.ylabel("Place Category", fontsize=12)
    plt.xticks(ticks=range(24), labels=[f'{hour}:00' for hour in range(24)], rotation=45)
    plt.show()

def plot_sunburst(mp):
    """
    Creates a Sunburst chart showing the hierarchical relationship between place_category and CATEGORY_TAGS.
    """
    sunburst_data = mp.groupby(['place_category', 'CATEGORY_TAGS']).size().reset_index(name='count')

    fig = px.sunburst(sunburst_data,
                      path=['place_category', 'CATEGORY_TAGS'],  # Hierarchy levels
                      values='count',
                      color='place_category',
                      color_discrete_sequence=px.colors.qualitative.Set3)  # Ensures distinct colors

    fig.update_layout(title="Sunburst Chart: Place Category → Category Tags", width=800, height=800)

    fig.show()
