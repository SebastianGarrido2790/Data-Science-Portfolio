import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_settings():
    """
    Configure global plot settings for customer analysis visualizations.
    """
    # Use seaborn's theme for a clean look with white grid backgrounds
    sns.set_theme(context="talk", style="whitegrid", palette="viridis")

    # Set default figure size for clear and detailed plots
    plt.rcParams["figure.figsize"] = (12, 8)

    # Customize title and label sizes for axes
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14

    # Set tick label sizes for clarity
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # Configure legend properties
    plt.rcParams["legend.fontsize"] = 12

    # Enhance line plots with thicker lines
    plt.rcParams["lines.linewidth"] = 2

    # Customize grid appearance
    plt.rcParams["grid.color"] = "gray"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
