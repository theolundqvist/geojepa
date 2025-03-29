import sys
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame


def read_dates(file_path):
    """
    Reads dates from a file, one per line.

    Args:
        file_path (str): Path to the input file.

    Returns:
        list: List of date strings.
    """
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            # Strip whitespace and ignore empty lines
            dates = [line.strip() for line in lines if line.strip()]
            return dates
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)


def parse_dates(dates):
    """
    Converts a list of date strings to pandas datetime objects.

    Args:
        dates (list): List of date strings.

    Returns:
        pandas.DatetimeIndex: Datetime objects.
    """
    try:
        date_objects = pd.to_datetime(dates, format="%Y-%m-%d")
        return date_objects
    except ValueError as ve:
        print(f"Date parsing error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while parsing dates: {e}")
        sys.exit(1)


def aggregate_by_month(date_objects):
    """
    Aggregates the number of articles published per month.

    Args:
        date_objects (pandas.DatetimeIndex): Datetime objects.

    Returns:
        pandas.Series: Counts of articles per month.
    """
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({"date": date_objects})

    # Set 'date' as the DataFrame index
    df.set_index("date", inplace=True)

    # Resample the data by month and count the number of articles
    monthly_counts = df.resample("M").size()

    # Adjust the index to set the day to 27 for labeling purposes
    monthly_counts.index = monthly_counts.index.map(lambda x: x.replace(day=27))

    return monthly_counts


def plot_img(monthly_counts):
    """
    Plots the monthly article counts and their moving average.

    Args:
        monthly_counts (pandas.Series): Counts of articles per month.
        moving_avg (pandas.Series): Moving average of article counts.
        window_size (int): The window size used for the moving average.
    """
    plt.figure(figsize=(14, 7))

    # Plot monthly counts as bars
    monthly_counts.plot(
        kind="bar",
        color="skyblue",
        edgecolor="black",
        label="Monthly Counts",
        alpha=0.6,
    )

    # Improve spacing and rotation
    plt.xlabel("Month")
    plt.ylabel("Number of Articles")
    plt.title("JEPA Articles Per Month")
    plt.legend()

    # Adjust x-ticks to match the new date format
    plt.xticks(
        ticks=range(len(monthly_counts)),
        labels=[dt.strftime("%b %Y") for dt in monthly_counts.index],
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    # Display the plot
    plt.show()

    # Optionally, save the plot as an image
    # plt.savefig('articles_with_moving_average.png')


def plot_img_cum(monthly_counts: DataFrame):
    cumulative_counts: DataFrame = monthly_counts.cumsum()

    plt.figure(figsize=(14, 7))

    # Plot the cumulative counts as bars
    ax = cumulative_counts.plot(
        kind="bar",
        color="skyblue",
        edgecolor="black",
        label="Cumulative Count",
        alpha=0.6,
    )

    # Add labels where the cumulative count increases
    prev_value = 0  # Initialize the previous value
    labels = []
    for i, (index, value) in enumerate(cumulative_counts.items()):
        if value > prev_value:  # Check if the value increased
            labels.append(index.strftime("%b %Y"))  # Label every other month
        else:
            labels.append("")
        prev_value = value

    # Add labels, title, and legend
    plt.xlabel("Month")
    plt.ylabel("Cumulative Number of Articles")
    plt.title("Cumulative Count of JEPA Articles Over Time")
    plt.legend()

    # Improve x-ticks for readability
    plt.xticks(
        ticks=range(len(cumulative_counts)),
        labels=labels,
        rotation=45,
        ha="right",
        fontsize=14,
    )

    plt.tight_layout()

    # Display the plot
    plt.show()

    # Optionally, save the plot as an image
    # plt.savefig('cumulative_articles_plot.png')


def main():
    """
    Main function to execute the script.
    """
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python article_moving_average.py <input_file> [window_size]")
        print("Example: python article_moving_average.py dates.txt 3")
        sys.exit(1)

    input_file = sys.argv[1]

    # Optional: window size for moving average
    if len(sys.argv) == 3:
        try:
            window_size = int(sys.argv[2])
            if window_size < 1:
                raise ValueError
        except ValueError:
            print("Error: window_size must be a positive integer.")
            sys.exit(1)
    else:
        window_size = 3  # Default window size

    # Step 1: Read dates from the input file
    dates = read_dates(input_file)

    if not dates:
        print("The input file is empty or contains no valid dates.")
        sys.exit(1)

    # Step 2: Parse the date strings into datetime objects
    date_objects = parse_dates(dates)

    # Step 3: Aggregate the number of articles per month
    monthly_counts = aggregate_by_month(date_objects)

    if monthly_counts.empty:
        print("No valid dates found to plot.")
        sys.exit(1)

    # Step 4: Calculate the moving average

    # Step 5: Plot the data with moving average
    plot_img_cum(monthly_counts)


if __name__ == "__main__":
    main()
