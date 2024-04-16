import matplotlib.pyplot as plt


def plot_pie_chart(data_dict, save_path=None):
    # Count the occurrences of each value in the dictionary
    value_counts = {}
    for value in data_dict.values():
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    # Prepare data for the pie chart
    values = list(value_counts.keys())
    counts = list(value_counts.values())

    # Sort the data in descending order
    sorted_data = sorted(zip(values, counts), key=lambda x: x[0], reverse=True)
    values, counts = zip(*sorted_data)

    # Create a pie chart
    plt.figure(figsize=(12, 10))
    plt.pie(counts, labels=values, autopct='%1.1f%%', startangle=90)
    plt.title("Pie Chart of bin info")

    # Add a legend
    legend_labels = [f"{value}: {count}" for value, count in zip(values, counts)]
    plt.legend(legend_labels, title="Value: Count", loc="upper right")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        pass

    # Show the pie chart (if save_path is not provided)
    # if not save_path:
    #     plt.show()
