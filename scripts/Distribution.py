import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def analyze_dataset(directory):
    """
    Analyze the dataset and return statistics by plant type.
    Counts only image files (png, jpg, jpeg).
    """
    plant_types = {}
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            try:
                plant_type, disease = subdir.split('_', 1)
                if plant_type not in plant_types:
                    plant_types[plant_type] = {}
                num_images = len([
                    f for f in os.listdir(path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
                plant_types[plant_type][disease] = num_images
            except ValueError:
                print(f"Warning: Ignoring directory '{subdir}' "
                      "(doesn't follow plant_disease format)")
    return plant_types


def plot_distribution(plant_type, disease_counts):
    """
    Plot bar chart and pie chart for disease distribution.
    """
    labels, values = zip(*disease_counts.items())
    total_images = sum(values)
    proportions = [value / total_images * 100 for value in values]

    plt.figure(figsize=(14, 6))

    # Bar chart
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, proportions, color=[
        '#3498db', '#e74c3c', '#2ecc71',
        '#f39c12', '#9b59b6', '#1abc9c'
    ])
    plt.xlabel('Disease Category', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage of images', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of diseases for {plant_type}',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, prop in zip(bars, proportions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{prop:.1f}%',
                 ha='center', va='bottom', fontsize=9)

    # Pie chart
    plt.subplot(1, 2, 2)
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
              '#9b59b6', '#1abc9c']
    plt.pie(proportions, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title(f'Proportion of diseases for {plant_type}',
              fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()


def main(directory):
    plant_types = analyze_dataset(directory)
    for plant_type, disease_counts in plant_types.items():
        plot_distribution(plant_type, disease_counts)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python distribution.py <directory>")
    else:
        print(f"Analyzing dataset in {sys.argv[1]}")
        main(sys.argv[1])
