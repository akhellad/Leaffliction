import os
import matplotlib.pyplot as plt


def analyze_dataset(directory):
    plant_types = {}
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if os.path.isdir(path):
            plant_type, disease = subdir.split('_', 1)
            if plant_type not in plant_types:
                plant_types[plant_type] = {}
            num_images = len(os.listdir(path))
            plant_types[plant_type][disease] = num_images
    return plant_types


def plot_distribution(plant_type, disease_counts):
    labels, values = zip(*disease_counts.items())
    total_images = sum(values)
    proportions = [value / total_images * 100 for value in values]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(labels, proportions)
    plt.xlabel('Disease Category')
    plt.ylabel('Percentage of images')
    plt.title(f'Distribution of diseases for {plant_type}')

    plt.subplot(1, 2, 2)
    plt.pie(proportions, labels=labels, autopct='%1.1f%%')
    plt.title(f'Proportion of diseases for {plant_type}')

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
