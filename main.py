import random
import time
import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(0)

def generate_data(start_coef, start_var, std_dev, num_points):
    dimensions = start_coef.shape[0]
    data = []
    for i in range(dimensions):
        points = np.random.standard_normal(num_points) * std_dev + start_coef[i] * start_var[i]
        data.append(points)
    return np.array(data).T  # Transpose to get points as rows

def k_means(data, k, max_iters=100):
    # Initialize centroids by randomly selecting k data points
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    prev_error = float('inf')

    for iteration in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2) ** 2
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else data[np.random.choice(data.shape[0])] for i in range(k)])

        # Calculate overall squared error
        error = np.sum(np.min(distances, axis=1))
        # Comment out the error printing inside the loop
        # print(f"Iteration {iteration + 1}, Squared Error: {error}")

        # Warning check if error increases
        if error > prev_error:
            print("Warning: Squared error increased!")
        prev_error = error

        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels, error

def compute_friend_matrix(labels):
    n = len(labels)
    friend_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                friend_matrix[i, j] = 1
    return friend_matrix

def generate_and_plot_data(start_points, start_var, std_dev, num_points):
    # Generate training data
    train_data = np.vstack([generate_data(start_coef, start_var, std_dev, num_points) for start_coef in start_points])
    # Generate testing data
    test_data = np.vstack([generate_data(start_coef, start_var, std_dev, num_points) for start_coef in start_points])

    # Perform k-means clustering with multiple initializations on training data
    k = 4
    max_iters = 100
    num_initializations = 100
    best_error = float('inf')
    best_centroids = None
    best_labels = None
    prev_test_error = float('inf')

        # Plot training data
    plt.figure()
    for i, start_coef in enumerate(start_points):
        cluster_data = train_data[i * num_points:(i + 1) * num_points]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, label=f'Train Cluster {i + 1}')
    plt.title(f"Initial Training Data (Std.Dev={std_dev})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Plot testing data
    plt.figure()
    for i, start_coef in enumerate(start_points):
        cluster_data = test_data[i * num_points:(i + 1) * num_points]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, label=f'Test Cluster {i + 1}')
    plt.title(f"Initial Testing Data (Std.Dev={std_dev})")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Ground truth labels for computing the error of the ground truth clusterings
    ground_truth_labels = np.concatenate([np.full(num_points, i) for i in range(k)])
    ground_truth_centroids = np.array([start_points[i].flatten() for i in range(k)])
    ground_truth_distances = np.linalg.norm(train_data[:, np.newaxis] - ground_truth_centroids, axis=2) ** 2
    ground_truth_error = np.sum(np.min(ground_truth_distances, axis=1))
    print(f"Ground truth error: {ground_truth_error}")

    for i in range(num_initializations):
        centroids, labels, train_error = k_means(train_data, k, max_iters)
        test_distances = np.linalg.norm(test_data[:, np.newaxis] - centroids, axis=2) ** 2
        test_error = np.sum(np.min(test_distances, axis=1))

        if test_error > prev_test_error:
            print("Warning: Test error increased!")
        prev_test_error = test_error

        if train_error < best_error:
            best_error = train_error
            best_centroids = centroids
            best_labels = labels

        print(f"Initialization {i + 1}, Train Error: {train_error}, Test Error: {test_error}")

    # Compute friend matrices and similarity measure
    FGT = compute_friend_matrix(ground_truth_labels)
    Ftrain = compute_friend_matrix(best_labels)
    Fintersect = FGT * Ftrain
    S = (np.sum(Fintersect) / np.sum(FGT) + np.sum(Fintersect) / np.sum(Ftrain)) / 2
    print(f"Cluster similarity measure S: {S}")



    # Plot clustered training data for the best initialization
    plt.figure()
    for i in range(k):
        cluster_data = train_data[best_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, label=f'Train Cluster {i + 1}')
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='red', marker='x', label='Centroids')
    plt.title(f'K-Means Clustering on Training Data (Best Initialization, std_dev={std_dev})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Assign clusters to testing data using the centroids from training data
    test_distances = np.linalg.norm(test_data[:, np.newaxis] - best_centroids, axis=2) ** 2
    test_labels = np.argmin(test_distances, axis=1)

    # Plot clustered testing data
    plt.figure()
    for i in range(k):
        cluster_data = test_data[test_labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], alpha=0.5, label=f'Test Cluster {i + 1}')
    plt.scatter(best_centroids[:, 0], best_centroids[:, 1], c='red', marker='x', label='Centroids')
    plt.title(f'K-Means Clustering on Testing Data (std_dev={std_dev})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    # Print the best error
    print(f"Best training error at convergence: {best_error}")

     # Calculate and print the ground truth error
    ground_truth_labels = np.concatenate([np.full(num_points, i) for i in range(k)])
    ground_truth_centroids = np.array([start_points[i].flatten() for i in range(k)])
    ground_truth_distances = np.linalg.norm(train_data[:, np.newaxis] - ground_truth_centroids, axis=2) ** 2
    ground_truth_error = np.sum(np.min(ground_truth_distances, axis=1))
    print(f"Ground truth error (Std. Dev = {std_dev}): {ground_truth_error}")


def main():
    # Generate data
    start_points = np.array([
        [[.3], [.3]],  # 2D starting coefficients
        [[.3], [.7]],  # 2D starting coefficients
        [[.7], [.3]],  # 2D starting coefficients
        [[.7], [.7]]   # 2D starting coefficients
    ])
    
    start_var = np.array([[1], [1]])   # 2D starting variables
    std_dev_1 = 0.1 # standard deviation for the first set of points
    std_dev_2 = 0.02 # standard deviation for the second set of points
    num_points = 50

    generate_and_plot_data(start_points, start_var, std_dev_1, num_points)

    generate_and_plot_data(start_points, start_var, std_dev_2, num_points)




if __name__ == "__main__":
    main()