import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib.figure import Figure

np.random.seed(0)

def generate_data(start_coef, std_dev, num_points):
    dimensions = start_coef.shape[0]
    data = []
    for i in range(dimensions):
        points = np.random.standard_normal(num_points) * std_dev + start_coef[i]
        data.append(points)
    return np.array(data).T  # Transpose to get points as rows

def k_means(data, k, max_iters=100):
    start_time = time.time()  # Start time measurement

    # Initialize centroids by randomly selecting k data points
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]
    prev_error = float('inf')
    p = 2  # data.shape[1]

    errors = []  # List to store errors at each iteration

    for iteration in range(max_iters):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2, ord=p) ** p
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else data[np.random.choice(data.shape[0])] for i in range(k)])

        # Calculate overall squared error
        error = np.sum(np.min(distances, axis=1))
        errors.append(error)  # Store the error

        # Warning check if error increases
        if error > prev_error:
            print("Warning: Squared error increased!")
        prev_error = error

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    end_time = time.time()  # End time measurement
    runtime = end_time - start_time
    # print(f"K-Means runtime: {runtime:.4f} seconds")

    return centroids, labels, error, runtime

def compute_friend_matrix(labels):
    n = len(labels)
    friend_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                friend_matrix[i, j] = 1
    return friend_matrix

def generate_and_plot_data(start_points, std_dev, num_points, plot_frame, num_initializations=100):
    # Generate training data
    train_data = np.vstack([generate_data(start_coef, std_dev, num_points) for start_coef in start_points])
    # Generate testing data
    test_data = np.vstack([generate_data(start_coef, std_dev, num_points) for start_coef in start_points])
    # Perform k-means clustering with multiple initializations on training data
    k = start_points.shape[0]
    max_iters = 100
    best_error = float('inf')
    best_centroids = None
    best_labels = None
    prev_test_error = float('inf')
    total_runtime = 0  # Variable to accumulate total runtime
    errors = []  # List to store errors for each initialization

    for i in range(num_initializations):
        centroids, labels, error, runtime = k_means(train_data, k, max_iters)
        total_runtime += runtime

        # Calculate error on test data
        test_distances = np.linalg.norm(test_data[:, np.newaxis] - centroids, axis=2) ** 2
        test_error = np.sum(np.min(test_distances, axis=1))
        errors.append(test_error)

        if test_error < best_error:
            best_error = test_error
            best_centroids = centroids
            best_labels = labels

    # Plot the error versus the number of initializations
    fig, ax = plt.subplots()
    ax.plot(range(1, num_initializations + 1), errors, marker='o')
    ax.set_title('Testing Error vs Number of Initializations')
    ax.set_xlabel('Number of Initializations')
    ax.set_ylabel('Testing Error')
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Print the best error
    print(f"Best testing error at convergence: {best_error:.4f}")
    # Calculate and print the ground truth error
    ground_truth_labels = np.concatenate([np.full(num_points, i) for i in range(k)])
    ground_truth_centroids = np.array([start_points[i].flatten() for i in range(k)])
    ground_truth_distances = np.linalg.norm(test_data[:, np.newaxis] - ground_truth_centroids, axis=2) ** 2
    ground_truth_error = np.sum(np.min(ground_truth_distances, axis=1))
    print(f"Ground truth error (Std. Dev = {std_dev}): {ground_truth_error:.4f}")
    # Print the best error found by k-means
    print(f"Best error found by k-means: {best_error:.4f}")
    # Print runtime statistics
    average_runtime = total_runtime / num_initializations
    print(f"Total runtime for k-means over {num_initializations} initializations: {total_runtime:.4f} seconds")
    print(f"Average runtime per initialization: {average_runtime:.4f} seconds")

class KMeansGUI:
    def __init__(self, master):
        self.master = master
        master.title("K-Means Clustering GUI")

        # Create input fields
        self.label_num_clusters = tk.Label(master, text="Number of Clusters:")
        self.entry_num_clusters = tk.Entry(master)
        self.entry_num_clusters.insert(0, "3")

        self.label_num_points = tk.Label(master, text="Number of Points per Cluster:")
        self.entry_num_points = tk.Entry(master)
        self.entry_num_points.insert(0, "50")

        self.label_num_dimensions = tk.Label(master, text="Number of Dimensions:")
        self.entry_num_dimensions = tk.Entry(master)
        self.entry_num_dimensions.insert(0, "2")

        self.label_std_dev = tk.Label(master, text="Standard Deviation:")
        self.entry_std_dev = tk.Entry(master)
        self.entry_std_dev.insert(0, "0.1")

        # Arrange input fields in grid
        self.label_num_clusters.grid(row=0, column=0, sticky=tk.W)
        self.entry_num_clusters.grid(row=0, column=1)

        self.label_num_points.grid(row=1, column=0, sticky=tk.W)
        self.entry_num_points.grid(row=1, column=1)

        self.label_num_dimensions.grid(row=2, column=0, sticky=tk.W)
        self.entry_num_dimensions.grid(row=2, column=1)

        self.label_std_dev.grid(row=3, column=0, sticky=tk.W)
        self.entry_std_dev.grid(row=3, column=1)

        # Create Train and Test button
        self.button_train = tk.Button(master, text="Train and Test Model", command=self.train_and_test)
        self.button_train.grid(row=4, column=0, columnspan=2, pady=10)

        # Create a Text widget to display statistics
        self.text_output = tk.Text(master, height=15, width=70)
        self.text_output.grid(row=5, column=0, columnspan=2)

        # Create a frame to hold the plot
        self.plot_frame = tk.Frame(master)
        self.plot_frame.grid(row=6, column=0, columnspan=2)

    def train_and_test(self):
        # Get parameters from input fields
        try:
            num_clusters = int(self.entry_num_clusters.get())
            num_points = int(self.entry_num_points.get())
            num_dimensions = int(self.entry_num_dimensions.get())
            std_dev = float(self.entry_std_dev.get())
            if num_clusters <= 0 or num_points <= 0 or num_dimensions <= 0 or std_dev <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid positive numerical values.")
            return

        # Call the generate_and_plot_data function with parameters
        # Redirect prints to the text_output widget
        self.text_output.delete('1.0', tk.END)

        # We need to capture the outputs from generate_and_plot_data
        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()

        # Run the function
        try:
            self.run_generate_and_plot_data(num_clusters, num_dimensions, num_points, std_dev)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            sys.stdout = old_stdout
            return

        # Get the output and display it in text_output
        sys.stdout = old_stdout
        output = mystdout.getvalue()
        self.text_output.insert(tk.END, output)

    def run_generate_and_plot_data(self, num_clusters, num_dimensions, num_points, std_dev):
        # Randomly generate cluster centers
        start_points = np.random.rand(num_clusters, num_dimensions)

        generate_and_plot_data(start_points, std_dev, num_points, self.plot_frame, num_initializations=10)

if __name__ == "__main__":
    root = tk.Tk()
    gui = KMeansGUI(root)
    root.mainloop()
