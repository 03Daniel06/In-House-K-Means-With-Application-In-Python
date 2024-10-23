# In-House-K-Means

In this project, I, Daniel Northcott, have created a K-Means algorithm that initially handled two-dimensional data (in `main.py`) and was later extended to work in n-dimensions (in `harder_data.py`). The purpose of this project is to generate normally distributed data around predefined centroids and use the K-Means algorithm to cluster this data, minimizing the squared error as the algorithm iterates.

## Project Overview

The goal of the project is to synthesize data points in a Euclidean space around given centroids. The generated points are normally distributed, and the K-Means algorithm attempts to iteratively adjust centroid positions to minimize the mean squared distance between points and their assigned centroids.

### First Iteration (`main.py`)

The first version of the project is a simple command-line program that generates data points in 2D space. The program creates points with a fixed standard deviation from randomly initialized centroids, and the K-Means algorithm is applied to find the "true" centroid locations. The algorithm outputs the squared error for each iteration, which we aim to minimize. All parameters, such as the number of points and the standard deviation, are tuneable within the code.

### Second Iteration (`harder_data.py`)

The second version expands on the first by allowing data to exist in n-dimensions. The core K-Means algorithm is the same, but it now supports higher-dimensional datasets, enabling the user to explore more complex clustering problems.

### GUI Version (`k-means-application.py`)

The latest addition to the project is a GUI-based application that builds on the n-dimensional functionality. This application allows users to interactively specify the number of clusters, dimensions, points, and standard deviation for the data. Users can visualize the clustering process for 2D and 3D datasets and see statistics such as runtime and errors directly in the interface.

To run the GUI-based version, please execute the `k-means-application.py` file.

## Running the Project

1. **Simple 2D K-Means:**
   - Run `main.py` for a basic command-line implementation that handles 2D data.

2. **K-Means in N-Dimensions:**
   - Run `harder_data.py` for a more advanced version that operates on data in n-dimensions.

3. **GUI Version:**
   - Run `k-means-application.py` for an interactive graphical interface where you can configure clustering parameters and view the results visually.

### Instructions for Running the GUI:

- Ensure you have the necessary dependencies installed, particularly `tkinter`, `matplotlib`, and `numpy`.
- Run the script with `python k-means-application.py`.
- Input the desired parameters such as the number of clusters, dimensions, and points.
- Click "Train and Test Model" to perform K-Means clustering and visualize the results.

Happy clustering!

-Daniel Northcott
