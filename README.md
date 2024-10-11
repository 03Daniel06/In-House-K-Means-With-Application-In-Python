# In-House-K-Means
In this project, I created a k-means algorithm at first in two dimensions inside the main.py and in n-dimensions in the harder_data.py

This project aims at synthesizing data in space that are noramlly distributed from a 'centroid.' The data is sythensized first around four points and fifty generated points from the centroid. The algorithm then selects a random point in the space and the avearge euclidean distance to this point is then calculated. Each iteration, a better 'guess' of where the true centroid is made and the mean square distance is then calcualted again. This is out objective function that we are trying to minimize. 

My first iteration of the project was very simple and displayed command line outputs, this is inside the 'main.py' file. In this program I instantiated points in Euclidean space with a set standard deviation. All of the parameteters are set within the 'main' function of the program and are tuneable there. 

The second instantiation (harder_data.py) takes the problem to n-dimensions and can take vectors of any size, and the process is repeated. The 'k-means-application.py' is an GUI based version of the harder data problem.

-Daniel Northcott
