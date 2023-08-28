
'''
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
'''

import numpy as np


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-5):  # No need to implement
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria w.r.t relative change of loss
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):# [2 pts]
        """
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        
        num_points = self.points.shape[0] # Number of points
        num_clusters = self.K # Number of clusters
        ind = np.random.choice(num_points, size = num_clusters, replace = False) # Choose random starting point for each cluster
        centers = self.points[ind] # Use these indices to select the initial centers
        return centers

    def kmpp_init(self):# [3 pts]
        """
            Use the intuition that points further away from each other will probably be better initial centers
        Return:
            self.centers : K x D numpy array, the centers.
        """
        
        num_points = self.points.shape[0] # Number of points
        n_samples = int(0.01 * num_points) # Number of samples to take from dataset (1%)
        sample_indices = np.random.choice(num_points, n_samples, replace=False) # Choose n samples indicies from total number of points
        sample_points = self.points[sample_indices, :] # Get points out from dataset
        center_indices = [np.random.choice(n_samples)] # Select only one random point to be the first cluster center.

        # Repeat until all k-centers have been assigned.
        for _ in range(1, self.K):

            # For each point in the sampled dataset, find the nearest currently established cluster center
            dists = pairwise_dist(sample_points, self.points[center_indices, :])
            min_dists = np.min(dists, axis=1)

            # Examine all the squared distances and take the point with the maximum squared distance as a new cluster center
            max_dist_index = np.argmax(min_dists)
            center_indices.append(max_dist_index)

            centers = sample_points[center_indices, :] # Use the indices of the selected points to get the centers

        return centers

    def update_assignment(self):  # [5 pts]
        """
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: You could call pairwise_dist() function
        """
        dists = pairwise_dist(self.points, self.centers)  # compute pairwise distance from each point to each of the centers
        self.assignments = np.argmin(dists, axis=1)  # assign each point to closest cluster center
        return self.assignments        

    def update_centers(self):  # [5 pts]
        """
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.

        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        """
        for cluster in range(self.K):
            # get the indices of the points assigned to a cluster
            cluster_indices = np.where(self.assignments == cluster)[0]
            if len(cluster_indices) > 0:
                # compute the mean of the points assigned to cluster k
                self.centers[cluster] = np.mean(self.points[cluster_indices], axis=0)
        return self.centers

    def get_loss(self):  # [5 pts]
        """
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """

        dist = np.linalg.norm(self.points - self.centers[self.assignments]) #Find the euclidean distance between each point and its respective center
        sq_dist = dist ** 2 # Square distances
        self.loss = np.sum(sq_dist) # Calculate total loss
        return self.loss

    def train(self):    # [10 pts]
        """
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster, 
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned, 
                     pick a random point in the dataset to be the new center and 
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference 
                     in loss compared to the previous iteration is less than the given 
                     relative tolerance threshold (self.rel_tol). 
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
                
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        """

        for i in range(self.max_iters):
           
            self.update_assignment() # Update cluster assignment for each point

            # Update cluster centers based on new assignments
            for cluster in range(self.K):
                cluster_pts = self.points[self.assignments == cluster] # Get all points in each cluster
                if cluster_pts.shape[0] == 0:
                    # if there are no points in cluster, choose random point from dataset
                    self.centers[cluster] = self.points[np.random.choice(self.points.shape[0], size=1), :]
                else:
                    self.centers[cluster] = np.mean(cluster_pts, axis=0)

            # Check for empty clusters and reassign points
            empty_clusters = np.where(np.isnan(self.centers).any(axis=1))[0]
            if len(empty_clusters) > 0:
                for cluster in empty_clusters:
                    new_center = self.points[np.random.choice(self.points.shape[0], size=1), :]
                    self.centers[cluster] = new_center
                    distances = self.pairwise_dist(self.points, self.centers)
                    self.assignments = np.argmin(distances, axis=1)

            # Calculate loss and check for convergence
            if i > 0:
                prev_loss = self.loss # Save previous loss value
                self.loss = self.get_loss() # Assign new loss value
                rel_change = abs(self.loss - prev_loss) / prev_loss # Calculate the relative change is loss
                if rel_change < self.rel_tol: # Check for convergence
                    break
            else: # Only excecute on the first iteration to avoid error
                self.loss = self.get_loss() # Assign loss value

        return self.centers, self.assignments, self.loss


def pairwise_dist(x, y):  # [5 pts]
        np.random.seed(1)

        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
                dist: N x M array, where dist2[i, j] is the euclidean distance between
                x[i, :] and y[j, :]
        """
        # (X-Y)^2 = X^2 + Y^2 - 2XY

        x_2 = np.sum(np.square(x), axis=1) # Sum(X^2)
        y_2 = np.sum(np.square(y), axis=1) # Sum(Y^2)
        xy = np.dot(x, y.T) # Need to transpose y for dimensions to match

        x_2 = x_2[:, np.newaxis] # Change shape of x_2 from (N,) to (N,1), so that we call allow for broadcasting

        # Compute the Euclidean distance matrix
        dist = np.sqrt(x_2 + y_2 - (2 * xy))

        return dist

        # return np.sqrt(np.sum(np.square(x[:, np.newaxis, :] - y), axis=2))
