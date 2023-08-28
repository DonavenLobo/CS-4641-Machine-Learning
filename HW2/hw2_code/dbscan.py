import numpy as np
from kmeans import pairwise_dist
class DBSCAN(object):
    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset
    def fit(self):
        """Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        Hint: Using sets for visitedIndices may be helpful here.
        Iterate through the dataset sequentially and keep track of your points' cluster assignments.
        If a point is unvisited or is a noise point (has fewer than the minimum number of neighbor points), then its cluster assignment should be -1.
        Set the first cluster as C = 0
        """
        
        cluster_idx = np.ones(self.dataset.shape[0], dtype = 'int') * -1
        visitedIndices = set()
        C = 0
        for i in range(self.dataset.shape[0]):
            # print(C)
            # print(self.dataset.shape[0])
            if i in visitedIndices:
                continue
            visitedIndices.add(i)
            # print('1')
            # print(visitedIndices)
            neighborIndices = self.regionQuery(i)
            if len(neighborIndices) >= self.minPts:
                # if i == 0:
                #     C = 0
                # else:
                #     C = C + 1
                self.expandCluster(i, neighborIndices, C, cluster_idx, visitedIndices)
                C = C + 1
                # print('2')
                # print(visitedIndices)
                # print(cluster_idx)

        return cluster_idx  


        

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
           HINT: regionQuery could be used in your implementation
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (N, ) int numpy array, indices of all points witin P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        Hints: 
            np.concatenate(), np.unique(), np.sort(), and np.take() may be helpful here
            A while loop may be better than a for loop
        """
        cluster_idx[index] = C
        # print('In Expand:')
        # print(C)
        i = 0
        while i < len(neighborIndices):
            neighborIndex = neighborIndices[i]
            if neighborIndex not in visitedIndices:
                visitedIndices.add(neighborIndex)
                p_neighbors = self.regionQuery(neighborIndex)
                if len(p_neighbors) >= self.minPts:
                    neighborIndices = np.concatenate((neighborIndices, p_neighbors), axis = 0) # Worked on this in office hours. Connect both arrays about x axis (flatten)
                    uniqueInd = np.unique(neighborIndices, return_index = True)[1] # Worked on this in office hours. Find all the indicies of unique values
                    neighborIndices = np.take(neighborIndices,np.sort(uniqueInd)) # Worked on this in office hours.
    
            if cluster_idx[neighborIndex] < 0:
                cluster_idx[neighborIndex] = C
            i += 1



    def regionQuery(self, pointIndex):
        """Returns all points within P's eps-neighborhood (including P)

        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        Hint: pairwise_dist (implemented above) and np.argwhere may be helpful here
        """
        p = self.dataset[pointIndex].reshape(1,-1) # Assign point to variable
        # p = p[np.newaxis, :] # Change shape to allow for pairwise_dist to function correctly
        dist = pairwise_dist(self.dataset, p) # Find all distance from points to given point
        ind = np.argwhere(dist.flatten() <= self.eps).flatten() # Return the indicies of points in the region
        return ind

        