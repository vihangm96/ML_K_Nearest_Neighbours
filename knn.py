import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k, distance_function):
        """
        :param k: int
        :param distance_function
        """
        self.k = k
        self.distance_function = distance_function
        self.features = []
        self.labels = []

    # TODO: save features and lable to self
    def train(self, features, labels):
        """
        In this function, features is simply training data which is a 2D list with float values.
        For example, if the data looks like the following: Student 1 with features age 25, grade 3.8 and labeled as 0,
        Student 2 with features age 22, grade 3.0 and labeled as 1, then the feature data would be
        [ [25.0, 3.8], [22.0,3.0] ] and the corresponding label would be [0,1]

        For KNN, the training process is just loading of training data. Thus, all you need to do in this function
        is create some local variable in KNN class to store this data so you can use the data in later process.
        :param features: List[List[float]]
        :param labels: List[int]
        """
        #raise NotImplementedError
        self.features = features
        self.labels = labels

    # TODO: predict labels of a list of points
    def predict(self, features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        #raise NotImplementedError
        predicted = []
        for f in features:
            votes = {}
            votes[0] = votes[1] = 0
            for n in self.get_k_neighbors(f):
                votes[n] += 1

            c=0
            if(votes[1]>votes[0]):
                c = 1
            predicted.append(c)

        return predicted

    # TODO: find KNN of one point
    def get_k_neighbors(self, point):
        """
        This function takes one single data point and finds k-nearest neighbours in the training set.
        You already have your k value, distance function and you just stored all training data in KNN class with the
        train function. This function needs to return a list of labels of all k neighours.
        :param point: List[float]
        :return:  List[int]
        """
        #raise NotImplementedError
        distances = []

        for x in range(len(self.features)):
            distances.append((self.distance_function(point,self.features[x]),int(self.labels[x])))

        distances = (sorted(distances,key= lambda x :float(x[0])))[:self.k]

        kNeighbors = []

        limit = 0
        if(self.k>len(self.features)):
            limit = len(self.features)
        limit = self.k

        for k in range(0,limit):
            kNeighbors.append(int(distances[k][1]))

        return kNeighbors

if __name__ == '__main__':
    print(np.__version__)
