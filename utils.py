import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    #raise NotImplementedError
    truePos = falPos = trueNeg = falNeg = 0

    for x in range(len(real_labels)):
        real_labels[x] = int(real_labels[x])
        predicted_labels[x] = int(predicted_labels[x])

    for x in range(len(real_labels)):
        if(real_labels[x] == predicted_labels[x]):
            if(real_labels[x] == 0):
                trueNeg +=1
            else:
                truePos +=1
        elif(real_labels[x] == 0 and predicted_labels == 1):
            falPos += 1
        else:
            falNeg += 1

    assert falNeg+falPos+trueNeg+truePos == len(predicted_labels)

    if(truePos==0):
        if(falPos == 0):
            p = 1
        if(falNeg == 0):
            r = 1

    if(truePos+falPos != 0):
        p = truePos/(truePos+falPos)
    if(truePos+falNeg != 0):
        r = truePos/(truePos+falNeg)

    if(p*r == 0):
        return 0

    return 2*(p*r)/(p+r)

class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        distance = 0
        for x in range(len(point1)):
            distance += pow(abs(point1[x] - point2[x]),3)
        return pow(distance,1/3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        distance = 0
        for x in range(len(point1)):
            distance += pow((point1[x] - point2[x]),2)
        return pow(distance,0.5)

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        #raise NotImplementedError
        return np.inner(point1,point2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        #raise NotImplementedError
        return 1 - (np.dot(point1,point2) / (np.sqrt(np.dot(point1,point1)) * np.sqrt(np.dot(point2,point2)) ) )

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        #raise NotImplementedError

        val = 0
        for x in range(len(point1)):
            val += pow(point1[x] - point2[x],2)

        return -1 * np.exp((-0.5)*val)

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        #raise NotImplementedError

        self.best_scaler = None
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        bestYet = 0

        for dist_func in distance_funcs.values():
            for k in range(1,30,2):
                myKnn = KNN(k,dist_func)
                myKnn.train(x_train,y_train)
                fScore = f1_score(y_val,myKnn.predict(x_val))
                if(fScore > bestYet):
                    bestYet = fScore
                    self.best_k = k
                    self.best_distance_function = dist_func
                    self.best_model = myKnn

        for dist_func_name, dist_func in distance_funcs.items():
            if(dist_func is self.best_distance_function):
                self.best_distance_function = dist_func_name
                break

        #print("bYwo: "+str(bestYet))

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        #raise NotImplementedError

        x_train_scaled = []
        x_val_scaled = []

        bestYet = 0
        for scaling_class in scaling_classes.values():
            my_scaling_class = scaling_class()
            x_train_scaled = my_scaling_class(x_train)
            x_val_scaled = my_scaling_class(x_val)
            for dist_func in distance_funcs.values():
                for k in range(1,30,2):
                    myKnn = KNN(k,dist_func)
                    myKnn.train(x_train_scaled,y_train)
                    fScore = f1_score(y_val,myKnn.predict(x_val_scaled))
                    if(fScore > bestYet):
                        bestYet = fScore
                        self.best_k = k
                        self.best_distance_function = dist_func
                        self.best_scaler = scaling_class
                        self.best_model = myKnn

        for dist_func_name, dist_func in distance_funcs.items():
            if(dist_func is self.best_distance_function):
                self.best_distance_function = dist_func_name
                break

        for scaler_name, scaling_class in scaling_classes.items():
            if(scaling_class is self.best_scaler):
                self.best_scaler = scaler_name
                break

class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        normFeatures = []

        for feature in features:
            normFeature = []
            sum = 0
            for f in feature:
                sum += f*f
            sum = np.sqrt(sum)

            if(sum == 0):
                for f in feature:
                    normFeature.append(0)
            else:
                for f in feature:
                    normFeature.append(f/sum)

            normFeatures.append(normFeature)
        return normFeatures

class MinMaxScaler:

    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.minMaxScalingFlag = False
        self.minFeatures = []
        self.maxFeatures = []
        pass

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        #raise NotImplementedError
        scaledFeatures = np.zeros((len(features),len(features[0])))
        if(self.minMaxScalingFlag is False):
            self.minMaxScalingFlag = True

            for i in range(len(features[0])):
                self.minFeatures.append(features[0][i])
                self.maxFeatures.append(features[0][i])

            for i in range(1,len(features)):
                for f in range(0,len(features[0])):
                    if(features[i][f] > self.maxFeatures[f]):
                        self.maxFeatures[f] = features[i][f]
                    if(features[i][f] < self.minFeatures[f]):
                        self.minFeatures[f] = features[i][f]


        for i in range(0,len(features)):
            for f in range(0,len(features[0])):
                if(self.maxFeatures[f] != self.minFeatures[f]):
                    scaledFeatures[i][f] = (features[i][f] - self.minFeatures[f]) / (self.maxFeatures[f] - self.minFeatures[f])
                else:
                    scaledFeatures[i][f] = 0

        return scaledFeatures