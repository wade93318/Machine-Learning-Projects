import numpy as np
from scipy.special import expit
import math


class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()


class Useless(Model):

    def __init__(self):
        super().__init__()
        self.reference_example = None
        self.reference_label = None

    def fit(self, X, y):
        self.num_input_features = X.shape[1]
        # Designate the first training example as the 'reference' example
        # It's shape is [1, num_features]
        self.reference_example = X[0, :]
        # Designate the first training label as the 'reference' label
        self.reference_label = y[0]
        self.opposite_label = 1 - self.reference_label

    def predict(self, X):
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        # Compute the dot products between the reference example and X examples
        # The element-wise multiply relies on broadcasting; here, it's as if we first
        # replicate the reference example over rows to form a [num_examples, num_input_features]
        # array, but it's done more efficiently. This forms a [num_examples, num_input_features]
        # sparse matrix, which we then sum over axis 1.
        dot_products = X.multiply(self.reference_example).sum(axis=1)
        # dot_products is now a [num_examples, 1] dense matrix. We'll turn it into a
        # 1-D array with shape [num_examples], to be consistent with our desired predictions.
        dot_products = np.asarray(dot_products).flatten()
        # If positive, return the same label; otherwise return the opposite label.
        same_label_mask = dot_products >= 0
        opposite_label_mask = ~same_label_mask
        y_hat = np.empty([num_examples], dtype=np.int)
        y_hat[same_label_mask] = self.reference_label
        y_hat[opposite_label_mask] = self.opposite_label
        return y_hat


class LogisticRegression(Model):

    def __init__(self,learning_rate,iterations,num_features_select):
        super().__init__()
        # TODO: Initializations etc. go here.

        # initialize the class attributes that we want to use.
        self.weights = None
        self.learning_rate = learning_rate
        self.iterations = iterations
        # Get the total number of features that we want to choose.
        self.num_features_select = num_features_select

    def feature_selection(self, X, y):

        num_examples, num_features = X.shape
        # Get the number of data examples and number of column of features.
        threshold = []
        # We use a list to store all threshold value of all features in one dataset.
        res_dict = {}

        X = X.toarray()

        for num in range(num_features):
            data_fea = X[:,num]
            # get all data example at specific number feature.

            for i in range(data_fea.shape[0]):
                # We judge whether the value of specific feature is continuous or categorical.

                if (data_fea[i] != 1.0) and (data_fea[i] != 0.0):
                    # If we judge the value of this feature is continuous, we will compute the threshold value of this feature.
                    thre_val = np.mean(data_fea,axis=0)
                    threshold.append(thre_val)
                    break
                else:
                    # If the value of feature is categorical, we just use 0.5 as its threshold value to help us separate "0" and "1".
                    if i == data_fea.shape[0] - 1:
                        threshold.append(0.5)

        for num in range(num_features):
            # We create a loop to do iteration on number of features.
            Pcont = 0
            Ncont = 0
            PPlabel = 0
            PNlabel = 0
            NPlabel = 0
            NNlabel = 0
            # create respective variables that we will use to help us calculate different numbers of occurrences at different situation in the loop.
            data = X[:,num]
            for cont in range(len(data)):
                if data[cont] >= threshold[num]:
                    Pcont = Pcont + 1
                    # Calculate the total number of data examples whose value of feature are over threshold.
                    if y[cont] == 1:
                        PPlabel = PPlabel + 1
                        # Calculate the total number of data examples whose y label == 1.
                    else:
                        PNlabel = PNlabel + 1
                        # Calculate the total number of data examples whose y label == 0.
                else:
                    Ncont = Ncont + 1
                    # Calculate the total number of data examples whose value of feature are below threshold.
                    if y[cont] == 1:
                        NPlabel = NPlabel + 1
                        # Calculate the total number of data examples whose y label == 1.
                    else:
                        NNlabel = NNlabel + 1
                        # Calculate the total number of data examples whose y label == 0.

            H_Condition_1 = 0
            H_Condition_2 = 0

            # Define two variables to store the temporary values of conditional entropy.
            # At this point, we just need to calculate the conditional entropy of every feature and we can find the first 10 features which have highest 
            # information gains by finding the 10 features which have lowest value of conditional entropy.


            # Now, we compute the conditional entropy step by step.
            if PPlabel == 0:
                if PNlabel == 0:
                    H_Condition_1 = 0
                else:
                    H_Condition_1 = (- float(PNlabel)/Pcont * math.log(float(PNlabel)/Pcont,2)) * (float(Pcont) / num_examples)
            elif PNlabel == 0:
                H_Condition_1 = (-(float(PPlabel)/Pcont) * math.log(float(PPlabel)/Pcont,2)) * (float(Pcont) / num_examples)
            else:
                H_Condition_1 = (- float(PNlabel)/Pcont * math.log(float(PNlabel)/Pcont,2)) * (float(Pcont) / num_examples) + (-(float(PPlabel)/Pcont) * math.log(float(PPlabel)/Pcont,2)) * (float(Pcont) / num_examples)


            if NPlabel == 0:
                if NNlabel == 0:
                    H_Condition_2 = 0
                else:
                    H_Condition_2 = (-(float(NNlabel)/Ncont) * math.log(float(NNlabel)/Ncont,2)) * (float(Ncont) / num_examples)
            elif NNlabel == 0:
                H_Condition_2 = (- (float(NPlabel)/Ncont) * math.log(float(NPlabel)/Ncont,2)) * (float(Ncont) / num_examples)
            else:
                H_Condition_2 = (-(float(NNlabel)/Ncont) * math.log(float(NNlabel)/Ncont,2)) * (float(Ncont) / num_examples) + (- (float(NPlabel)/Ncont) * math.log(float(NPlabel)/Ncont,2)) * (float(Ncont) / num_examples)

            H_num_feature = H_Condition_1 + H_Condition_2
            # Finally we get the value of conditional entropy of every feature.


            if num not in res_dict:
                res_dict[num] = H_num_feature
                # We store the respective value of conditional entropy in a dictionary whose key is "number sort of features."

        res_fea = []
        sort_dict=sorted(res_dict.items(),key=lambda e:e[1])
        # We sorted this dict according to the value of conditional entropy of every feature order by ascent. 
        for t in range(self.num_features_select):
            res_fea.append(sort_dict[t][0])

        # Get the first number of "num_features_select" features and store them into a list. Then return this final list.
        return res_fea

    def fit(self, X, y):
        # TODO: Write code to fit the model.

        self.num_input_features = X.shape[1]
        num_examples, num_features = X.shape
        self.weights = np.zeros((1,num_features))

        if self.num_features_select != -1 and self.num_features_select < num_features:
            chosen_fea = self.feature_selection(X,y)
            # Use the "feature_selection" to help use to find chosen features.

            X = X.toarray()
            for cont in range(self.iterations):
                dwj = np.zeros([num_features],dtype=np.float)
                for num in range(num_examples):
                    data = X[num,:]
                    res_product = np.dot(self.weights,data)
                    PLogisticfunc = expit(res_product)
                    NLogisticfunc = expit(-res_product)
                    # Calcualte the logistic function.

                    for fea in range(num_features):
                        if fea not in chosen_fea:
                            dwj[fea] = 0
                            # If this feature is not in "chosen feature" list, we will set the respective position's wj = 0.
                        else:
                            # Use the normal gradient descent method to update the derivate of weights at each feature direction each time.
                            gradient_fea = y[num] * NLogisticfunc * data[fea] + (1 - y[num]) * PLogisticfunc * (-data[fea])
                            dwj[fea] = dwj[fea] + gradient_fea
                # Update the weights.
                self.weights = self.weights + self.learning_rate * dwj
        else:
            # This part will be run if we don't give any command parameter about the number of feature selection.
            # Then we will do logistic regression and gradient descent on total features.
            for cont in range(self.iterations):
                dwj = np.zeros([num_features],dtype=np.float)

            X = X.toarray()

            for num in range(num_examples):
                data = X[num,:]
                res_product = np.dot(self.weights,data)
                PLogisticfunc = expit(res_product)
                NLogisticfunc = expit(-res_product)

                for fea in range(num_features):
                    gradient_fea = y[num] * NLogisticfunc * data[fea] + (1 - y[num]) * PLogisticfunc * (-data[fea])
                    dwj[fea] = dwj[fea] + gradient_fea

            # Update the weights.
            self.weights = self.weights + self.learning_rate * dwj

    def predict(self, X):
        # TODO: Write code to make predictions.
        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')
        # Perhaps fewer features are seen at test time than train time, in
        # which case X.shape[1] < self.num_input_features. If this is the case,
        # we can simply 'grow' the rows of X with zeros. (The copy isn't
        # necessary here; it's just a simple way to avoid modifying the
        # argument X.)
        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        # Or perhaps more features are seen at test time, in which case we will
        # simply ignore them.
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        y_hat = np.empty([num_examples], dtype=int)
        # Create a "y_hat" array in order to store final predicted y labels.

        X = X.toarray()
        for num in range(num_examples):
            data = X[num,:]
            res_product = np.dot(self.weights,data)
            logistic_func = expit(res_product)
            # Do logistic regression computation to get the probability.

            if logistic_func >= 0.5:
                y_hat[num] = 1
            else:
                y_hat[num] = 0
            # Do the judgement to predict the predicted binary label.

        return y_hat

# TODO: Add other Models as necessary.
