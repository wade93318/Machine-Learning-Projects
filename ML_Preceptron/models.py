import numpy as np


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


class SumOfFeatures(Model):

    def __init__(self):
        super().__init__()
        # TODO: Initializations etc. go here.

    def fit(self, X, y):
        # NOTE: Not needed for SumOfFeatures classifier. However, do not modify.
        pass

    def predict(self, X):
        data = X
        # Put the input data without feature labels into variable "data".
        num_examples = X.shape[0]
        # Get the total number of data example and put this value into variable "num_examples".
        y_hat = np.ones((num_examples,1),dtype=np.int32)
        # Create an array which size is num_examples * 1 and all of elements in it will be "1" by using np.ones.
        # Put this array to the variable "y_hat".
        length = data[0,:].shape[1] 
        #get the length of one row. In other words, we need to know the number of features for each data example.

        # Now, we start to classify every data sample by according to the rules of "Sum of Features".
        for i in range(num_examples):
            # If the length of one row is even, we don't need to omit the feature in the middle.
            if length % 2 == 0:
                if np.sum(data[i,:int((length/2))],axis=1) >= np.sum(data[i,int((length/2)):],axis=1):
                    y_hat[i] = 1
                    # Update the label info.
                else:
                    y_hat[i] = 0
                    # Update the label info.
            else:
            # If the length of one row is odd, we need to omit the feature in the middle.
                if np.sum(data[i,:int((length-1)/2)],axis=1) >= np.sum(data[i,int(((length-1)/2)+1):],axis=1):
                    y_hat[i] = 1
                    # Update the label info.
                else:
                    y_hat[i] = 0
                    # Update the label info.

        #Return all the predicted labels stored in "y_hat" array.
        return y_hat

class Perceptron(Model):

    def __init__(self,learning_rate,iterations):
        super().__init__()
        # TODO: Initializations etc. go here.
        self.learning_rate = learning_rate
        # Add arribute "learning_rate" which can be customized thorugh command line.Hence, we can change number of it.
        self.iterations = iterations
        # Add attribute "iterations" in order to change the number of iteration of training model.
        self.w = None
        # Define the "w" which is the weight vector of Perceptron.Initialize w as None.

    def fit(self, X, y):
        self.num_input_features = X.shape[1]

        num_features = X.shape[1]
        num_examples = X.shape[0]
        # Get the number of features and number of examples.

        self.w = np.zeros((1,num_features))
        # Use np.zeros to create an array which size is (1,num_features) for self.w.
        for count in range(self.iterations):
            # According to the number of self.iterations to loop the training process.
            for num in range(num_examples):
                # Calculate the number of iteration of total number of data examples.
                data = X[num,:].toarray().flatten()
                # Get one row (one data example) every time and change the data format from csr_matrix to array.
                res_product = np.dot(self.w,data)
                # Do dot product between weight vector and one data example.
                if res_product >= 0:
                    #If w · xi ≥ 0, the preticted value is 1
                    # y_hat is the predicted label
                    y_hat = 1
                else:
                    y_hat = 0
                    #If w · xi < 0, the preticted value is 0
                    # y_hat is the predicted label

                if y_hat != y[num]:
                    # If the predicted label is not equal to original true label, we need to update the weight vector.
                    if y[num] == 0:
                        self.w = self.w + self.learning_rate * -1 * data
                    else:
                        self.w = self.w + self.learning_rate * 1 * data

        # TODO: Write code to fit the model.

    def predict(self, X):
        # TODO: Write code to make predictions.

        if self.num_input_features is None:
            raise Exception('fit must be called before predict.')

        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)

        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]

        #Above code can help to avoid the some potential problems caused by size of features in different dataset.

        num_examples = X.shape[0]
        num_input_features = X.shape[1]
        # Get the number of features and number of examples.
        y_hat = np.empty((num_examples,1), dtype=int)
        # Create an initialization array for y_hat in order to store the values of total predicted labels.

        for num in range(num_examples):
            # Loop for every row data sample.
            data = X[num,:].toarray().flatten()
            # Get one row (one data example) every time and change the data format from csr_matrix to array.
            res_product = np.dot(self.w,data)
            # Do the dot product between weight vector and one data example.
            if res_product < 0:
                #If w · xi ≥ 0, the preticted value is 1
                # y_hat is the predicted label
                y_hat[num] = 0
            else:
                #If w · xi < 0, the preticted value is 1
                # y_hat is the predicted label
                y_hat[num] = 1

        return y_hat
        # Return the final total predicted label for all of data test example.

# TODO: Add other Models as necessary.
