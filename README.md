# ProjectModels
A repo to run and test different ML models in the Titanic dataset from Kaggle.

This repo is divided into 3 different families of ML models, considering 2 different train_sets.

Training Sets:
    1. Training set consisting of the usual Kaggle Titanic dataset (just rounding the age and dropping some other cols).
    2. Training set of hot-encoding, adding 1 feature engineered col, transforming the ~10cols of dataset 1 to ~70 cols (improved XGBoost model by ~4%). 

ML Models:
    1. Logistic Regression (Train Set 1):
        a. Sklearn imported logistic Regression model;
        b. Built-in Logistic Regression (using the Andrew Ng lessons as parameters) - basic structure of a sigmoid activation function, cost function and a 
            Gradient Descent algorithm that runs model b. in almost the same performance as a..

    2. Neural Networks (Train Set 1):
        a. TF Keras imported model consisting in 4 layers (input, 2 inner layers, output), using the sigmoid activation function (despite the better  
            performance of the ReLu), Adam optimizer and Binary Cross Entropy as loss function; also implemented an early-stopping function with a 25 epoch 
            patience on the validation data loss maintence value.
        b. Built-in NN model (2 in total: 1. with vector linearization to improve the performance (np.dot vs matmul)); used the calculated weights in the  
            2. a. NN model and used the sigmoid activation function as a forward propagation model, achieving the same performance (slightly faster run time, however there's not a great malleability in the model parameters).

    3. Decision Tree / Random Forrest - XGBoost (Train Set 2):
        a. Used the XGBost Classifier imported model, considering the Train Set 2 (approximately 70 cols w/ one-hot encoding in all). Used Grid_Search on 
            the set to toggle parameters, increasing the performance (~4%).
        b. Built-in Decision Tree Classifier using the entropy and Information Gain algorithms to achieve the best tree, however this model did not perform 
            adequately (needs improving in the train set also).

