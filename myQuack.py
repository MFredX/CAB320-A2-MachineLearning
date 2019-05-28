
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# Common imports
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (9879120, 'Payagalagae', 'Fernando'), (9860665, 'Manavdeep', 'Signh') ]
   

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    #A seperate extraction is done to ensure that the string is extractable
    mydatastring=np.genfromtxt(dataset_path,delimiter=",",dtype="U1")
    #Another array is created to extract data in the float64 data type
    mydata=np.genfromtxt(dataset_path,delimiter=",")
    #Conduting the appropriate array slicing
    #X is a 2D array that includes respective attributes excluding the ID number
    X=mydata[:,2:]
    #y is a one dimensional array containing the classes M or B 
    y=mydatastring[:,1]
    #Converting M-->1 B-->0
    for i in range(len(y)):
        if y[i]=="M":
            y[i]=1
        else:
            y[i]=0
    y = y.astype(np.float)

    
    
    return X,y


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"   
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    
    # Train Decision Tree Classifer
    clf=DecisionTreeClassifier(criterion = "gini")
    clf = clf.fit(X_training,y_training)
    
    
    return clf
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":  
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerLine2D
    from sklearn.metrics import accuracy_score

    X,y=prepare_dataset('medical_records.data');
    ratio_train, ratio_test = 0.8 , 0.2
    
    #Creating the training data set 80% of the data
    X_training, X_testandVal, y_training, y_testandVal = train_test_split(X, y, train_size=ratio_train,test_size=ratio_test, shuffle=False)
    
    #The validation and testing sets are created 
    #from th remaining 20% by splitting that in to two 10% parts
    test_ratio=0.5
    X_validation,X_test,y_validation,y_test=train_test_split(X_testandVal,y_testandVal,test_size=test_ratio,shuffle=False)
    
    def optimal_Max_branch_DT():
        '''  
        Plots graphs and demonstrates the nature in which the optimal value of
        man_branch hyper parameter have been obtained for a Decision Tree Classifier.
        The plotted graphs are;
        Tree Depth vs AUC score
        Tree Depth vs Mean Accuracy score
        Tree Depth vs Cross Validated Accuracy Score
        
        @param 
        none
    
        @return
        	clf : the classifier with the optimal max_branch hyper-parameter
        '''
    
        # call your functions here
        #clf=build_DecisionTree_classifier(X_training, y_training)
        #Predict the response for test dataset

        
        #print(cross_val_score(clf, X_training, y_training, cv=3, scoring="accuracy"))
        
        #Tuning the max_depth hyper parameter for the decision tree classifier
        #The area under the curve will be used as a metric since this is a binary
        #classification problem
        
        

        
        #Creating an array of max_depths to loop through to find
        #optimal hyper parameter
        max_depths = np.linspace(1, 30, 30)
        
        training_results = []
        testing_results = []
        testing_meanScoreresults = []
        training_meanScoreresults = []
        cross_Vals=[]
        
        
        
        
        #Iterating through max_depths array to find the optimal value of max_depth
        for i in max_depths:
            dt = DecisionTreeClassifier(max_depth=i)
            #Building the classifier from the training data sets
            dt.fit(X_training, y_training)
            
            #Predicting the response of training data set
            train_pred = dt.predict(X_training)
            
            #A Reciever Operating Characteristic Curve is computed
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_training,y_score=train_pred)
            
            #Area under the Reciever Operating Characteristic Curve is computed from the
            #prediction scores
            roc_auc = auc(false_positive_rate, true_positive_rate)
            
            # Add auc score to previous train results
            training_results.append(roc_auc)
            
            #Predicting the response of the test data set
            y_pred = dt.predict(X_test)
            
             #A Reciever Operating Characteristic Curve is computed from the result of 
             #
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            
            #Area under the Reciever Operating Characteristic Curve is computed from the
            #prediction scores
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # Add auc score to previous test results
            testing_results.append(roc_auc)
            
            #Getting Mean Score Results for Training
            #This is the mean accuracy on the given test data and labels
            #Training data and Training labels
            l=dt.score(X_training, y_training)
            training_meanScoreresults.append(l)
            
            #Getting Mean Score Results for Testing
            #This is the mean accuracy on the given training data and labels
            #Testing data and Testing labels
            s=dt.score(X_test, y_test)
            testing_meanScoreresults.append(s)
            
            #Obtain cross_val_score for DecisionTree classifier with max_depth=i
            crossvals=cross_val_score(dt, X_training, y_training, cv=5,scoring='accuracy')
            #Appending the mean score to the scores list
            cross_Vals.append(crossvals.mean())
            
        
        
        #Plotting the the respective areas under the curve
        line1, = plt.plot(max_depths, training_results, 'b', label='Train AUC')
        line2, = plt.plot(max_depths, testing_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('Score')
        plt.xlabel('Tree depth')
        plt.show()
        
        #Plotting the the respective mean score results
        line3, = plt.plot(max_depths, testing_meanScoreresults, 'g', label='Testing Mean Score Results')
        line4, = plt.plot(max_depths, training_meanScoreresults, 'y', label='Training Mean Score Results')
        plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
        plt.ylabel('Mean Accuracy Score')
        plt.xlabel('Tree depth')
        plt.show()
        
        #Plotting the the mean cross valuation score as tree depth changes
        line5,= plt.plot(max_depths,cross_Vals,'c',label='Mean Cross Val Score')
        plt.legend(handler_map={line5: HandlerLine2D(numpoints=1)})
        plt.ylabel('Cross Validated Accuracy Score')
        plt.xlabel('Tree depth')
        plt.show()
        
        
        #We decicde that the optimal hyper parameter for this instance is max_depth=5
        #Creating the classifier with above said hyper parameter
        dt_optimal = DecisionTreeClassifier(max_depth=5)
        #Building the classifier from the training data sets
        dt_optimal.fit(X_training,y_training)
        #should I use X_test or X_validation?????
        #Predicting the class for X
        y_pred_optimal=dt_optimal.predict(X_validation)
        #Computing the accuracy score by comparing the predicted set
        #with the validation set
        print ("Accuracy is", accuracy_score(y_validation,y_pred_optimal)*100)
        return dt_optimal
    
    #optimal_Max_branch_DT()
    
    def optimal_num_of_neighbours_NNC():
        '''  
        Plots graphs and demonstrates the nature in which the optimal value of
        "number of neighbours" hyper-parameter have been obtained for a 
        K Nearest Neighbours Classifier.
        The plotted graphs are;
        Number of Neighbours vs AUC score
        Number of Neighbours vs Mean Accuracy score
        Number of Neighbours vs Cross Validated Accuracy Score
        
        @param 
        none
    
        @return
        	nn- a K Nearest Neighbours Classifier with optimal Number of Neighbours
        '''
        
        #Tuning the number of neighbours hyper-parameter for the K Nearest Neighbours Classifier
        #The area under the curve will be used as a metric since this is a binary
        #classification problem
        
        
        #Creating an array of max_depths to loop through to find
        #optimal hyper parameter
        max_neighbours=np.linspace(1, 30, 30)
        max_neighbours=max_neighbours.astype(int)
        training_results = []
        testing_results = []
        testing_meanScoreresults = []
        training_meanScoreresults = []
        cross_Vals=[]

        for i in max_neighbours:
            nn=KNeighborsClassifier(n_neighbors=i)
            #Building the classifier from the training data sets
            nn.fit(X_training, y_training)
            
            #Predicting the response of training data set
            train_pred = nn.predict(X_training)
            
            #A Reciever Operating Characteristic Curve is computed
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true=y_training,y_score=train_pred)
            
            #Area under the Reciever Operating Characteristic Curve is computed from the
            #prediction scores
            roc_auc = auc(false_positive_rate, true_positive_rate)
            
            # Add auc score to previous train results
            training_results.append(roc_auc)
            
            #Predicting the response of the test data set
            y_pred = nn.predict(X_test)
            
             #A Reciever Operating Characteristic Curve is computed from the result of 
             #
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
            
            #Area under the Reciever Operating Characteristic Curve is computed from the
            #prediction scores
            roc_auc = auc(false_positive_rate, true_positive_rate)
            # Add auc score to previous test results
            testing_results.append(roc_auc)

            #Getting Mean Score Results for Training
            #This is the mean accuracy on the given test data and labels
            #Training data and Training labels
            l=nn.score(X_training, y_training)
            training_meanScoreresults.append(l)
            
            #Getting Mean Score Results for Testing
            #This is the mean accuracy on the given training data and labels
            #Testing data and Testing labels
            s=nn.score(X_test, y_test)
            testing_meanScoreresults.append(s)
            
            #Obtain cross_val_score for K Nearest Neighbours classifier with n_neighbors=i
            crossvals=cross_val_score(nn, X_training, y_training, cv=5,scoring='accuracy')
            #Appending the mean score to the scores list
            cross_Vals.append(crossvals.mean())
            
            
        #Plotting the the respective areas under the curve
        line1, = plt.plot(max_neighbours, training_results, 'b', label='Train AUC')
        line2, = plt.plot(max_neighbours, testing_results, 'r', label='Test AUC')
        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.ylabel('Score')
        plt.xlabel('Number of Neighbours')
        plt.show()

        #Plotting the the respective mean score results
        line3, = plt.plot(max_neighbours, testing_meanScoreresults, 'g', label='Testing Mean Score Results')
        line4, = plt.plot(max_neighbours, training_meanScoreresults, 'y', label='Training Mean Score Results')
        plt.legend(handler_map={line3: HandlerLine2D(numpoints=2)})
        plt.ylabel('Mean Accuracy Score')
        plt.xlabel('Number of Neighbours')
        plt.show()
    
        #Plotting the the mean cross valuation score as tree depth changes
        line5,= plt.plot(max_neighbours,cross_Vals,'c',label='Mean Cross Val Score')
        plt.legend(handler_map={line5: HandlerLine2D(numpoints=1)})
        plt.ylabel('Cross Validated Accuracy Score')
        plt.xlabel('Number of Neighbours')
        plt.show()
            
        
        #We decicde that the optimal hyper parameter for this instance is n_neighbors=8
        #Creating the classifier with above said hyper parameter
        nn_optimal = KNeighborsClassifier(n_neighbors=8)
        #Building the classifier from the training data sets
        nn_optimal.fit(X_training,y_training)
        #should I use X_test or X_validation?????
        #Predicting the class for X
        y_pred_optimal=nn_optimal.predict(X_validation)
        #Computing the accuracy score by comparing the predicted set
        #with the validation set
        print ("Accuracy of the optimised NNC is", accuracy_score(y_validation,y_pred_optimal)*100)
        return nn_optimal
    
    optimal_num_of_neighbours_NNC()


        
            
        

