
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import accuracy_score
import statistics
from scipy import stats
from sklearn import svm
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
    from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()
    
    # Train and building Decision Tree Classifer
    #We have discovered that the best value for max_depth is 3 based on our analysis.
    clf=DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(X_training,y_training)
    
    return clf

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
    
    X,y=prepare_dataset('medical_records.data');
    ratio_train, ratio_test = 0.8 , 0.2
    
    #Creating the training data set of 80%
    # Testing set is made from the remaining 20%
    X_training, X_test, y_training, y_test = train_test_split(X, y, train_size=ratio_train,test_size=ratio_test, shuffle=True,random_state=7654)
    
    def optimal_Max_depth_DT():
        '''  
        Optimal max_depth hyper parameter is obtained for the Decision Tree Classifier.
        This is found using cross validation
        
        @param 
        none

        @return
        final : the classifier with the optimal max_depth hyper-parameter
        '''       
        #Creating an array of max_depths to loop through to find
        #optimal hyper parameter
        max_depths = np.linspace(1, 30, 30)
        
        #List to store optimal max_depth values of tests
        optimal_depths=[]
        #max_depth of the model is found 30 times
        for j in range(0,30):
            #Iterating through max_depths array to find the optimal value of max_depth
            cross_Vals=[]
            for i in max_depths:
                dt = DecisionTreeClassifier(max_depth=i)
                #Building the classifier from the training data sets
                dt.fit(X_training, y_training)
                #Obtain cross_val_score for DecisionTree classifier with max_depth=i
                crossvals=cross_val_score(dt, X_training, y_training, cv=5,scoring='accuracy')
                #Appending the mean score to the scores list
                cross_Vals.append(crossvals.mean())
    
            #Finding highest Cross Validated Accuracy Score
            maxCVAS = np.amax(cross_Vals)
            #Finding relevant index where the highest Cross Validated Score is present
            result = np.where(cross_Vals == np.amax(cross_Vals))
            #Retriving the optimal_depth value by indexing the above index found above
            #on max_depths 
            optimal_depth=max_depths[result]
            #print(result)
            
            #Creating the classifier with above said hyper parameter
            dt_optimal = DecisionTreeClassifier(max_depth=optimal_depth[0])
            #Building the classifier from the training data sets
            dt_optimal.fit(X_training,y_training)
            #optimal_crossvals=cross_val_score(dt_optimal, X_training, y_training, cv=5,scoring='accuracy')
            #Predicting the class for X
            y_pred_optimal=dt_optimal.predict(X_test) 
            #Computing the accuracy score by comparing the predicted set
            #with the validation set
            print('This is test number:',(j+1))
            print('The maximum value of CVAS of this test is is:',maxCVAS)
            print('Optimal Depth of this test is:',optimal_depth[0])
            #print('Optimal Cross Val of this is:',(optimal_crossvals.mean()))
            print ("Accuracy is", accuracy_score(y_test,y_pred_optimal)*100)
            print('----------------------------------------------')
            
            #Optimal depth in each test is found and added to a list
            optimal_depths.append(optimal_depth[0])
        
        print("Evaluvating the optimized model")
        print(optimal_depths)
        #print("The mean of the optimal_depths list is:",statistics.mean(optimal_depths))
        #The max_depth that was most frequent in our tests was selected as the final
        # value of max depth
        mode_optimal_depths=stats.mode(optimal_depths)[0][0]
        print("The mode of the optimal_depths list is",mode_optimal_depths)
        #Final classifier will be made with the depth that was found to be optimal in most 
        # of our tests
        dt_final = DecisionTreeClassifier(max_depth=mode_optimal_depths)
        #Building final classifier
        dt_final.fit(X_training,y_training)
        #Testing final model
        y_pred_final=dt_final.predict(X_test)
        print("Accuracy of the final model is:",accuracy_score(y_test, y_pred_final))
        print('----------------------------------------------')
        
        return dt_final;

    
    
    
    def optimal_num_of_neighbours_NNC():
        '''     
        Optimal n_neighbors hyper-parameteris obtained for the  
        K Nearest Neighbours Classifier using cross validation
            
        @param 
        none
    
        @return
        	clf : K Nearest Neighbours Classifier with the optimal n_neighbors hyper-parameter
        '''
        
        #Creating an array of max_depths to loop through to find
        #optimal hyper parameter
        max_neighbours = np.linspace(1, 30, 30)
        max_neighbours=max_neighbours.astype(int)
        
        #List to store optimal max_neighbours values of tests
        optimal_ns=[]
        #max_depth of the model is found 30 times
        for j in range(0,30):
            #Iterating through max_depths array to find the optimal value of max_neighbours
            #List to store results from cross validation
            cross_Vals=[]
            for i in max_neighbours:
                nn = KNeighborsClassifier(n_neighbors=i)
                #print(i)
                #Building the classifier from the training data sets
                nn.fit(X_training, y_training)
                #Obtain cross_val_score for DecisionTree classifier with max_depth=i
                crossvals=cross_val_score(nn, X_training, y_training, cv=5,scoring='accuracy')
                #Appending the mean score to the scores list
                cross_Vals.append(crossvals.mean())
                #print(crossvals.mean())
    
            #Finding highest Cross Validated Accuracy Score
            maxCVAS = np.amax(cross_Vals)
            #Finding relevant index where the highest Cross Validated Score is present
            result = np.where(cross_Vals == np.amax(cross_Vals))
            #Retriving the optimal_depth value by indexing the above index found above
            #on max_neighbours 
            optimal_n=max_neighbours[result]
            
    
            #Creating the classifier with above said hyper parameter
            nn_optimal = KNeighborsClassifier(n_neighbors=optimal_n[0])
            #Building the classifier from the training data sets
            nn_optimal.fit(X_training,y_training)
            #optimal_crossvals=cross_val_score(dt_optimal, X_training, y_training, cv=5,scoring='accuracy')
            #Predicting the class for X
            y_pred_optimal=nn_optimal.predict(X_test) 
            print('This is test model number:',(j+1))
            print('The maximum value of CVAS of this test is is:',maxCVAS)
            print('Optimal Depth of this test is:',optimal_n[0])
            #print('Optimal Cross Val of this is:',(optimal_crossvals.mean()))
            #Finding accuracy
            print ("Accuracy is", accuracy_score(y_test,y_pred_optimal)*100)
            print('----------------------------------------------')
            
            #Optimal depth in each test is found and added to a list
            optimal_ns.append(optimal_n[0])
        
        print("Evaluvating the optimized model")
        print(optimal_ns)
        #The max_depth that was most frequent in our tests was selected as the final
        # value of max depth
        mode_optimal_ns=stats.mode(optimal_ns)[0][0]
        print("The mode of the n_neighbors list is",mode_optimal_ns)
        #Final classifier will be made with the depth that was found to be optimal in most 
        # of our tests
        nn_final = KNeighborsClassifier(n_neighbors=mode_optimal_ns)
        #Building final classifier
        nn_final.fit(X_training,y_training)
        #Testing final model
        y_pred_final=nn_final.predict(X_test)
        print("Accuracy of the final model is:",accuracy_score(y_test, y_pred_final))
        print('----------------------------------------------')
        
        return nn_final;
        
    
    
    def optimal_num_of_neurons_NeuralNetwork_C():
        
        
        
        neurons=np.linspace(10, 15, 5)
        #neurons=np.linspace(10,50,40)
        #Toggle between range of neurons variable for increased accuracy at
        # the cost of longer processing time
        neurons=neurons.astype(int)
        
        optimal_neurons=[]
        #for j in range(0,31):
        #Toggle between range of the for loops for increased accuracy at
        # the cost of longer processing time
        for j in range(0,4):
            cross_Vals=[]
            from sklearn.neural_network import MLPClassifier
            for i in neurons:
                hidden_layers=[i,i,i]  # define the layers/depth of the NN
                mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, verbose=True)
                mlp.fit(X_training, y_training)  # fit features over NN
                
                #Obtain cross_val_score for K Nearest Neighbours classifier with n_neighbors=i
                crossvals=cross_val_score(mlp, X_training, y_training, cv=5,scoring='accuracy')
                #Appending the mean score to the scores list
                cross_Vals.append(crossvals.mean())
                
            #Finding highest Cross Validated Accuracy Score
            maxCVAS = np.amax(cross_Vals)
            #Finding relevant index where the highest Cross Validated Score is present
            result = np.where(cross_Vals == np.amax(cross_Vals))
            #Retriving the optimal_depth value by indexing the above index found above
            #on max_depths 
            
            optimal_neuron=neurons[result]
            print(optimal_neuron[0])
            
            #Creating the classifier with above said hyper parameter
            optimal_hidden_layers=[optimal_neuron[0],optimal_neuron[0],optimal_neuron[0]]
            mlp_optimal = MLPClassifier(hidden_layer_sizes=optimal_hidden_layers,verbose=True)
            #Building the classifier from the training data sets
            mlp_optimal.fit(X_training,y_training)
            #optimal_crossvals=cross_val_score(dt_optimal, X_training, y_training, cv=5,scoring='accuracy')
            #Predicting the class for X
            y_pred_optimal=mlp_optimal.predict(X_test) 
            print('This is test model number:',(j+1))
            print('The maximum value of CVAS of this test is is:',maxCVAS)
            print('Optimal # neurons in the hidden layers of this test is:',optimal_neuron[0])
            #print('Optimal Cross Val of this is:',(optimal_crossvals.mean()))
            #Finding accuracy
            print ("Accuracy is", accuracy_score(y_test,y_pred_optimal)*100)
            print('----------------------------------------------')
            
            #Optimal depth in each test is found and added to a list
            optimal_neurons.append(optimal_neuron[0])
            
        print("Evaluvating the optimized model")
        print(optimal_neurons)
        #The max_depth that was most frequent in our tests was selected as the final
        # value of max depth
        mode_optimal_neurons=stats.mode(optimal_neurons)[0][0]
        print("The mode of the optimal_neurons list is",mode_optimal_neurons)
        #Final classifier will be made with the depth that was found to be optimal in most 
        # of our tests
        final_optimal_hidden_layers=[mode_optimal_neurons,mode_optimal_neurons,mode_optimal_neurons]
        mlp_final = MLPClassifier(hidden_layer_sizes=final_optimal_hidden_layers)
        #Building final classifier
        mlp_final.fit(X_training,y_training)
        #Testing final model
        y_pred_final=mlp_final.predict(X_test)
        print("Accuracy of the final model is:",accuracy_score(y_test, y_pred_final))
        print('----------------------------------------------')
        
        return mlp_final;
            
    
    
    def optimal_Cparam_SVM():
        C_list=np.array([0.01,0.1,1,10,100,1000])
        optimal_Cs=[]
        
        for j in range(0,30):
            cross_Vals=[]
            for i in C_list:
                clf = svm.SVC(C=i,gamma='scale')
                clf.fit(X_training, y_training)  # fit features over NN
                #Obtain cross_val_score for K Nearest Neighbours classifier with n_neighbors=i
                crossvals=cross_val_score(clf, X_training, y_training, cv=5,scoring='accuracy')
                #Appending the mean score to the scores list
                cross_Vals.append(crossvals.mean())
            
            #Finding highest Cross Validated Accuracy Score
            maxCVAS = np.amax(cross_Vals)
            #Finding relevant index where the highest Cross Validated Score is present
            result = np.where(cross_Vals == np.amax(cross_Vals))
            #Retriving the optimal_depth value by indexing the above index found above
            #on max_neighbours 
            optimal_C=C_list[result]
            print(optimal_C[0])
            
            #Creating the classifier with above said hyper parameter
            clf_optimal = svm.SVC(C=optimal_C,gamma='scale')
            #Building the classifier from the training data sets
            clf_optimal.fit(X_training,y_training)
            #optimal_crossvals=cross_val_score(dt_optimal, X_training, y_training, cv=5,scoring='accuracy')
            #Predicting the class for X
            y_pred_optimal=clf_optimal.predict(X_test) 
            print('This is test model number:',(j+1))
            print('The maximum value of CVAS of this test is is:',maxCVAS)
            print('Optimal Depth of this test is:',optimal_C[0])
            #print('Optimal Cross Val of this is:',(optimal_crossvals.mean()))
            #Finding accuracy
            print ("Accuracy is", accuracy_score(y_test,y_pred_optimal)*100)
            print('----------------------------------------------')
            
            #Optimal depth in each test is found and added to a list
            optimal_Cs.append(optimal_C[0])
            
        print("Evaluvating the optimized model")
        print("List of optimal C values found",optimal_Cs)
        #The max_depth that was most frequent in our tests was selected as the final
        # value of max depth
        mode_optimal_Cs=stats.mode(optimal_Cs)[0][0]
        print("The mode of the optimal_Cs list is",mode_optimal_Cs)
        #Final classifier will be made with the depth that was found to be optimal in most 
        # of our tests
        clf_final = svm.SVC(C=mode_optimal_Cs,gamma='scale')
        #Building final classifier
        clf_final.fit(X_training,y_training)
        #Testing final model
        y_pred_final=clf_final.predict(X_test)
        print("Accuracy of the final model is:",accuracy_score(y_test, y_pred_final))
        print('----------------------------------------------')
    
    #optimal_Max_depth_DT()
    optimal_num_of_neighbours_NNC()
    #optimal_Cparam_SVM()
    #optimal_num_of_neurons_NeuralNetwork_C()
    
            

        
            
        

