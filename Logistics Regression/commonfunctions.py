import numpy as np

# Visualize the training set results
def plot_decision_boundary(X, Y, num_classes, step, classifier, plt):
    # Plot the data points         
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']  # add more colors if more classes 
    for i in range(0, num_classes):
        plt.scatter(x = X[Y == i, 0], y = X[Y == i, 1], color = colors[i], marker = '.')
     

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
 
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)  
   
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)  
    
# function to plot decision boundaries
def plot_classification_summary(X_train, Y_train, X_test, Y_test, classifier, plt):
    f, axs = plt.subplots(1,2,figsize=(10,5))
    
    plt.subplot(1, 2, 1)
    plot_decision_boundary(X_train, Y_train, 2, 0.02, classifier, plt)
    plt.title('Decision boundary for training data')
    plt.xlabel('Length')
    plt.ylabel('Width')
    
    plt.subplot(1, 2, 2)
    plot_decision_boundary(X_test, Y_test, 2, 0.02, classifier, plt)
    plt.title('Decision boundary for test data')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.tight_layout()
    plt.show()
    
# function to print classifier results
def print_classification_results(Y_true, Y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report    
    print('Confusion Matrix:\n', confusion_matrix(Y_true, Y_pred))
    print('Accuracy score:', accuracy_score(Y_true, Y_pred))
    print('Classification Report:\n',classification_report(Y_true, Y_pred))