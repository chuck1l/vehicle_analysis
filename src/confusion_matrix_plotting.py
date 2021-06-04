import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def create_cm_plot(y_test, y_hat):
    '''
    This function simply plots the results from XGBoost Classifier
    confusion matrix.

    Input: Y true values from the test data (y_test), and the 
    Y predicted values from our model (y_hat)

    Returns: Nothing, saves the image. 
    '''
    confusion = confusion_matrix(y_test, y_hat)
    # Metrics for Confusion Matrices
    accuracy = accuracy_score(y_test, y_hat)
    precision = precision_score(y_test, y_hat, average='macro')
    recall = recall_score(y_test, y_hat, average='macro')
    f1 = f1_score(y_test, y_hat, average='macro')
    
    stats_text = "\n\nAccuracy={:0.1f}%\nPrecision={:0.1f}%\nRecall={:0.1f}%\nF1 Score={:0.1f}%".format(
        accuracy,precision,recall,f1)

    # Plotting the confusion matrices
    group_names = ['True Neg', 'False Pos', 'False Neg' , 'True Pos']
    categories = ['Zero', 'One']

    group_counts = ['{:0.0f}'.format(value) for value in confusion.flatten()]

    group_percentages = ['{0:.1%}'.format(value) for value in
                        confusion.flatten()/np.sum(confusion)]

    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
            zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(
        confusion, annot=labels, fmt='', cmap='Blues', cbar=True,
        xticklabels=categories, yticklabels=categories
    )
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + stats_text)
    plt.title(f'Confusion Matrix For Trim Predictor')
    plt.tight_layout()
    plt.savefig(f'../img/confusion_matrix.png', dpi=500, orientation='landscape');

    return None