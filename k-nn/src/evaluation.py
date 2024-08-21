from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model performance by calculating accuracy, confusion matrix, 
    and generating a classification report.
    """
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate a confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate a classification report (precision, recall, F1-score)
    report = classification_report(y_true, y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    return accuracy, cm, report
