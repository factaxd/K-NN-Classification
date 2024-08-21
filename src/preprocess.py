import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path):
    """
    Load data from a CSV file and perform train/test split and normalization.
    """
    # Load data from the CSV file
    data = pd.read_csv(file_path)
    
    # Separate features (X) and target variable (y)
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
