from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(features, labels, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Args:
        features (np.array): Array of feature vectors.
        labels (list): List of labels.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where:
            - X_train: Features for the training set.
            - X_test: Features for the testing set.
            - y_train: Labels for the training set.
            - y_test: Labels for the testing set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """
    Train an SVM model with hyperparameter tuning and evaluate it.

    Args:
        X_train (np.array): Features for the training set.
        X_test (np.array): Features for the testing set.
        y_train (list): Labels for the training set.
        y_test (list): Labels for the testing set.

    Returns:
        model: The trained SVM model with the best hyperparameters.
    """
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'rbf'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=SVC(probability=True, random_state=42),  # Enable probability for predict_proba
        param_grid=param_grid,  # Hyperparameter grid
        cv=5,  # 5-fold cross-validation
        scoring='accuracy',  # Metric to optimize
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit GridSearchCV to find the best model
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_accuracy * 100:.2f}%")
    print(f"Test set accuracy: {test_accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return best_model