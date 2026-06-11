"""
Iris Classifier – CLI Training & Evaluation
For batch runs, debugging, or headless environments.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    # Load
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print("🌸 Iris Dataset loaded")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Classes: {target_names}\n")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = knn.predict(X_test_scaled)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Example prediction on a custom flower
    print("\n--- Example Prediction ---")
    # Sample: sepal length 5.1, sepal width 3.5, petal length 1.4, petal width 0.2 -> Setosa
    sample = [[5.1, 3.5, 1.4, 0.2]]
    sample_scaled = scaler.transform(sample)
    pred = knn.predict(sample_scaled)[0]
    proba = knn.predict_proba(sample_scaled)[0][pred]
    print(f"Input: {sample[0]}")
    print(f"Predicted: {target_names[pred]} (confidence: {proba:.1%})")

if __name__ == "__main__":
    main()