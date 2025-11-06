import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. Load and Prepare Data ---

print("Loading data...")
# Load the dataset
try:
    data = pd.read_csv("emails.csv")
except FileNotFoundError:
    print("Error: emails.csv file not found. Please make sure it's in the correct directory.")
    exit() # Exit the script if the file isn't found

# Define Features (X) and Target (y)
# X = all columns EXCEPT 'Email' and 'Prediction'
X = data.drop(columns=['Email', 'Prediction'])
# y = only the 'Prediction' column
y = data['Prediction']

print(f"Data loaded. Found {X.shape[0]} emails with {X.shape[1]} features each.")
print("-" * 30)

# --- 2. Split and Scale Data ---

print("Splitting and scaling data...")
# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
# SVM and KNN are sensitive to the scale of data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data ready for training.")
print("-" * 30)

# --- 3. K-Nearest Neighbors (KNN) ---

print("Training K-Nearest Neighbors (KNN) model...")
# Initialize the classifier (k=5 is a common default)
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the scaled training data
knn.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_knn = knn.predict(X_test_scaled)

# --- 4. Support Vector Machine (SVM) ---

print("Training Support Vector Machine (SVM) model...")
# Initialize the classifier (a linear kernel is fast and a good baseline)
svm = SVC(kernel='linear', random_state=42)

# Train the model on the scaled training data
svm.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred_svm = svm.predict(X_test_scaled)

print("Training complete.")
print("-" * 30)

# --- 5. Analyze and Compare Performance ---

print("\n========== KNN MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nConfusion Matrix (KNN):")
print(confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn))

print("\n========== SVM MODEL PERFORMANCE ==========")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print("\nConfusion Matrix (SVM):")
print(confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report (SVM):")
print(classification_report(y_test, y_pred_svm))


# --- 6. Quick Analysis ---

print("\n========== ANALYSIS SUMMARY ==========")
knn_acc = accuracy_score(y_test, y_pred_knn)
svm_acc = accuracy_score(y_test, y_pred_svm)

print(f"KNN Accuracy: {knn_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")

if svm_acc > knn_acc:
    print("\nConclusion: SVM performed better overall.")
elif knn_acc > svm_acc:
    print("\nConclusion: KNN performed better overall.")
else:
    print("\nConclusion: Both models performed equally well in terms of accuracy.")
