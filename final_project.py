import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# ==========================================
# DATA LOADING & PREPARATION
# ==========================================
df = pd.read_csv('exchange_rate.csv', sep=';')

# Input Features: Last 7 Days
X = df[['Day_1', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7']]
y = df['Target']

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardization (Z-score)
train_mean = X_train.mean()
train_std = X_train.std()

X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

# Add Bias column (x0 = 1) specifically for Custom Logistic Regression
X_train = np.c_[np.ones((X_train_scaled.shape[0], 1)), X_train_scaled]
X_test = np.c_[np.ones((X_test_scaled.shape[0], 1)), X_test_scaled]

Y_train = y_train.values
Y_test = y_test.values

# ==========================================
# CUSTOM LOGISTIC REGRESSION FUNCTIONS
# ==========================================
def initialize(dim):
    return np.random.rand(dim) * 0.01 

def sigmoid(x):
    # x = np.clip(x, -250, 250)
    return 1 / (1 + np.exp(-x))

def predict_Y(theta, X):
    return sigmoid(np.dot(X, theta))

def cost_function(y, y_hat):
    m = len(y)
    # epsilon = 1e-15  
    # y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    total_cost = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    return total_cost

def update_theta(x, y, y_hat, theta_o, learning_rate):
    dw = np.dot(x.T, (y_hat - y)) / len(y)
    return theta_o - learning_rate * dw

def run_gradient_descent(X, Y, alpha, num_iterations):

    theta = initialize(X.shape[1])
    cost_history = [] 
    
    for each_iter in range(num_iterations):
        Y_hat = predict_Y(theta, X)
        this_cost = cost_function(Y, Y_hat) 
        theta = update_theta(X, Y, Y_hat, theta, alpha)
        
        if(each_iter % 100 == 0):
            cost_history.append([each_iter, this_cost])
            
    return pd.DataFrame(cost_history, columns=['iteration', 'cost']), theta


# ==========================================
# TRAINING THE CUSTOM MODEL
# ==========================================
gd_iterations_df, trained_theta = run_gradient_descent(X_train, Y_train, alpha=0.03, num_iterations=5000)

# ==========================================
# PREDICTION AND EVALUATION (CUSTOM LR)
# ==========================================
print("\n" + "="*50)
print(" CUSTOM LOGISTIC REGRESSION EVALUATION")
print("="*50)

Y_test_prob = predict_Y(trained_theta, X_test)
THRESHOLD = 0.17
Y_test_pred_custom = (Y_test_prob >= THRESHOLD).astype(int)

accuracy_custom = accuracy_score(Y_test, Y_test_pred_custom)
print(f"Accuracy Score (with Threshold={THRESHOLD}): {accuracy_custom * 100:.2f}%\n")

# ----------------------------------------------------
# CONFUSION MATRIX & REPORT
# ----------------------------------------------------
print("--- CONFUSION MATRIX ---")
print(confusion_matrix(Y_test, Y_test_pred_custom))

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(Y_test, Y_test_pred_custom, zero_division=0))



# ====================================================
# 5. MODEL PERFORMANCE COMPARISON
# ====================================================
print("\n" + "="*50)
print(" MACHINE LEARNING MODELS COMPARISON")
print("="*50)

# Random Forest Classifier 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, Y_train)
rf_pred = rf_model.predict(X_test_scaled)
acc_rf = accuracy_score(Y_test, rf_pred)

# Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, Y_train)
svm_pred = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(Y_test, svm_pred)

print(f"[1] Custom Logistic Regression : {accuracy_custom * 100:.2f}%")
print(f"[2] Support Vector Machine     : {acc_svm * 100:.2f}%")
print(f"[3] Random Forest Classifier   : {acc_rf * 100:.2f}%")


# =======================
# PLOT LEARNING CURVE
# =======================
plt.figure(figsize=(8, 4))
plt.plot(gd_iterations_df['iteration'], gd_iterations_df['cost'], color='#1f77b4', linewidth=2)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost (Log Loss)")
plt.title("Gradient Descent: Learning Curve")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ====================================================
# REAL-WORLD FORECASTING (LAST 7 DAYS)
# ====================================================
# print("\n--- PREDICTION FOR RECENT 7 DAYS ---")
# Day = 1
# for i in range(len(X_test_lr) - 7, len(X_test_lr)):
#     prob = Y_test_prob[i] * 100
#     pred = "UP" if Y_test_prob[i] >= THRESHOLD else "DOWN/FLAT"
#     actual = "UP" if Y_test[i] == 1 else "DOWN/FLAT"
#     print(f"Day {Day} | UP Probability: {prob:5.1f}% -> Predicted: {pred:<9} (Actual: {actual})")
#     Day += 1
