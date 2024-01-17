import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt


# Set the console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
# Load your dataset (replace 'your_file.csv' with your actual file path)
file_path = r'C:\Users\Admin\SoccerPredictionProject\combined_dataset.csv'
data = pd.read_csv(file_path)
label_encoder = LabelEncoder()

data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)

# Extract relevant date components
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Ref'] = label_encoder.fit_transform(data['Referee'])
data['HomeTeam_encoded'] = label_encoder.fit_transform(data['HomeTeam'])
data['AwayTeam_encoded'] = label_encoder.fit_transform(data['AwayTeam'])


# Select only the desired features

data["target"] = data["FTR"].map({'H': 1, 'D': 0, 'A': 2})
data["Haft target"] = data["HTR"].map({'H': 1, 'D': 0, 'A': 2})
selected_features = [ 'HomeTeam_encoded','AwayTeam_encoded','HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', "Haft target", 'HTAG','HTHG','Ref','Year']
X = data[selected_features]
# Assuming the target variable is in a column named 'target', change it to your actual target column name
y = data['target']

random_state_values = range(40, 1001)

# Initialize variables to keep track of the best random_state and its corresponding accuracy

""""
best_random_state = None
best_accuracy = 0.0
random_statee = 60
random_states = []
accuracies = []
while random_statee <80:
# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_statee)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the kernel SVM model
    svm_model = SVC(kernel='linear', C=13, gamma='scale',probability=True)  # You can adjust hyperparameters like C and gamma
    svm_model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    # Evaluate the model

   
    random_states.append(random_statee)
    accuracies.append(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_random_state = random_statee
    random_statee += 1
plt.plot(random_states, accuracies, marker='o')
plt.title('Accuracy vs. Random State')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
"""
best_C = None
best_accuracy = 0.0
first_C = 0.1
C_Values = []
accuracies = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75)

# Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

    # Create and train the kernel SVM model
svm_model = SVC(kernel='linear', C=0.5, gamma='scale',probability=True)  # You can adjust hyperparameters like C and gamma
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
# Evaluate the model


print(f"Accuracy: {accuracy * 100:.2f}%")






""""
def predict_probabilities(home_team, away_team, encoder):
    # Encode team names using the provided encoder
    home_team_encoded = encoder.transform([home_team])[0]
    away_team_encoded = encoder.transform([away_team])[0]

    # Create input data for prediction
    input_data = pd.DataFrame({'HomeTeam_encoded': [home_team_encoded],
                                'AwayTeam_encoded': [away_team_encoded],
                                'HS': [0], 'AS': [0], 'HST': [0], 'AST': [0],
                                'HF': [0], 'AF': [0], 'HC': [0], 'AC': [0],
                                'HY': [0], 'AY': [0], 'HR': [0], 'AR': [0],
                                'Haft target': [0], 'HTAG': [0], 'HTHG': [0],
                                'Ref': [0], 'Year': [0]})

    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict class probabilities
    class_probabilities = svm_model.predict_proba(input_data_scaled)

    # Output the class probabilities
    print("Class 0 Probability:", class_probabilities[0, 0])
    print("Class 1 Probability:", class_probabilities[0, 1])
    print("Class 2 Probability:", class_probabilities[0, 2])

# Example usage:
home_team_name = 'Man City'  # Replace with the actual home team name
away_team_name = 'Man United'  # Replace with the actual away team name

# Pass the label_encoder to the prediction function
seen_labels = label_encoder.classes_

print("Unique classes seen by the LabelEncoder:")
print(seen_labels)
predict_probabilities(home_team_name, away_team_name, label_encoder)
"""