import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("data/processed/k2_clean.csv", comment="#")
# print(df.head())
# print(df.dtypes)

#set up variables and known outputs
df = df.dropna(subset=["dispositionNum"])

X = df.drop(columns=["pl_name", "disposition", "dispositionNum"])  #Dropping Non-numerics
X = pd.get_dummies(X, columns=["discoverymethod"])
y = df["dispositionNum"]

#Set TRaining vs Testing split at 80/20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)


#OPtional Scaling For SVM or Logistic reg

# scaler = StandardScaler()
# X_train_scaled=scaler.transform(X_train)
# X_test_scaled= scaler.transform(X_test)

# Using RandomForest for easy Machine Learning Classifications

rain = RandomForestClassifier(n_estimators=100, random_state=42)
rain.fit(X_train, y_train)

# Eval

y_pred = rain.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))