# Importing the dataset
import pandas as pd
df = pd.read_csv("data.csv")
df.columns

# Label Encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
#male --> 1
#female --> 0
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state= 1)
print(X_train.shape)

# Fit the Linear SVC Classifier
from sklearn.svm import LinearSVC, SVC
classifier = LinearSVC()
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
classifier.score(X_test, y_test)

# Obtain Performance Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_predict, y_test))

# Fit the SVC Classifier
classifier = SVC()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)