# asthma-disease-prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('/content/drive/MyDrive/asthma.csv')
data
data.shape
data.info()
data.head()
data.tail()
data.describe()
data["Gender_Female"].value_counts()
data["Gender_Male"].value_counts()
data["Severity_Mild"].value_counts()
data["Severity_Moderate"].value_counts()
data["Severity_None"].value_counts()
import matplotlib.pyplot as plt
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Severity_None']==1]['Nasal-Congestion'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('Having asthma')

data_len=data[data['Severity_None']==0]['Nasal-Congestion'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having asthma')

fig.suptitle('Nasal-Congestion')
plt.show()
data.duplicated()
newdata=data.drop_duplicates()
newdata
data.isnull().sum() #checking for total null values
data[1:5]
from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["Difficulty-in-Breathing", "Dry-Cough", "Tiredness", "Sore-Throat"])
scaled_df.head()
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Severity_None'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Severity_None']
len(train_X), len(train_Y), len(test_X), len(test_Y)
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)

# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)
