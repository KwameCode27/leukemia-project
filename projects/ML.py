import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report , roc_curve, f1_score, accuracy_score, recall_score , roc_auc_score,make_scorer
import re
import json

#loading dataset
df = pd.read_csv('merged.csv' , encoding_errors= 'replace')

data = df.values

#create scaled DataFrame where each variable has mean of 0 and standard dev of 1
scaler = StandardScaler()
scaled_df = scaler.fit_transform(data)
plt.plot(scaled_df)
plt.show()

#view first five rows of scaled DataFrame
print(scaled_df[:4])
from sklearn.preprocessing import StandardScaler, normalize

# Normalizing the Data
X_normalized = normalize(scaled_df)
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

#making input varibles

x = data[:,0:17]
y = data[:,18]

#dataset visualization

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8, random_state=42, shuffle=True, stratify=None)
skf = StratifiedKFold(n_splits=10)

print("Number of  X_train dataset: ", X_train.shape)
print("Number of y_train dataset: ", y_train.shape)
print("Number of  X_test dataset: ", X_test.shape)
print("Number of  y_test dataset: ", y_test.shape)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print (X_train_smote.shape, y_train_smote.shape)
print (X_test.shape, y_test.shape)


def fun(y_pred):
    l = []
    for i in y_pred:
        res = list(map(int, re.findall('\d', str(i))))
        l.append(res[0])
    return l

# LogisticRegression Model
reg = LogisticRegression().fit(X_train_smote, y_train_smote)
y_pred = reg.predict(X_test)
print("Accuracy for LogisticRegression:",metrics.accuracy_score(y_test, fun(y_pred)))
print(classification_report(y_test,(fun(y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,fun(y_pred))
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for LR',fontsize=17)
plt.show()

# DecisionTreeClassifier

DT = DecisionTreeClassifier().fit(X_train_smote, y_train_smote)
y_pred = DT.predict(X_test)
print("Accuracy for DT",metrics.accuracy_score(y_test, fun(y_pred)))
print(classification_report(y_test,(fun(y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,fun(y_pred))
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for DT',fontsize=17)
plt.show()

#GradientBoostingClassifier

GB = GradientBoostingClassifier().fit(X_train_smote, y_train_smote)
y_pred = GB.predict(X_test)
print("Accuracy for GradientBoostingClassifier",metrics.accuracy_score(y_test, fun(y_pred)))
print(classification_report(y_test,(fun(y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,fun(y_pred))
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for GB',fontsize=17)
plt.show()

#RandomForestClassifier
RF = RandomForestClassifier().fit(X_train_smote, y_train_smote)
y_pred = RF.predict(X_test)
print(confusion_matrix(y_test, fun(y_pred)))
acc_train_log = metrics.accuracy_score(y_test, fun(y_pred))
print("Accuracy for RandomForestClassifier:",metrics.accuracy_score(y_test, fun(y_pred)))
print(classification_report(y_test,(fun(y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,fun(y_pred))
#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for RF',fontsize=17)
plt.show()

#NaiveBayess

model1 = MultinomialNB().fit(X_train_smote, y_train_smote)
y_pred = model1.predict(X_test)
print(confusion_matrix(y_test, fun(y_pred)))
acc_train_log = metrics.accuracy_score(y_test, fun(y_pred))
print("Accuracy for NaiveBayess:",metrics.accuracy_score(y_test, fun(y_pred)))
print(classification_report(y_test,(fun(y_pred))))

# compute the confusion matrix
cm = confusion_matrix(y_test,fun(y_pred))

#Plot the confusion matrix.
sns.set(font_scale=1.4)
sns.heatmap(cm,
            annot=True,
            fmt='g')
plt.ylabel('Prediction',fontsize=20)
plt.xlabel('Actual',fontsize=20)
plt.title('Confusion Matrix for NB',fontsize=17)
plt.show()

#plotting all models ghraph in one ghraph
plt.figure()

# Add the models to the list that you want to view on the ROC plot
models = [
{
    'label': 'Logistic Regression',
    'model': LogisticRegression(),
},
{
    'label': 'Gradient Boosting',
    'model': GradientBoostingClassifier(),
},
{
    'label': 'DecisionTreeClassifier',
    'model': DecisionTreeClassifier(),
},
{
    'label': 'RandomForestClassifier',
    'model': RandomForestClassifier(),
},
{
    'label': 'NaiveBayess',
    'model': MultinomialNB()
}
]

# Below for loop iterates through your models listd
for m in models:
    model = m['model'] # select the model
    model.fit(X_train_smote, y_train_smote) # train the model
    y_pred=model.predict(X_test) # predict the test data
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = metrics.roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
    auc = metrics.roc_auc_score(y_test,model.predict(X_test))
# Now, plot the computed values
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('ROC for 5-ML Models')
plt.legend(loc="lower right")
plt.show() # Display



# initialize a dictionary to hold all metrics
metrics_summary = {}

# helper function to calculate key metrics for each model
def evaluate_model(name, y_true, y_pred, y_proba=None):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_true, y_proba[:, 1]) if y_proba is not None else None
    
    metrics_summary[name] = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "auc": round(float(auc), 4) if auc is not None else None,
        "confusion_matrix": cm.tolist()
    }

# evaluate all trained models
evaluate_model("Logistic Regression", y_test, fun(reg.predict(X_test)), reg.predict_proba(X_test))
evaluate_model("Decision Tree", y_test, fun(DT.predict(X_test)), DT.predict_proba(X_test))
evaluate_model("Gradient Boosting", y_test, fun(GB.predict(X_test)), GB.predict_proba(X_test))
evaluate_model("Random Forest", y_test, fun(RF.predict(X_test)), RF.predict_proba(X_test))
evaluate_model("Naive Bayes", y_test, fun(model1.predict(X_test)), model1.predict_proba(X_test))

# save metrics to a JSON file
with open("metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)

print("\n✅ All model metrics saved to 'metrics.json'")
