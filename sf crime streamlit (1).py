import numpy as np
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


train = pd.read_csv('/media/rasa/207A54047A53D55E/E into G/dars/karshenasi arshad/term 3/machine learning/project/train.csv.zip', parse_dates=['Dates'])

st.write("""# San Francisco Crime Classification
## Predict the category of crimes that occurred in the city by the bay""")

st.sidebar.header('User Input Parameters')

# algorithms
st.write("""### **Algorithms**""")
st.write("""| **Algorithm** | **Parameters** |                                            
| :-----------:   |   :----------: |
| Logistic regression  | penalty= 'l1', C= 1, solver= 'saga', multi_class= 'ovr', max_iter= 1 |
| Logistic regression  | penalty= 'l1', C= 1, solver= 'saga', multi_class= 'multinomial', max_iter= 1 |
| Logistic regression  | penalty= 'l2', C= 1, solver= 'lbfgs', multi_class= 'ovr', max_iter= 1 |
| Logistic regression  | penalty= 'l2', C= 1, solver= 'lbfgs, multi_class= 'multinomial', max_iter= 1|
| SVC           | C= 1.0, gamma= 0.1, kernel= 'rbf', max_iter= 1, probability= True |
| Decision Tree | criterion= 'gini', max_depth= 3, random_state= 42 |
| Random Forest | n_estimators= 1, criterion= 'gini', random_state= 42|
| AdaBoost | base_estimator= None, n_estimators= 3 |
| Gradient Boost | learning_rate= 0.1, n_estimators= 1, max_depth= 3, random_state= 42 |
| XGBoost       | n_estimators= 1, criterion= 'gini',learning_rate= 0.1, max_depth= 3 ,gamma= 10, reg_lambda= 1 , objective= 'multi:softmax', random_state= 42|
"""
)

# """# **Data Preprocessing**"""

# drop duplicates datas
train.drop_duplicates(inplace=True)

# replace outlier coordinates with their mean
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
imp = SimpleImputer(strategy='mean')
for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])


labelEncoderPdDistrict = LabelEncoder()
labelEncoderDayOfWeek = LabelEncoder()
labelEncoderAddress = LabelEncoder()

# split date and time
def split_date(data):
    data['Date'] = data['Dates'].dt.date

    # """# **PdDistrict encoding**"""
    data['PdDistrict'] = labelEncoderPdDistrict.fit_transform(data['PdDistrict'])

    # """# **DayOfWeek encoding**"""
    data['DayOfWeek'] = labelEncoderDayOfWeek.fit_transform(data['DayOfWeek'])

    # """# **Address encoding**"""
    data['Block'] = data['Address'].str.contains('block', case=False)
    data['ST'] = data['Address'].str.contains('ST', case=False)
    data['Address'] = labelEncoderAddress.fit_transform(data['Address'])

    data["X_Y"] = data["X"] - data["Y"]
    data["XY"] = data["X"] + data["Y"]

    data['Year'] = data['Dates'].dt.year
    data['Month'] = data['Dates'].dt.month
    data['Day'] = data['Dates'].dt.day

    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Second'] = data['Dates'].dt.second

    data['n_days'] = (data['Date'] - data['Date'].min()).apply(lambda x: x.days)

# """# **Category encoding**"""
labelEncoderCategory = LabelEncoder()
target = labelEncoderCategory.fit_transform(train['Category'])

split_date(train)
# split_date(test)

PdDistrict = st.sidebar.selectbox('PdDistrict', labelEncoderPdDistrict.inverse_transform(train['PdDistrict'].unique()))
DayOfWeek = st.sidebar.selectbox('DayOfWeek', labelEncoderDayOfWeek.inverse_transform(np.sort(train['DayOfWeek'].unique())))
Address = st.sidebar.selectbox('Address', labelEncoderAddress.inverse_transform(train['Address'].unique()))
Longitude = st.sidebar.slider('X Longitude', np.float(train['X'].min()), np.float(train['X'].max()), np.float(train['X'].mean()))
latitude = st.sidebar.slider('Y Longitude', np.float(train['Y'].min()), np.float(train['Y'].max()), np.float(train['Y'].mean()))
BlockOrStreet = st.sidebar.checkbox('In block')
Date = st.sidebar.date_input("Date", train['Dates'].min(), train['Dates'].min(), train['Dates'].max())
Time = st.sidebar.time_input("Time", train['Dates'].dt.time.values[0])
n_days = (Date - train['Date'].min()).days

X_Y = Longitude - latitude
XY = Longitude + latitude

def user_input_features():
    data = {'DayOfWeek': labelEncoderDayOfWeek.transform(np.array([DayOfWeek]).reshape((1))),
            'PdDistrict': labelEncoderPdDistrict.transform(np.array([PdDistrict]).reshape((1))),
            'Address': labelEncoderAddress.transform(np.array([Address]).reshape((1))),
            'X': Longitude,
            'Y': latitude,
            'Block': BlockOrStreet,
            'ST': not(BlockOrStreet),
            'X_Y': X_Y,
            'XY': XY,
            'Year': Date.year,
            'Month': Date.month,
            'Day': Date.day,
            'Hour': Time.hour,
            'Minute': Time.minute,
            'Second': Time.second,
            'n_days': n_days,
            }
    features = pd.DataFrame(data, index=[0])
    return features

X_test = user_input_features()
st.write("## User input parameters")
st.write(X_test)

# """# **Drop colums**"""
train.drop(['Dates','Date','Descript','Resolution', 'Category'], axis='columns', inplace=True)

X_train = train
y_train = target

# ---------------------------------->logistic regression l1 ovr<----------------------------------
logisticRegressionl1ovrModel = LogisticRegression(penalty= 'l1', C= 1, solver= 'saga', multi_class= 'ovr', max_iter=1 )
st.write("## **LogisticRegression(penalty= 'l1', C= 1, solver= 'saga', multi_class= 'ovr', max_iter= 1)**")
logisticRegressionl1ovrModel.fit(X_train, y_train)
st.write("#### Predict")
logisticRegressionl1ovrModelPredict = logisticRegressionl1ovrModel.predict(X_test).reshape((1))
logisticRegressionl1ovrModelPredict = labelEncoderCategory.inverse_transform(logisticRegressionl1ovrModelPredict)
st.write(pd.DataFrame(logisticRegressionl1ovrModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
logisticRegressionl1ovrModelPredictProba = logisticRegressionl1ovrModel.predict_proba(X_test)
st.write(pd.DataFrame(logisticRegressionl1ovrModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->logistic regression l1 multinomial<----------------------------------
st.write("## **LogisticRegression(penalty= 'l1', C= 1, solver= 'saga', multi_class= 'multinomial', max_iter= 1)**")
logisticRegressionl1multinomialModel = LogisticRegression(penalty= 'l1', C= 1, solver= 'saga', multi_class= 'multinomial', max_iter= 1)
logisticRegressionl1multinomialModel.fit(X_train, y_train)
st.write("#### Predict")
logisticRegressionl1multinomialModelPredict = logisticRegressionl1multinomialModel.predict(X_test).reshape((1))
logisticRegressionl1multinomialModelPredict = labelEncoderCategory.inverse_transform(logisticRegressionl1multinomialModelPredict)
st.write(pd.DataFrame(logisticRegressionl1multinomialModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
logisticRegressionl1multinomialModelPredictProba = logisticRegressionl1multinomialModel.predict_proba(X_test)
st.write(pd.DataFrame(logisticRegressionl1multinomialModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->logistic regression l2 ovr<----------------------------------
st.write("## **LogisticRegression(penalty= 'l2', C= 1, solver= 'lbfgs', multi_class= 'ovr', max_iter= 1)**")
logisticRegressionl2ovrModel = LogisticRegression(penalty= 'l2', C= 1, solver= 'lbfgs', multi_class= 'ovr', max_iter= 1)
logisticRegressionl2ovrModel.fit(X_train, y_train)
st.write("#### Predict")
logisticRegressionl2ovrModelPredict = logisticRegressionl2ovrModel.predict(X_test).reshape((1))
logisticRegressionl2ovrModelPredict = labelEncoderCategory.inverse_transform(logisticRegressionl2ovrModelPredict)
st.write(pd.DataFrame(logisticRegressionl2ovrModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
logisticRegressionl2ovrModelPredictProba = logisticRegressionl2ovrModel.predict_proba(X_test)
st.write(pd.DataFrame(logisticRegressionl2ovrModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->logistic regression l2 multinomial<----------------------------------
st.write("## **LogisticRegression(penalty= 'l2', C= 1, solver= 'lbfgs', multi_class= 'multinomial', max_iter= 1)**")
logisticRegressionl2multinomialModel = LogisticRegression(penalty= 'l2', C= 1, solver= 'lbfgs', multi_class= 'multinomial', max_iter= 1)
logisticRegressionl2multinomialModel.fit(X_train, y_train)
st.write("#### Predict")
logisticRegressionl2multinomialModelPredict = logisticRegressionl2multinomialModel.predict(X_test).reshape((1))
logisticRegressionl2multinomialModelPredict = labelEncoderCategory.inverse_transform(logisticRegressionl2multinomialModelPredict)
st.write(pd.DataFrame(logisticRegressionl2multinomialModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
logisticRegressionl2multinomialModelPredictProba = logisticRegressionl2multinomialModel.predict_proba(X_test)
st.write(pd.DataFrame(logisticRegressionl2multinomialModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->support vector machine<----------------------------------
st.write("## **SVC(C= 1.0, gamma= 0.1, kernel= 'rbf', max_iter= 1, probability= True)**")
supportVectorMachineModel = SVC(C= 1.0, gamma= 0.1, kernel= 'rbf', max_iter= 1, probability= True)
supportVectorMachineModel.fit(X_train, y_train)
st.write("#### Predict")
supportVectorMachineModelPredict = supportVectorMachineModel.predict(X_test).reshape((1))
supportVectorMachineModelPredict = labelEncoderCategory.inverse_transform(supportVectorMachineModelPredict)
st.write(pd.DataFrame(supportVectorMachineModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
supportVectorMachineModelPredictProba = supportVectorMachineModel.predict_proba(X_test)
st.write(pd.DataFrame(supportVectorMachineModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->decision tree<----------------------------------
st.write("## **DecisionTreeClassifier(criterion='gini', max_depth= 3, random_state=42)**")
decisionTreeClassifierModel = DecisionTreeClassifier(criterion='gini', max_depth= 3, random_state=42)
decisionTreeClassifierModel.fit(X_train, y_train)
st.write("#### Predict")
decisionTreeClassifierModelPredict = decisionTreeClassifierModel.predict(X_test).reshape((1))
decisionTreeClassifierModelPredict = labelEncoderCategory.inverse_transform(decisionTreeClassifierModelPredict)
st.write(pd.DataFrame(decisionTreeClassifierModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
decisionTreeClassifierModelPredictProba = decisionTreeClassifierModel.predict_proba(X_test)
st.write(pd.DataFrame(decisionTreeClassifierModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->random forest<----------------------------------
st.write("## **RandomForestClassifier(n_estimators= 1, criterion='gini', random_state=42)**")
randomForestClassifierModel = RandomForestClassifier(n_estimators= 1, criterion= 'gini', random_state= 42)
randomForestClassifierModel.fit(X_train, y_train)
st.write("#### Predict")
randomForestClassifierModelPredict = randomForestClassifierModel.predict(X_test).reshape((1))
randomForestClassifierModelPredict = labelEncoderCategory.inverse_transform(randomForestClassifierModelPredict)
st.write(pd.DataFrame(randomForestClassifierModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
randomForestClassifierModelPredictProba = randomForestClassifierModel.predict_proba(X_test)
st.write(pd.DataFrame(randomForestClassifierModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->adaboost<----------------------------------
st.write("## **AdaBoostClassifier(base_estimator= None, n_estimators= 3)**")
adaBoostClassifierModel = AdaBoostClassifier(base_estimator= None, n_estimators= 3)
adaBoostClassifierModel.fit(X_train, y_train)
st.write("#### Predict")
adaBoostClassifierModelPredict = adaBoostClassifierModel.predict(X_test).reshape((1))
adaBoostClassifierModelPredict = labelEncoderCategory.inverse_transform(adaBoostClassifierModelPredict)
st.write(pd.DataFrame(adaBoostClassifierModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
adaBoostClassifierModelPredictProba = adaBoostClassifierModel.predict_proba(X_test)
st.write(pd.DataFrame(adaBoostClassifierModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->gradient boost<----------------------------------
st.write("## **GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 1, max_depth= 3, random_state= 42)**")
gradientBoostingClassifierModel = GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 1, max_depth= 3, random_state= 42)
gradientBoostingClassifierModel.fit(X_train, y_train)
st.write("#### Predict")
gradientBoostingClassifierModelPredict = gradientBoostingClassifierModel.predict(X_test).reshape((1))
gradientBoostingClassifierModelPredict = labelEncoderCategory.inverse_transform(gradientBoostingClassifierModelPredict)
st.write(pd.DataFrame(gradientBoostingClassifierModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
gradientBoostingClassifierModelPredictProba = gradientBoostingClassifierModel.predict_proba(X_test)
st.write(pd.DataFrame(gradientBoostingClassifierModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))

st.write(''' --- ''')

# ---------------------------------->extreme gradient boosting<----------------------------------
st.write("## **XGBClassifier(n_estimators= 1, criterion= 'gini',learning_rate= 0.1, max_depth= 3 ,gamma= 10, reg_lambda= 1 , objective= 'multi:softmax', random_state= 42)**")

XGBClassifierModel = XGBClassifier(n_estimators= 1, criterion= 'gini',learning_rate= 0.1, max_depth= 3 ,gamma= 10, reg_lambda= 1 , objective= 'multi:softmax', random_state= 42)
XGBClassifierModel.fit(X_train, y_train)
st.write("#### Predict")
XGBClassifierModelPredict = XGBClassifierModel.predict(X_test).reshape((1))
# XGBClassifierModelPredict = XGBClassifierModel.predict(X_test)
XGBClassifierModelPredict = labelEncoderCategory.inverse_transform(XGBClassifierModelPredict)
st.write(pd.DataFrame(XGBClassifierModelPredict, columns= ["Crime"]))
st.write("#### Predict probablities")
XGBClassifierModelPredictProba = XGBClassifierModel.predict_proba(X_test)
st.write(pd.DataFrame(XGBClassifierModelPredictProba, columns= labelEncoderCategory.inverse_transform(np.arange(39))))


st.write(
    '''
    # **Sources:**

    1.   https://www.kaggle.com/c/sf-crime
    2.   https://docs.streamlit.io/en/stable/api.html
    3.   https://scikit-learn.org/stable/modules/classes.html
    4.   https://www.kaggle.com/yannisp/sf-crime-analysis-prediction
    5.   https://www.kaggle.com/sjun4530/sf-crime-classification-hyper/
    6.   https://xgboost.readthedocs.io/en/latest/python/python_api.html
    7.   https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    8.   https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
    9.   https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    10.  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
    11.  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    12.  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    13.  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    '''
)