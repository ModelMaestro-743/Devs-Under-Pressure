import pandas as pd
import numpy as np
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import xgboost as xgb

# --------------------------
# Data Preparation
# --------------------------

def load_and_preprocess_data():
    
    mentalhealth_df = pd.read_csv('survey.csv')

    #Get rid of the unnecessary columns
    mentalhealth_df = mentalhealth_df.drop(['comments'], axis= 1)
    mentalhealth_df = mentalhealth_df.drop(['state'], axis= 1)
    mentalhealth_df = mentalhealth_df.drop(['Timestamp'], axis= 1)
    
    #Assign default values for each data type
    defaultInt = 0
    defaultString = 'NaN'
    defaultFloat = 0.0

    # Create lists by data type
    intFeatures = ['Age']
    stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
                 'seek_help']
    floatFeatures = []

    for feature in mentalhealth_df:
        if feature in intFeatures:
           mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultInt)
        elif feature in stringFeatures:
           mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultString)
        elif feature in floatFeatures:
           mentalhealth_df[feature] = mentalhealth_df[feature].fillna(defaultFloat)
        else:
           print('Error: Feature %s not recognized.' % feature)


    #Creating gender groups for each gender type. Then replacing the values with an assigned default for each gender. As well as removing invalid values
    gender = mentalhealth_df['Gender'].str.lower() #lower case all column's elements

    gender = mentalhealth_df['Gender'].unique() #Select unique elements

    # Making gender groups
    male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
    trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
    female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

    for (row, col) in mentalhealth_df.iterrows():

        if str.lower(col.Gender) in male_str:
            mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

        if str.lower(col.Gender) in female_str:
            mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

        if str.lower(col.Gender) in trans_str:
            mentalhealth_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

    # Removing invalid values
    stk_list = ['A little about you', 'p']
    mentalhealth_df = mentalhealth_df[~mentalhealth_df['Gender'].isin(stk_list)]

    # Replace missing age values with the median
    mentalhealth_df['Age'] = mentalhealth_df['Age'].fillna(mentalhealth_df['Age'].median())

    mentalhealth_df.loc[mentalhealth_df['Age'] < 18, 'Age'] = mentalhealth_df['Age'].median()
    mentalhealth_df.loc[mentalhealth_df['Age'] > 120, 'Age'] = mentalhealth_df['Age'].median()

    # Define Age Ranges
    mentalhealth_df['age_range'] = pd.cut(
        mentalhealth_df['Age'], bins=[0, 20, 30, 65, 100], 
        labels=["0-20", "21-30", "31-65", "66-100"], 
        include_lowest=True
    )

    #There are almost negilgable amount of entries of self employed so change NaN to NOT self_employed
    #Replace "NaN" string from defaultString
    mentalhealth_df['self_employed'] = mentalhealth_df['self_employed'].replace([defaultString], 'No')
   

    #There are only 0.2% of self work_interfere so change NaN to "Don't know
    #Replace "NaN" string from defaultString

    mentalhealth_df['work_interfere'] = mentalhealth_df['work_interfere'].replace([defaultString], 'Don\'t know' )
   


   # Encoding categorical features
    label_encoders = {}  

    for col in mentalhealth_df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        mentalhealth_df[col] = le.fit_transform(mentalhealth_df[col])
        label_encoders[col] = le 

    #Remove 'Country' attribute
    mentalhealth_df = mentalhealth_df.drop(['Country'], axis= 1)
    

    # Feature Selection
    features = ['Age', 'Gender', 'family_history', 'work_interfere', 'remote_work',
                'benefits', 'care_options', 'wellness_program', 'seek_help',
                'anonymity', 'leave', 'mental_health_consequence']

    target = 'treatment'


    X = mentalhealth_df[features]
    y = mentalhealth_df[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoders, features

# --------------------------
# Model Training
# --------------------------

def train_and_evaluate_model():
    X_train, X_test, y_train, y_test, label_encoders, features = load_and_preprocess_data()

    # --------------------------
    # Hyperparameter Tuning: RandomForest
    # --------------------------
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_  

    # --------------------------
    # Hyperparameter Tuning: XGBoost
    # --------------------------
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_grid = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=42), 
                            xgb_param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    xgb_grid.fit(X_train, y_train)

    best_xgb = xgb_grid.best_estimator_ 

    # --------------------------
    # Evaluate RandomForest
    # --------------------------
    rf_y_pred = best_rf.predict(X_test)
    rf_y_proba = best_rf.predict_proba(X_test)[:, 1]

    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    rf_precision = precision_score(y_test, rf_y_pred)
    rf_recall = recall_score(y_test, rf_y_pred)
    rf_f1 = f1_score(y_test, rf_y_pred)
    rf_roc_auc = roc_auc_score(y_test, rf_y_proba)

    print("\n🚀 **Random Forest Model Performance (Tuned)** 🚀")
    print(f"Accuracy: {rf_accuracy:.2f}")
    print(f"Precision: {rf_precision:.2f}")
    print(f"Recall: {rf_recall:.2f}")
    print(f"F1-score: {rf_f1:.2f}")
    print(f"ROC-AUC: {rf_roc_auc:.2f}")

    # --------------------------
    # Evaluate XGBoost
    # --------------------------
    xgb_y_pred = best_xgb.predict(X_test)
    xgb_y_proba = best_xgb.predict_proba(X_test)[:, 1]

    xgb_accuracy = accuracy_score(y_test, xgb_y_pred)
    xgb_precision = precision_score(y_test, xgb_y_pred)
    xgb_recall = recall_score(y_test, xgb_y_pred)
    xgb_f1 = f1_score(y_test, xgb_y_pred)
    xgb_roc_auc = roc_auc_score(y_test, xgb_y_proba)

    print("\n🚀 **XGBoost Model Performance (Tuned)** 🚀")
    print(f"Accuracy: {xgb_accuracy:.2f}")
    print(f"Precision: {xgb_precision:.2f}")
    print(f"Recall: {xgb_recall:.2f}")
    print(f"F1-score: {xgb_f1:.2f}")
    print(f"ROC-AUC: {xgb_roc_auc:.2f}")

    
    joblib.dump(best_rf, 'mental_health_rf_model.pkl')
    joblib.dump(best_xgb, 'mental_health_xgb_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    return best_rf, best_xgb, X_test, y_test, features

# --------------------------
# SHAP Model Explanation
# -------------------------

def explain_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure()  
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)  # Disable automatic display

    fig.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    plt.close(fig)  

# --------------------------
# Main Execution
# --------------------------

if __name__ == "__main__":
    # Train model and generate SHAP explanation
    best_rf, best_xgb, X_test, y_test, features = train_and_evaluate_model()

    explain_model(best_rf, X_test)
    
    explain_model(best_xgb, X_test)
