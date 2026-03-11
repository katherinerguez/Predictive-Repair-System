from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, hamming_loss, f1_score, precision_score, recall_score, make_scorer
from sklearn.preprocessing import StandardScaler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import GridSearchCV

import pandas as pd
import joblib

with open('data/processed/train.csv', 'r') as f:
    data_train=pd.read_csv(f)
with open('data/processed/validation.csv', 'r') as f:
    data_val=pd.read_csv(f)
with open('data/processed/test.csv', 'r') as f:
    data_test=pd.read_csv(f)
    
labels = ["TWF", "HDF", "PWF", "OSF", "RNF"]
X_train=data_train[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]']]
Y_train=data_train[labels]
X_test=data_test[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]']]
Y_test=data_test[labels]
X_val=data_val[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]', 'Tool wear [min]']]
Y_val=data_val[labels]

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_val_s= scaler.transform(X_val)

#modelo de prediccion 
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20],
    'min_samples_leaf': [1, 3],
    'class_weight': ['balanced', 'balanced_subsample']
}

cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'f1_micro': 'f1_micro',
    'f1_macro': 'f1_macro',
    'hamming': make_scorer(hamming_loss, greater_is_better=False)
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1_micro',      
    cv=cv,                    
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_s, Y_train)

best = grid.best_estimator_
Y_pred = best.predict(X_test_s)

cv_results_df = pd.DataFrame(grid.cv_results_)
cv_results_df.to_csv("outputs/models/rf_results.csv", index=False)

# Métricas
print("Mejores parámetros:", grid.best_params_)
print(classification_report(Y_test, Y_pred, target_names=labels))
print("Hamming loss:", hamming_loss(Y_test, Y_pred))
print("F1 micro:", f1_score(Y_test, Y_pred, average='micro'))
print("F1 macro:", f1_score(Y_test, Y_pred, average='macro'))

joblib.dump(best, 'outputs/models/random_forest_model.pkl') 