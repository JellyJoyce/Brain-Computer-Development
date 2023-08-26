import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.feature_extraction import EfficientFCParameters
from multiprocessing import Pool
from sklearn.feature_selection import RFE
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import tree
import time



def main():
    # Load extracted features from a pickle file
    with open('extracted_features.pkl', 'rb') as f:
        features_df = pickle.load(f)

    # Save extracted features to a CSV file
    features_df.to_csv('extracted_features.csv', index=False)

    # Fill NaN and Inf values with the mean else ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
    features_df[:] = np.nan_to_num(features_df)

    features_df = features_df.drop('value__query_similarity_count__query_None__threshold_0.0', axis = 1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_df.drop('label', axis=1), features_df['label'], test_size=0.2, random_state=42)

    # Create a random forest classifier to use for feature selection
    rfe_method = RFE(
        RandomForestClassifier(n_estimators=20, random_state=10),
        n_features_to_select=8,
        step=2,
    )

    rfe_method.fit(X_train, y_train)

    # Print the names of the most important features
    important_features = X_train.columns[(rfe_method.get_support())]
    important_features = [ important_features[0], important_features[1], important_features[3], 
                          important_features[4],important_features[6],important_features[7]]
    important_features = sorted(important_features)
    print(important_features)
    
    X = features_df.loc[:,important_features].values
    y = features_df['label'].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

    kf = KFold(n_splits=5, shuffle=True, random_state=60)

    accuracy_test = []
    accuracy_train = []

    print(time.ctime())

    # 设定参数网格
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],  # 最大深度
        'ccp_alpha': [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1],  # 最小代价复杂性参数
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 创建决策树模型
    dtree = DecisionTreeClassifier(random_state=22)

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, scoring='accuracy')

    # 在训练集上执行网格搜索
    grid_search.fit(X_train, y_train)

    # 打印最优参数和最优分数
    print("DT Best Parameters: ", grid_search.best_params_)
    print("DT Best Score: ", grid_search.best_score_)

    # 使用最优参数创建新模型
    dtree = grid_search.best_estimator_

    feature_importances = dtree.feature_importances_
    for name, importance in zip(important_features, feature_importances):
        print("Feature: {}, Importance: {}".format(name, importance))

    bagging_dtree = BaggingClassifier(estimator=dtree, n_estimators=10, random_state=0)
    bagging_dtree.fit(X_train, y_train)

    feature_importances = dtree.feature_importances_
    for name, importance in zip(important_features, feature_importances):
        print("Feature: {}, Importance: {}".format(name, importance))

    print(time.ctime())

    # SVM and KNN accuracies lists
    accuracy_test_svm = []
    accuracy_train_svm = []
    accuracy_test_knn = []
    accuracy_train_knn = []
    accuracy_test_rf = []
    accuracy_train_rf = []

    # SVM and KNN models
    svm = SVC(kernel='linear', random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    print(time.ctime())

    # RandomForest model
    param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)

    rf = grid_search.best_estimator_
    bagging_rf = BaggingClassifier(estimator=rf, n_estimators=10, random_state=0)
    bagging_rf.fit(X_train, y_train)

    print(time.ctime())

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(time.time())

        #Train and evaluate DecisionTree
        bagging_dtree.fit(X_train,y_train)
        y_pred = bagging_dtree.predict(X_test)
        # dtree.fit(X_train,y_train)
        # y_pred = dtree.predict(X_test)
        accuracy_test.append(accuracy_score(y_test, y_pred))
        accuracy_train.append(accuracy_score(y_true = y_train, y_pred = bagging_dtree.predict(X_train)))
        
        print(time.time())

        # Train and evaluate SVM
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        accuracy_test_svm.append(accuracy_score(y_test, y_pred_svm))
        accuracy_train_svm.append(accuracy_score(y_true = y_train, y_pred = svm.predict(X_train)))

        # Train and evaluate KNN
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_test_knn.append(accuracy_score(y_test, y_pred_knn))
        accuracy_train_knn.append(accuracy_score(y_true = y_train, y_pred = knn.predict(X_train)))

        print(time.time())

        bagging_rf.fit(X_train, y_train)
        y_pred_bagging_rf = bagging_rf.predict(X_test)
        accuracy_test_rf.append(accuracy_score(y_test, y_pred_bagging_rf))
        accuracy_train_rf.append(accuracy_score(y_true = y_train, y_pred = bagging_rf.predict(X_train)))

        print(time.time())


    print("Decision Tree train data accuracy:",np.mean(accuracy_train))
    print("Decision Tree test data accuracy:",np.mean(accuracy_test))
    print("SVM train data accuracy:",np.mean(accuracy_train_svm))
    print("SVM test data accuracy:",np.mean(accuracy_test_svm))
    print("KNN train data accuracy:",np.mean(accuracy_train_knn))
    print("KNN test data accuracy:",np.mean(accuracy_test_knn))
    print("Random Forest train data accuracy:", np.mean(accuracy_train_rf))
    print("Random Forest test data accuracy:", np.mean(accuracy_test_rf))


    # # Train and evaluate DecisionTree
    # bagging_dtree.fit(X_train,y_train)
    # y_pred = bagging_dtree.predict(X_test)
    # print('Confusion matrix for Decision Tree:\n', confusion_matrix(y_test, y_pred))

    # # Train and evaluate SVM
    # svm.fit(X_train, y_train)
    # y_pred_svm = svm.predict(X_test)
    # print('Confusion matrix for SVM:\n', confusion_matrix(y_test, y_pred_svm))

    # # Train and evaluate KNN
    # knn.fit(X_train, y_train)
    # y_pred_knn = knn.predict(X_test)
    # print('Confusion matrix for KNN:\n', confusion_matrix(y_test, y_pred_knn))

    import matplotlib.pyplot as plt
   # assuming accuracy_test, accuracy_test_svm, and accuracy_test_knn are lists containing your accuracy scores
    data_to_plot = [accuracy_test, accuracy_test_svm, accuracy_test_knn, accuracy_test_rf]

    plt.figure(figsize=(10, 8))
    plt.boxplot(data_to_plot)

    plt.title('Accuracy Scores Across 5-Fold Cross Validation')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks([1, 2, 3, 4], ['Decision Tree', 'SVM', 'KNN', 'Random Forest'])

    plt.show()





if __name__ == "__main__":
    main()
