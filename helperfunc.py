import pandas as pd
import numpy as np
import os
from os.path import join as pjoin

# from utils import is_number

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import SMOTE
#Impute Libraries
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as MICE


from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report,confusion_matrix
#Import SVM
from sklearn.svm import SVC
#Import library for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from math import *
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier #RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, LeaveOneOut
from math import *


import sys
#import sklearn.neighbors._base
#sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

#from missingpy import MissForest
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
from scipy import interp


import sweetviz as sv

import math
import numpy as np
from xgboost import XGBClassifier
import catboost as cb
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier
from sklearn.preprocessing import StandardScaler



def missing_imputaion(x,imputer='none'):
    xt=x
    if imputer=='knn':

        X = xt
        imputer = KNNImputer(n_neighbors=2, weights="uniform")
        Knn_data=imputer.fit_transform(X)
        X1=pd.DataFrame(Knn_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='mice':
        Mice_data=MICE().fit_transform(xt)
        X1=pd.DataFrame(Mice_data)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    elif imputer=='randomforest':
        imputer = MissForest()
        Rf = imputer.fit_transform(xt)
        X1=pd.DataFrame(Rf)
        y1=list(xt.columns.values)
        X1.columns=y1
        return X1
    else:
        X1=xt.dropna(axis=0)
        return(X1)








def get_leave_one_out_cv(data_input_local, label_input_local, resample_condition_input_local=True, return_type='index'):
    smote_object_local = SMOTE()
    index_local = data_input_local.columns
    #data_input_local = data_input_local.values
    data_input_local = StandardScaler().fit_transform(data_input_local)
    data_input_local = np.array(data_input_local)
    label_input_local = np.array(label_input_local)

    fold_object_local = LeaveOneOut()
    fold_object_local.get_n_splits(data_input_local)

    if return_type=='index':
        fold_num_local = 0
        fold_index_dict_local = {}
        for train_index_local, test_index_local in fold_object_local.split(data_input_local, label_input_local):
            train_index_local, val_index_local = train_test_split(train_index_local, test_size=0.2)
            fold_index_dict_local[fold_num_local] = [train_index_local, val_index_local, test_index_local]
            fold_num_local = fold_num_local + 1

        return fold_index_dict_local

    if return_type == 'data':
        smote_object_local = SMOTE()

        data_train_list_local = []
        label_train_list_local = []
        data_test_list_local = []
        label_test_list_local = []
        for train_index_local, test_index_local in fold_object_local.split(data_input_local, label_input_local):
            each_data_train_list_local = data_input_local[train_index_local]
            each_label_train_list_local = label_input_local[train_index_local]

            each_data_test_list_local = data_input_local[test_index_local]
            each_label_test_list_local = label_input_local[test_index_local]

            if resample_condition_input_local == True:
                each_data_train_list_local, each_label_train_list_local = smote_object_local.fit_resample(each_data_train_list_local, each_label_train_list_local)

            data_train_list_local.append(each_data_train_list_local)
            label_train_list_local.append(each_label_train_list_local)
            data_test_list_local.append(each_data_test_list_local)
            label_test_list_local.append(each_label_test_list_local)

        fold_data_dict_local = {'data': (data_train_licomst_local, data_test_list_local, label_train_list_local, label_test_list_local), 'index': index_local}
        return fold_data_dict_local

# def cv_fold_2(df):
#     import numpy as np
#     from sklearn.model_selection import StratifiedKFold
#     from imblearn.over_sampling import SMOTE
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.preprocessing import QuantileTransformer
#     random_seed = 2019

#     # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
#     datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
#     transformer = QuantileTransformer(output_distribution='normal')
#     xx1= transformer.fit_transform(datas)
#     yt = df['label']

#     X=np.array(xx1)

#     y=np.array(yt)
#     cc=datas.columns


#     xtrain = []
#     xtest = []
#     ytrain = []
#     ytest = []

#     for _ in range(5):
#         # Perform the train-test split with the first 50 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=2019, stratify=y)

#         # Append the split data to the lists
#         xtrain.append(X_train)
#         xtest.append(X_test)
#         ytrain.append(y_train)
#         ytest.append(y_test)
#     d={'data':(xtrain,xtest,ytrain,ytest),'index':cc}
#     return d

def cv_fold_2(df):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    random_seed = 2019

    # Shuffle the DataFrame
    # df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    transformer = QuantileTransformer(output_distribution='normal')
    xx1= transformer.fit_transform(datas)
    yt = df['label']

    X=np.array(xx1)

    y=np.array(yt)
    cc=datas.columns


    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

    for _ in range(5):
        # Perform the train-test split with the first 50 samples as the test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=96, random_state=2019, stratify=y)

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)
    d={'data':(xtrain,xtest,ytrain,ytest),'index':cc}
    return d

def cv_fold_sumon(df):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    transformer = QuantileTransformer(output_distribution='normal')
    xx1= transformer.fit_transform(datas)
    yt = df['label']

    X=np.array(xx1)

    y=np.array(yt)
    cc=datas.columns


    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = X[:-1254]
        y_train = y[:-1254]
        X_test = X[-1254:]
        y_test = y[-1254:]

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d




def cv_fold_head(df,head_index):#
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    yt = df['label']
    transformer = QuantileTransformer(output_distribution='normal')
    Xtr_main = datas.tail(len(datas)-int(head_index))
    Ytr_main = yt.tail(len(datas)-int(head_index))
    Xt_main =datas.head(int(head_index))
    Yt_main = yt.head(int(head_index))



    Xtr_main_n= transformer.fit_transform(Xtr_main)
    Xt_main_n = transformer.fit_transform(Xt_main)

    Xtr_main_n_a =np.array(Xtr_main_n)
    Ytr_main_a = np.array(Ytr_main)
    Xt_main_n_a = np.array(Xt_main_n)
    Yt_main_a = np.array(Yt_main)

    cc=datas.columns

    print(Xtr_main)

    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = Xtr_main_n_a
        y_train = Ytr_main_a
        X_test = Xt_main_n_a
        y_test = Yt_main_a

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d






# def cv_foldcsv(start=1,n_splits=20, shuffle=False):
#     import numpy as np
#     from sklearn.model_selection import StratifiedKFold
#     from imblearn.over_sampling import SMOTE
#     from sklearn.preprocessing import StandardScaler
#     import pandas as pd

#     smote = SMOTE()

#     # Assuming you have fold_1_train.csv, fold_1_test.csv, ..., fold_5_train.csv, fold_5_test.csv
#     fold_prefix = "fold_"
#     train_suffix = "_train"
#     test_suffix = "_test"
#     csvpath = "D:/sumon2/Metabolomics/metabolomic_last/maindata/121data"
#     xtrain = []
#     xtest = []
#     ytrain = []
#     ytest = []

#     for fold in range(start, n_splits + 1):
#         train_filename = f'{csvpath}/{fold_prefix}{str(fold)}{train_suffix}.csv'
#         test_filename = f'{csvpath}/{fold_prefix}{str(fold)}{test_suffix}.csv'

#         feature_selection_model =['0.81_130.0507m/z','0.81_84.0447m/z','10.87_249.1085m/z','1.78_63.0440m/z','9.34_349.0774n','10.33_178.1441m/z','10.79_146.1171m/z','10.85_214.1306m/z','1.78_89.0597m/z','1.78_150.0899n','10.34_106.0865m/z','4.73_422.1307m/z','10.87_113.0592n','4.06_449.1629m/z','0.81_106.0632n','7.88_86.0965m/z','8.36_144.0935n','10.89_352.2131n','8.65_211.1376m/z','7.86_551.2283m/z']

# # Select only the features and the target variable ('label')
#         selected_columns = feature_selection_model + ['label']
#         # Load your data using pandas, adjust the read_csv parameters as needed
#         X_train = pd.read_csv(train_filename)
#         X_test = pd.read_csv(test_filename)
#         X_train = X_train[selected_columns]
#         X_test = X_test[selected_columns]
#         # Assuming the target variable is in the last column, adjust this if needed
#         y_train = X_train['label']
#         y_test = X_test['label']

#         # Drop the target variable from features
#         X_train = X_train.drop(columns=['label'])
#         X_test = X_test.drop(columns=['label'])

#         xx1 = StandardScaler().fit_transform(pd.concat([X_train, X_test]))
#         x1, y1 = smote.fit_resample(xx1[:len(X_train)], y_train)

#         xtrain.append(x1)
#         xtest.append(xx1[len(X_train):])
#         ytrain.append(y1)
#         ytest.append(y_test)

#     d = {'data': (xtrain, xtest, ytrain, ytest), 'index': X_train.columns}
#     return d
def cv_foldcsv(n_splits=5, shuffle=False, feature_selection_model=None):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    smote = SMOTE()

    # Assuming you have fold_1_train.csv, fold_1_test.csv, ..., fold_5_train.csv, fold_5_test.csv
    fold_prefix = "fold_"
    train_suffix = "_train"
    test_suffix = "_test"
    csvpath = "D:/sumon2/Metabolomics/metabolomic_last/maindata/121data"
    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

    for fold in range(1, n_splits + 1):
        train_filename = f'{csvpath}/{fold_prefix}{str(fold)}{train_suffix}.csv'
        test_filename = f'{csvpath}/{fold_prefix}{str(fold)}{test_suffix}.csv'

        if feature_selection_model is None:
            raise ValueError("Please provide a feature selection model list.")

        # Load your data using pandas, adjust the read_csv parameters as needed
        X_train = pd.read_csv(train_filename)
        X_test = pd.read_csv(test_filename)

        # Select only the features and the target variable ('label')
        selected_columns = feature_selection_model + ['label']
        X_train = X_train[selected_columns]
        X_test = X_test[selected_columns]

        # Assuming the target variable is in the last column, adjust this if needed
        y_train = X_train['label']
        y_test = X_test['label']

        # Drop the target variable from features
        X_train = X_train.drop(columns=['label'])
        X_test = X_test.drop(columns=['label'])

        xx1 = StandardScaler().fit_transform(pd.concat([X_train, X_test]))
        x1, y1 = smote.fit_resample(xx1[:len(X_train)], y_train)

        xtrain.append(x1)
        xtest.append(xx1[len(X_train):])
        ytrain.append(y1)
        ytest.append(y_test)

    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': X_train.columns}
    return d


def cv_fold_tail(df,tail_index):#
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
    datas = df.drop(columns=['label'])  # Replace 'target_column' with your target variable column name
    yt = df['label']
    transformer = QuantileTransformer(output_distribution='normal')
    Xtr_main = datas.head(len(datas)-int(tail_index))
    Ytr_main = yt.head(len(datas)-int(tail_index))
    Xt_main =datas.tail(int(tail_index))
    Yt_main = yt.tail(int(tail_index))


    Xtr_main_n= transformer.fit_transform(Xtr_main)
    Xt_main_n = transformer.fit_transform(Xt_main)

    Xtr_main_n_a =np.array(Xtr_main_n)
    Ytr_main_a = np.array(Ytr_main)
    Xt_main_n_a = np.array(Xt_main_n)
    Yt_main_a = np.array(Yt_main)

    cc=datas.columns



    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     print(len(X))

    # Assuming cc is the list of indices
    for _ in range(5):
        # Perform the train-test split with the last 1254 samples as the test set
#         X_train, X_test, y_train, y_test = train_test_split(X[:-1254], y[:-1254], test_size=1254, random_state=2019, stratify=y[:-1254])

        # Perform the train-test split with the last 1254 samples as the test set
        X_train = Xtr_main_n_a
        y_train = Ytr_main_a
        X_test = Xt_main_n_a
        y_test = Yt_main_a

        # Create shuffled indices for both training and testing sets
        train_indices = np.arange(len(X_train))
        test_indices = np.arange(len(X_test))
        np.random.seed(2019)  # Set a random seed for reproducibility
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Shuffle the data using the shuffled indices
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]

        # Append the split data to the lists
        xtrain.append(X_train)
        xtest.append(X_test)
        ytrain.append(y_train)
        ytest.append(y_test)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d










def cv_fold_tail_all_fold(df1,df2,df3,df4,df5,tail_index):#
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.model_selection import train_test_split
    random_seed = 2019

    # Shuffle the DataFrame
#     df = df.sample(frac=1, random_state=random_seed)
    datas = df1.drop(columns=['label'])   # Replace 'target_column' with your target variable column name
    def cv_create(df, tail_index):
        transformer = QuantileTransformer(output_distribution='normal')

        datas = df.drop(columns=['label'])  # Replace 'label' with your target variable column name
        yt = df['label']

        Xtr_main = datas.head(len(datas) - int(tail_index))
        Ytr_main = yt.head(len(datas) - int(tail_index))
        Xt_main = datas.tail(int(tail_index))
        Yt_main = yt.tail(int(tail_index))

        Xtr_main_n = transformer.fit_transform(Xtr_main)
        Xt_main_n = transformer.fit_transform(Xt_main)

        return Xtr_main_n, Ytr_main, Xt_main_n, Yt_main


    cc=datas.columns

    xtrain = []
    xtest = []
    ytrain = []
    ytest = []

#     tail_index = 122  # Replace with your desired tail index

    # Process df1
    Xtr_main_n1, Ytr_main1, Xt_main_n1, Yt_main1 = cv_create(df1, tail_index)
    xtrain.append(Xtr_main_n1)
    ytrain.append(Ytr_main1)
    xtest.append(Xt_main_n1)
    ytest.append(Yt_main1)

    # Process df2
    Xtr_main_n2, Ytr_main2, Xt_main_n2, Yt_main2 = cv_create(df2, tail_index)
    xtrain.append(Xtr_main_n2)
    ytrain.append(Ytr_main2)
    xtest.append(Xt_main_n2)
    ytest.append(Yt_main2)

    # Process df3
    Xtr_main_n3, Ytr_main3, Xt_main_n3, Yt_main3 = cv_create(df3, tail_index)
    xtrain.append(Xtr_main_n3)
    ytrain.append(Ytr_main3)
    xtest.append(Xt_main_n3)
    ytest.append(Yt_main3)

    # Process df4
    Xtr_main_n4, Ytr_main4, Xt_main_n4, Yt_main4 = cv_create(df4, tail_index)
    xtrain.append(Xtr_main_n4)
    ytrain.append(Ytr_main4)
    xtest.append(Xt_main_n4)
    ytest.append(Yt_main4)

    # Process df5
    Xtr_main_n5, Ytr_main5, Xt_main_n5, Yt_main5 = cv_create(df5, tail_index)
    xtrain.append(Xtr_main_n5)
    ytrain.append(Ytr_main5)
    xtest.append(Xt_main_n5)
    ytest.append(Yt_main5)

    # Create a dictionary to store the data and indices
    d = {'data': (xtrain, xtest, ytrain, ytest), 'index': cc}

    # Return the dictionary
    return d

















def cv_fold(X1,yt,n_splits=5,shuffle=False):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    smote=SMOTE()
    cc=X1.columns
    xx1 = StandardScaler().fit_transform(X1)
    X=np.array(xx1)
    y=np.array(yt)
    skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=shuffle)
    skf.get_n_splits(X, y)
    xtrain=[]
    xtest=[]
    ytrain=[]
    ytest=[]
    for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]
            x1,y1= smote.fit_resample(X_train, y_train)
            xtrain.append(x1)
            xtest.append(X_test)
            ytrain.append(y1)
            ytest.append(y_test)
    d={'data':(xtrain,xtest,ytrain,ytest),'index':cc}
    return d








def feature_selection(x,y, num_of_max_feature_for_genetic_extractor):
        X_train=x
        y_train=y
        mdl=[]
        mdl.append(xgb.XGBClassifier(
                        max_depth=4
                        ,learning_rate=0.2
                        ,reg_lambda=1
                        ,n_estimators=150
                        ,subsample = 0.9
                        ,colsample_bytree = 0.9))
        mdl.append(RandomForestClassifier(n_estimators=50,max_depth=10,
                                            random_state=0,class_weight=None,
                                            n_jobs=-1))
        mdl.append(ExtraTreesClassifier())
        ml1=['XGBoost','Random_Forest','Extra_Tree']
        feat_sel=[]
        for i in range(3):

            model=mdl[i]
            model.fit(X_train, y_train)
            model.feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns
            print("Feature ranking:")
            sel_feat=[]
            for f in range(X_train.shape[1]):
                    print("%d. feature no:%d feature name:%s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
                    sel_feat.append(feat_labels[indices[f]])
            top_n=20
            feat_sel.append(sel_feat)
            indices = indices[0:top_n]
            plt.subplots(figsize=(12, 10))
            g = sns.barplot(x=importances[indices], y=feat_labels[indices], orient='h', label='big') #import_feature.iloc[:Num_f]['col'].values[indices]

            g.set_title(ml1[i]+' feature selection',fontsize=25)
            g.set_xlabel("Relative importance",fontsize=25)
            g.set_ylabel("Features",fontsize=25)
            g.tick_params(labelsize=14)
            sns.despine()
                # plt.savefig('feature_importances_v3.png')
            plt.show()
            print('-----------------------------------------------------------------')
        xgboost=feat_sel[0]
        randomforest=feat_sel[1]
        extratree=feat_sel[2]

        from genetic_selection import GeneticSelectionCV
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier()
        model = GeneticSelectionCV(
            estimator, cv=10, verbose=0,
            scoring="accuracy", max_features=num_of_max_feature_for_genetic_extractor,
            n_population=100, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=50,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.04,
            tournament_size=3, n_gen_no_change=10,
            caching=True, n_jobs=-1)

        model = model.fit(x, y)
        genetic_feature_selection = list(x.columns[model.support_])

        print('****************Genectic Evolution Based Feature Selection****************')
        print("Number of Selected Features : " + str(len(genetic_feature_selection)))
        print("Features : ")
        iter = 1
        for each_feature in genetic_feature_selection:
          print(str(iter) + ". " + each_feature)
          iter=iter+1

        return(xgboost,randomforest,extratree,genetic_feature_selection)


def feature_selection_without_gen(x,y):
        X_train=x
        y_train=y
        mdl=[]
        mdl.append(xgb.XGBClassifier(
                        max_depth=4
                        ,learning_rate=0.2
                        ,reg_lambda=1
                        ,n_estimators=150
                        ,subsample = 0.9
                        ,colsample_bytree = 0.9))
        mdl.append(RandomForestClassifier(n_estimators=50,max_depth=10,
                                            random_state=0,class_weight=None,
                                            n_jobs=-1))
        mdl.append(ExtraTreesClassifier())
        ml1=['XGBoost','Random_Forest','Extra_Tree']
        feat_sel=[]
        for i in range(3):

            model=mdl[i]
            model.fit(X_train, y_train)
            model.feature_importances_
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feat_labels = X_train.columns
            print("Feature ranking:")
            sel_feat=[]
            for f in range(X_train.shape[1]):
                    print("%d. feature no:%d feature name:%s (%f)" % (f+1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
                    sel_feat.append(feat_labels[indices[f]])
            top_n=20
            feat_sel.append(sel_feat)
            indices = indices[0:top_n]
            plt.subplots(figsize=(12, 10))
            g = sns.barplot(importances[indices],feat_labels[indices], orient='h',label='big') #import_feature.iloc[:Num_f]['col'].values[indices]

            g.set_title(ml1[i]+' feature selection',fontsize=25)
            g.set_xlabel("Relative importance",fontsize=25)
            g.set_ylabel("Features",fontsize=25)
            g.tick_params(labelsize=14)
            sns.despine()
                # plt.savefig('feature_importances_v3.png')
            plt.show()
            print('-----------------------------------------------------------------')
        xgboost=feat_sel[0]
        randomforest=feat_sel[1]
        extratree=feat_sel[2]

        return(xgboost,randomforest,extratree)


def feature_genetic_extractor(x,y, num_of_max_feature_for_genetic_extractor, num_of_population=100, num_of_generation=50, num_of_gen_no_change=10):
        X_train=x
        y_train=y
        from genetic_selection import GeneticSelectionCV
        from sklearn.tree import DecisionTreeClassifier

        estimator = DecisionTreeClassifier()
        model = GeneticSelectionCV(
            estimator, cv=10, verbose=0,
            scoring="accuracy", max_features=num_of_max_feature_for_genetic_extractor,
            n_population=num_of_population, crossover_proba=0.5,
            mutation_proba=0.2, n_generations=num_of_generation,
            crossover_independent_proba=0.5,
            mutation_independent_proba=0.04,
            tournament_size=3, n_gen_no_change=num_of_gen_no_change,
            caching=True, n_jobs=-1)

        model = model.fit(x, y)
        genetic_feature_selection = list(x.columns[model.support_])

        print('****************Genectic Evolution Based Feature Selection****************')
        print("Number of Selected Features : " + str(len(genetic_feature_selection)))
        print("Features : ")
        iter = 1
        for each_feature in genetic_feature_selection:
          print(str(iter) + ". " + each_feature)
          iter=iter+1

        return genetic_feature_selection


def models():

        clf=[]
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(13, 13), learning_rate='constant',
               learning_rate_init=0.001, max_iter=500, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=111,
               shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
               verbose=False, warm_start=False)
        clf.append(MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500))


        clf.append(LinearDiscriminantAnalysis())

        clf.append(xgb.XGBClassifier(
                        max_depth=85
                        ,learning_rate=0.9388440565186442,
                        min_split_loss= 0.0
                        ,reg_lambda=5.935581318908179
                        ,min_child_weight= 2.769401581888831
                        ,colsample_bylevel= 0.7878344729848824
                        ,colsample_bynode=0.4895496034538383
                        ,alpha= 7.9692927383000445
                        ,n_estimators=150
                        ,subsample = 0.2656532818978606
                        ,colsample_bytree = 0.8365485367400313))

        clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=10, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0, warm_start=False))


        clf.append(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False))


        clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto',
                kernel='linear', max_iter=100, probability=True, random_state=0,
                shrinking=True, tol=0.001, verbose=False))


        clf.append(ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=0))

        clf.append(AdaBoostClassifier(n_estimators=100, random_state=0))

        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0))

        clf.append(XGBClassifier(n_estimators=400,
                      iterations=500,
                      learning_rate=0.001,
                      loss_function='Logloss'))
        clf.append(cb.CatBoostClassifier())
        clf.append(LGBMClassifier(learning_rate=0.01))
        clf.append(AdaBoostClassifier(learning_rate=0.001))
        clf.append(SVC(probability=True))
        clf.append(RandomForestClassifier())
        clf.append(ExtraTreesClassifier(bootstrap=True))
        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(LinearDiscriminantAnalysis())
        clf.append(LogisticRegression())
        clf.append(LogisticRegression(penalty='elasticnet', l1_ratio=0.01, solver='saga'))
        #clf.append(RidgeClassifier())




        clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier','XGB_untuned', 'CatBoost_untuned', 'LGBM_untuned', 'AdaBoost_untuned', 'SVC_untuned', 'RandomForest_untuned', 'ExtraTrees_untuned', 'KNeighbors_untuned', 'LDA_untuned', 'LogisticRegression_untuned', 'ElasticNet_untuned']
        #, 'Ridge_untuned'
        #Result.to_csv
        return(clf,clff )


def models_v_2():

        clf=[]
        MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
               beta_2=0.999, early_stopping=False, epsilon=1e-08,
               hidden_layer_sizes=(13, 13), learning_rate='constant',
               learning_rate_init=0.001, max_iter=500, momentum=0.9,
               nesterovs_momentum=True, power_t=0.5, random_state=111,
               shuffle=False, solver='adam', tol=0.0001, validation_fraction=0.1,
               verbose=False, warm_start=False)
        clf.append(MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500))


        clf.append(LinearDiscriminantAnalysis())

        clf.append(xgb.XGBClassifier(
                        max_depth=85
                        ,learning_rate=0.9388440565186442,
                        min_split_loss= 0.0
                        ,reg_lambda=5.935581318908179
                        ,min_child_weight= 2.769401581888831
                        ,colsample_bylevel= 0.7878344729848824
                        ,colsample_bynode=0.4895496034538383
                        ,alpha= 7.9692927383000445
                        ,n_estimators=150
                        ,subsample = 0.2656532818978606
                        ,colsample_bytree = 0.8365485367400313))

        clf.append(RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=10, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=100,
                               n_jobs=None, oob_score=False, random_state=0,
                               verbose=0, warm_start=False))


        clf.append(LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                           intercept_scaling=1, l1_ratio=None, max_iter=100,
                           multi_class='multinomial', n_jobs=None, penalty='l2',
                           random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                           warm_start=False))


        clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto',
                kernel='linear', max_iter=100, probability=True, random_state=0,
                shrinking=True, tol=0.001, verbose=False))


        clf.append(ExtraTreesClassifier(n_estimators=100, max_depth=8, min_samples_split=10, random_state=0))

        clf.append(AdaBoostClassifier(n_estimators=100, random_state=0))

        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=10, random_state=0))

        clf.append(XGBClassifier(n_estimators=400,
                      iterations=500,
                      learning_rate=0.001,
                      loss_function='Logloss'))
        clf.append(cb.CatBoostClassifier())
        clf.append(LGBMClassifier(learning_rate=0.01))
        clf.append(AdaBoostClassifier(learning_rate=0.001))
        clf.append(SVC(probability=True))
        clf.append(RandomForestClassifier())
        clf.append(ExtraTreesClassifier(bootstrap=True))
        clf.append(KNeighborsClassifier(n_neighbors=3))
        clf.append(LinearDiscriminantAnalysis())
        clf.append(LogisticRegression())
        clf.append(LogisticRegression(penalty='elasticnet', l1_ratio=0.01, solver='saga'))
        clf.append(RidgeClassifier())




        clff=['MLPClassifier','LinearDiscriminantAnalysis','XGBClassifier','RandomForestClassifier','LogisticRegression','SVM','ExtraTreesClassifier','AdaBoostClassifier','KNeighborsClassifier','GradientBoostingClassifier','XGB_untuned', 'CatBoost_untuned', 'LGBM_untuned', 'AdaBoost_untuned', 'SVC_untuned', 'RandomForest_untuned', 'ExtraTrees_untuned', 'KNeighbors_untuned', 'LDA_untuned', 'LogisticRegression_untuned', 'ElasticNet_untuned', 'Ridge_untuned']

        #
        #Result.to_csv
        return(clf,clff)

def classification_with_top_feature_v_2(data,feature_num,feature_selection_model,classifier,feat_increment):

        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models_v_2()


        if classifier=='all':
            l=0
            for c in range(22):

                clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]  #feature increasing
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(xt1)


                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())



                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
                    # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'




                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)



                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')

                print('---------------------------------------------------------------------')
                l=l+1

            return
        else:

                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0


#                 clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(np.array(xt1))

                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred

                    categories=list(pd.Series(y2).unique())
                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'

                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)


#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')


                return




def classification_with_top_feature(data,feature_num,feature_selection_model,classifier,feat_increment):

        xtrain,xtest,ytrain,ytest=data['data']
        ind=data['index'].to_list()
        num_feat=feature_num
        fsm=feature_selection_model
        feature=fsm[0:num_feat]
        clf,clff=models()


        if classifier=='all':
            l=0
            for c in range(21):

                clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]  #feature increasing
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(xt1)


                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred
                    categories=list(pd.Series(y2).unique())



                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
                    # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'




                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)



                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')

                print('---------------------------------------------------------------------')
                l=l+1

            return
        else:

                clf1=clf[classifier]  #model 0= MLP, 1= LDA, 2 = XGBoost, 3 = RF, 4= Logit, 5=SVC, 6 = Extra tree, 7= Adaboost, 8 = KNN, 9 = GradientBoost
                l=classifier
#             l=0


#                 clf1=clf[c]
                a=[]
                p=[]
                r=[]
                s=[]
                f=[]
                mean_tpr=[]
                mean_auc=[]

                feat=[]
                for i in list(range(0,num_feat,feat_increment)):

                    y_pred=[]
                    y2=[]
                    tl=fsm[0:i+1]
                    tprs = []
                    aucs = []
                    mean_fpr = np.linspace(0,1,100)

                    total_fold_num = len(xtrain)
                    for k in range(total_fold_num):
                        x11=pd.DataFrame(xtrain[k])
                        x11.columns=ind
                        x1=x11[tl]
                        y1=ytrain[k]
                        model = clf1.fit(np.array(x1),np.array(y1))
                        #model = clf1.fit(x[train],y.iloc[train])
                        xts=pd.DataFrame(xtest[k])
                        xts.columns=ind
                        xt1=xts[tl]
                        y_pr=model.predict(np.array(xt1))

                        y_pred.extend(y_pr)
                        y2.extend(ytest[k])



                    y21=y2
                    y_pred1=y_pred

                    categories=list(pd.Series(y2).unique())
                    from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                    # main confusion matrix
                    cm = confusion_matrix(y21, y_pred1)
                    cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                    # Overall Accuracy
                    Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                    Overall_Accuracy = round(Overall_Accuracy*100, 2)
                    # create confusion matrix table (pd.DataFrame)
                    # cm_table = pd.DataFrame(cm, index=categories , columns=categories)
                    if (i+1)!=1:
                      feature_no= 'top_'+str(i+1)+'_features'
                    else:
                      feature_no= 'top_'+str(i+1)+'_feature'

                    try:

                        Eval_Mat = []
                        # per class metricies
                        for i in range(len(categories)):
                            TN = cm_per_class[i][0][0]
                            FP = cm_per_class[i][0][1]
                            FN = cm_per_class[i][1][0]
                            TP = cm_per_class[i][1][1]
                            Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                            Precision = round(100*(TP)/(TP+FP), 2)
                            Sensitivity = round(100*(TP)/(TP+FN), 2)
                            F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                            Specificity = round(100*(TN)/(TN+FP), 2)
                            Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                        # sizes of each class
                        s2 = np.sum(cm,axis=1)
                        # create tmep excel table
                        headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                        temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                        # weighted average of per class metricies
                        ac=Overall_Accuracy
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                        rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                        f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                        sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)


                    except:
                        ac='NaN'
                        # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                        pr = 'NaN'
                        rc ='NaN'
                        f1 = 'NaN'
                        sp = 'NaN'

                    a.append(ac)
                    p.append(pr)
                    r.append(rc)
                    s.append(sp)
                    f.append(f1)
                    feat.append(feature_no)


#                     conf_matrix =confusion_matrix(y2, y_pred)

#                     print('************** ')
#                     print("Top %d  feature" %(i+1))
#                     print('************** ')
#                     print(conf_matrix)
#


                Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
                Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
                Result.index= feat
                #Result.to_csv
                print('---------------------------------------------------------------------')
                print('Result for '+clff[l]+' classifier')
                print('---------------------------------------------------------------------')
                print(Result)
    #             Result.to_csv('/content/'+clff[l]+'_classifier_for_top10_features.csv')
#                 l=l+1
                print('---------------------------------------------------------------------')


                return




def classification_with_combined_features(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        auc_all=[]
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]
        confusion_matrices = [] 
        accuracies = []
        for c in range(21):

            clf1=clf[c]

            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]

                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)
                    accuracy = accuracy_score(ytest[k], y_pr)
                    accuracies.append(accuracy)
                    
                    
                        
          



                categories=list(pd.Series(y2).unique())
                y21, y_pred1=y2,y_pred
                flat_pred_probabilities = [prob[1] for prob in probs]

                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'


                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                # print(f'{c} is {cm}')
                # Calculate confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                confusion_matrices.append({'index': c, 'confusion_matrix': cm})
                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)
                auc1 = roc_auc_score(y2, flat_pred_probabilities)

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0]
                    FP = cm_per_class[i][0][1]
                    FN = cm_per_class[i][1][0]
                    TP = cm_per_class[i][1][1]
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
              
                    Precision = round(100*(TP)/(TP+FP), 2)
                    Sensitivity = round(100*(TP)/(TP+FN), 2)
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                    Specificity = round(100*(TN)/(TN+FP), 2)

                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1)
                # create tmep excel table
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # print(f"Fold {k+1} Accuracy: {accuracies:.4f}")
              
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
                a.append(ac)
                auc_all.append(auc1)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)
          



        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
        Result.index= clff





        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred,confusion_matrices


####



def classification_with_combined_featuresR(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]

        for c in range(21):

            clf1=clf[c]  
          
            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]
            
                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]   
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)





                categories=list(pd.Series(y2).unique()) 
                y21, y_pred1=y2,y_pred
                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'
              

                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)
                

                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)

              

                

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0] 
                    FP = cm_per_class[i][0][1]   
                    FN = cm_per_class[i][1][0]  
                    TP = cm_per_class[i][1][1]  
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                    Precision = round(100*(TP)/(TP+FP), 2)  
                    Sensitivity = round(100*(TP)/(TP+FN), 2) 
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)  
                    Specificity = round(100*(TN)/(TN+FP), 2)  
                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1) 
                # create tmep excel table 
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2) 
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)  
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)  
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)  
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2) 
                a.append(ac)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)


        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score']
        Result.index= clff

        print(Result)
        
        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred
  





####

def classification_with_combined_features_mul(data,feature_num,feature_selection_model,classifier):

    xtrain,xtest,ytrain,ytest=data['data']
    ind=data['index'].to_list()
    num_feat=feature_num
    fsm=feature_selection_model
    # feature=fsm[0:num_feat]
    clf,clff=models()
    classifier='all'

    if classifier=='all':
        l=0
        auc_all=[]
        a=[]
        p=[]
        r=[]
        s=[]
        f=[]
        prb0=[]
        prb1=[]
        pred=[]
        tar=[]

        for c in range(21):

            clf1=clf[c]

            feat=[]
            for i in list(range(1)):

                y_pred=[]
                y2=[]
                tl=fsm[0:num_feat]
                probs=[]
                probss=[]

                total_fold_num = len(xtrain)
                for k in range(total_fold_num):
                    x11=pd.DataFrame(xtrain[k])
                    x11.columns=ind
                    x1=x11[tl]
                    y1=ytrain[k]
                    model = clf1.fit(np.array(x1),np.array(y1))
                    #model = clf1.fit(x[train],y.iloc[train])
                    xts=pd.DataFrame(xtest[k])
                    xts.columns=ind
                    xt1=xts[tl]
                    y_pr=model.predict(np.array(xt1))
                    y_prob=model.predict_proba(np.array(xt1))
                    y_pred.extend(y_pr)
                    y2.extend(ytest[k])
                    probs.extend(y_prob)
                    probss.append(y_prob)




                categories=list(pd.Series(y2).unique())
                y21, y_pred1=y2,y_pred
                flat_pred_probabilities = np.array([prob[1] for prob in probs])
                flat_pred_probabilities = flat_pred_probabilities.reshape((-1, 1))

                if (i+1)!=1:
                  feature_no= 'top_'+str(i+1)+'_features'
                else:
                  feature_no= 'top_'+str(i+1)+'_feature'


                from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
                # main confusion matrix
                cm = confusion_matrix(y21, y_pred1)


                cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
                # Overall Accuracy
                Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
                Overall_Accuracy = round(Overall_Accuracy*100, 2)
                auc1 = roc_auc_score(y2, flat_pred_probabilities, multi_class='ovr')

                Eval_Mat = []
                # per class metricies
                for i in range(len(categories)):
                    TN = cm_per_class[i][0][0]
                    FP = cm_per_class[i][0][1]
                    FN = cm_per_class[i][1][0]
                    TP = cm_per_class[i][1][1]
                    Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                    print(f"acc is {Accuracy}")
                    Precision = round(100*(TP)/(TP+FP), 2)
                    Sensitivity = round(100*(TP)/(TP+FN), 2)
                    F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                    Specificity = round(100*(TN)/(TN+FP), 2)

                    Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
                # sizes of each class
                s2 = np.sum(cm,axis=1)
                # create tmep excel table
                headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
                temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
                # weighted average of per class metricies
                ac=Overall_Accuracy
                # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
                pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
                rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
                f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
                sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
                a.append(ac)
                auc_all.append(auc1)
                p.append(pr)
                r.append(rc)
                s.append(sp)
                f.append(f1)
                feat.append(feature_no)
                prb0.append(probs)
                prb1.append(probss)
                pred.append(y_pred1)
                tar.append(y2)


        Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
        Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
        Result.index= clff

        print(Result)



        l=l+1
        print('---------------------------------------------------------------------')
        return  Result, prb1,prb0,ytest,tar,pred




def processed_data(ml1,ml2,ml3,td2):
    xts=[]
    xtr=[]
    yts=[]
    ytr=[]


    prf=[]
    for i in range(5):
      pl=np.concatenate((ml1[i],ml2[i],ml3[i]),1)
      prf.append(pl)

    for j in range(5):
      if j==1:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[2],td2[3],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[2],prf[3],prf[4],prf[0]),0))
      elif j==2:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[4],prf[0]),0))
      elif j==3:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[2],td2[4],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[2],prf[4],prf[0]),0))
      elif j==4:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[2],td2[0]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[2],prf[0]),0))
        
      elif j==0:
        yts.append(td2[j])
        ytr.append(np.concatenate((td2[1],td2[3],td2[2],td2[4]),0))
        xts.append(prf[j])
        xtr.append(np.concatenate((prf[1],prf[3],prf[2],prf[4]),0))
    return xtr,xts,ytr,yts



def stacking_classification(ml1,ml2,ml3,td2):

    xtrain,xtest,ytrain,ytest=processed_data(ml1,ml2,ml3,td2)
    clf,clff=models()
    classifier='all'
    if classifier=='all':
      l=0
      auc_all=[]
      a=[]
      p=[]
      r=[]
      s=[]
      f=[]
      prb0=[]
      prb1=[]
      pred=[]
      tar=[]

      for c in range(21):

          clf1=clf[c]

          feat=[]
          for i in list(range(1)):

              y_pred=[]
              y2=[]
              # tl=fsm[0:num_feat]
              probs=[]
              probss=[]

              total_fold_num = len(xtrain)
              for k in range(total_fold_num):
                  x1=pd.DataFrame(xtrain[k])
                  # x11.columns=ind
                  # x1=x11[tl]
                  y1=ytrain[k]
                  model = clf1.fit(np.array(x1),np.array(y1))
                  #model = clf1.fit(x[train],y.iloc[train])
                  xt1=pd.DataFrame(xtest[k])
                  # xts.columns=ind
                  # xt1=xts[tl]
                  y_pr=model.predict(np.array(xt1))
                  y_prob=model.predict_proba(np.array(xt1))
                  y_pred.extend(y_pr)
                  y2.extend(ytest[k])
                  probs.extend(y_prob)
                  probss.append(y_prob)





              categories=list(pd.Series(y2).unique())
              flat_pred_probabilities = [prob[1] for prob in probs]
              y21, y_pred1=y2,y_pred
              if (i+1)!=1:
                feature_no= 'top_'+str(i+1)+'_features'
              else:
                feature_no= 'top_'+str(i+1)+'_feature'


              from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
              # main confusion matrix
              cm = confusion_matrix(y21, y_pred1)
              print(f'{c} is {cm}')
              # cm_per_class: it returns a 2x2 confusion matrix for each class, where 'i' represnt  class index
              # cm_per_class[i][0][0]:TN,   cm_per_class[i][0][1]:FP,   cm_per_class[i][1][0]:FN,    cm_per_class[i][1][1]:TP
              cm_per_class = multilabel_confusion_matrix(y21, y_pred1)
              # Overall Accuracy
              Overall_Accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
              Overall_Accuracy = round(Overall_Accuracy*100, 2)
              auc1 = roc_auc_score(y2, flat_pred_probabilities)
              Eval_Mat = []
              # per class metricies
              for i in range(len(categories)):
                  TN = cm_per_class[i][0][0]
                  FP = cm_per_class[i][0][1]
                  FN = cm_per_class[i][1][0]
                  TP = cm_per_class[i][1][1]
                  Accuracy = round(100*(TP+TN)/(TP+TN+FP+FN), 2)
                  Precision = round(100*(TP)/(TP+FP), 2)
                  Sensitivity = round(100*(TP)/(TP+FN), 2)
                  F1_score = round((2*Precision*Sensitivity)/(Precision+Sensitivity), 2)
                  Specificity = round(100*(TN)/(TN+FP), 2)
                  Eval_Mat.append([Accuracy, Precision, Sensitivity, F1_score, Specificity])
              # sizes of each class
              s2 = np.sum(cm,axis=1)
              # create tmep excel table
              headers=['Accuracy', 'Precision', 'Sensitivity', 'F1_score', 'Specificity']
              temp_table = pd.DataFrame(Eval_Mat, index=categories ,columns=headers)
              # weighted average of per class metricies
              ac=Overall_Accuracy
              # ac = round(temp_table['Accuracy'].dot(s)/np.sum(s), 2)
              pr = round(temp_table['Precision'].dot(s2)/np.sum(s2), 2)
              rc = round(temp_table['Sensitivity'].dot(s2)/np.sum(s2), 2)
              f1 = round(temp_table['F1_score'].dot(s2)/np.sum(s2), 2)
              sp = round(temp_table['Specificity'].dot(s2)/np.sum(s2), 2)
              a.append(ac)
              auc_all.append(auc1)
              p.append(pr)
              r.append(rc)
              s.append(sp)
              f.append(f1)
              feat.append(feature_no)
              prb0.append(probs)
              prb1.append(probss)
              pred.append(y_pred1)
              tar.append(y2)

      Result=pd.concat([pd.DataFrame(a),pd.DataFrame(p),pd.DataFrame(r),pd.DataFrame(s),pd.DataFrame(f),pd.DataFrame(auc_all)],1)
      Result.columns=['Accuracy','Precision','Recall','Specificity','F1-score','Auc']
      Result.index= clff

      print(Result)

      l=l+1
      print('---------------------------------------------------------------------')
      return  Result, prb1,prb0,ytest,tar,pred