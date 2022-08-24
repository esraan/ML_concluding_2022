
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, silhouette_score, RocCurveDisplay, precision_recall_curve
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFdr
from skfeature.function.similarity_based.reliefF import reliefF
from skfeature.function.information_theoretical_based.MRMR import mrmr
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score as ACC
from sklearn.metrics import matthews_corrcoef as MCC
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import average_precision_score as PR
from sklearn.preprocessing import label_binarize
import sklearn.preprocessing as preprocessing
import sklearn
import os
from sklearn.model_selection import GridSearchCV
from skfeature.function.similarity_based.reliefF import reliefF
from sklearn.feature_selection import SelectKBest
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
import os
import skdim
from imblearn.over_sampling import BorderlineSMOTE
from skfeature.function.information_theoretical_based.MRMR import mrmr
import math
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
# from ML_4 import PreProcessor
from Ml_pipeline_automation import *

def PR_AUC(y_test, y_pred):
    if len(np.unique(y_test)) > 2:  # if it's multiclass data
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_true = label_binarize(y_pred, classes=lb.classes_)
        y_pred = lb.transform(y_pred)
        score = PR(y_true, y_pred, average='micro')
    else:
        score = PR(y_test, y_pred, average='weighted')
    return score


def multiclass_roc_auc_score(y_test, y_pred):
    try:
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return AUC(y_test, y_pred, average="macro")

    except  ValueError:
        return np.nan



def data_prep(path):
    df = pd.read_csv(path, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df.set_index('Unnamed: 0', inplace=True)
    y = df.loc[:, "class"]  # .to_numpy()
    df = df.drop("class", axis=1)
    X = df  # .to_numpy()
    was_filtered = False
    if (X.shape[1] > 1000):
        selector = SelectKBest(k=1000)
        selector.fit(X, y)
        features = X.columns[selector.get_support()]
        X = X[features].copy()
        was_filtered = True
    return X,y, was_filtered


main_path = "/Users/esraan/Desktop/ML_final/df_grid_results/"
original_path = "/Users/esraan/Desktop/ML_final/df_original/"
#
for dataset in os.listdir(main_path):
    if dataset.endswith("csv"):
        dataset_name=dataset[:-4]
        print(dataset_name)
        dataset_name_o = dataset_name.replace("_result", "")
        df_united = pd.read_csv(main_path+dataset_name+".csv",index_col=0)
        best_configure_id = df_united["mean_test_AUC"].idxmax()
        configure = df_united.iloc[best_configure_id]
        k_AUG = configure['param_reduce_dim__k']

        X,y,filtered = data_prep(original_path+dataset_name_o+".csv")
        n = X.shape[0]
        fs_function = configure["param_reduce_dim__score_func"]  # the column in dataset_result that countain the fs score function
        estimator_best = configure["param_clf__estimator"]

        if "reliefF" in fs_function:
            score_func_AUG = reliefF
        elif "mrmr" in fs_function:
            score_func_AUG = mrmr
        elif "RFE" in fs_function:
            score_func_AUG = RFE
        elif "SelectFdr" in fs_function:
            score_func_AUG = SelectFdr
        elif "hybrid_DReductionC" in fs_function:
            score_func_AUG = hybrid_DReductionC
        elif "GBC_new" in fs_function:
            score_func_AUG = GBC_new
        else:
            score_func_AUG = GBC

        estimator_aug = None
        if "GaussianNB" in estimator_best:
            estimator_aug = GaussianNB()
        elif "RandomForestClassifier" in estimator_best:
            estimator_aug = RandomForestClassifier()
        elif "LogisticRegression" in estimator_best:
            estimator_aug = LogisticRegression()
        elif "KNeighborsClassifier" in estimator_best:
            estimator_aug = KNeighborsClassifier()
        elif "SVC" in estimator_best:
            estimator_aug = SVC()

        proccessing_instance = PreProcessor()
        proccessing_instance = proccessing_instance.fit(X)
        X = proccessing_instance.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
        selector_aug = KBest_features(score_func=score_func_AUG, k=k_AUG)
        selector_aug.fit(X_train, y_train)
        X_train = selector_aug.transform(X_train, y_train)
        X_test = selector_aug.transform(X_test)

        ################# transforming the train ##############################

        linear_transformer = KernelPCA(kernel='linear')
        X_transformed_linear = linear_transformer.fit_transform(X_train)
        rbf_transformer = KernelPCA(kernel='rbf', n_components=n)
        X_transformed_rbf = rbf_transformer.fit_transform(X_transformed_linear)
        X_train = X_train.reset_index()
        X_train = pd.concat([pd.DataFrame(X_transformed_rbf).reset_index(), X_train], axis= 1)

        ################# transforming the test ##############################
        X_transformed_linear_test = linear_transformer.transform(X_test)  # only transform
        X_transformed_rbf_test = rbf_transformer.transform(X_transformed_linear_test)
        X_test = X_test.reset_index()# only transform
        X_test = pd.concat([pd.DataFrame(X_transformed_rbf_test).reset_index(), X_test],axis= 1)
        if dataset_name == "Carcinom_result":
            smoote = SMOTE(random_state=42,k_neighbors=4)
        elif dataset_name =="ALL_result":
            smoote = BorderlineSMOTE()
        else:
            smoote = SMOTE(random_state=42)
        x_train_smoote, y_train_smoote = smoote.fit_resample(X_train.drop("index",axis=1), y_train)

        # evaluate model
        cv = StratifiedKFold(n_splits=10, shuffle= True, random_state=98)

        scoring = {  # 'roc_auc_score':make_scorer(roc_auc_score),
            'Aug_MCC': make_scorer(MCC),
            'Aug_ACC': make_scorer(ACC),
            'Aug_AUC': make_scorer(multiclass_roc_auc_score),
            'Aug_precision_recall': make_scorer(PR_AUC)
        }

        # scores = sklearn.model_selection.cross_validate(estimator_aug, x_train_smoote, y_train_smoote, scoring=scoring,
        #                                                 cv=cv, n_jobs=-1)
        param_grid = {'estimator': [estimator_aug]}
        gscv = GridSearchCV(estimator=ClfSwitcher(),param_grid=param_grid, cv=cv, scoring=scoring, verbose=5, refit=False)
        sup = gscv.fit(x_train_smoote, y_train_smoote)
        scores = pd.DataFrame(gscv.cv_results_)
        results = pd.DataFrame.from_dict(scores)
        results = pd.concat([pd.DataFrame.from_dict([{"dataset_name_o": str(dataset)}]), results], axis=1)
        results = pd.concat([pd.DataFrame.from_dict([{"filtered": str(filtered)}]), results], axis=1)
        # results["dataset_name"] = [dataset_name] * results.shape[0]
        results.to_csv("/Users/esraan/Desktop/ML_final/df_aug_results/"+ dataset_name + "_aug.csv")



