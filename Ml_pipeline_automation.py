if __name__ == "__main__":

    #working
    from timeit import default_timer as timer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.base import BaseEstimator
    from sklearn.linear_model import SGDClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score as ACC
    from sklearn.metrics import matthews_corrcoef  as MCC
    from sklearn.metrics import roc_auc_score as AUC
    from sklearn.metrics import average_precision_score as PR
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeavePOut
    from sklearn.model_selection import *
    from sklearn import *
    import numpy as np
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import StandardScaler
    import skdim
    from skfeature.function.information_theoretical_based.MRMR import mrmr
    from sklearn import feature_selection
    import math
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import label_binarize
    from sklearn.feature_selection import RFE
    from sklearn.feature_selection import SelectFdr
    from skfeature.function.similarity_based.reliefF import reliefF
    from sklearn.feature_selection import SelectKBest
    import numpy as np
    from sklearn.feature_selection import SelectKBest
    import os
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import f_classif
    from sklearn.preprocessing import label_binarize
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from scipy.sparse import *
    import copy
    import random
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from skfeature.function.information_theoretical_based.MRMR import mrmr
    from scipy.special import logsumexp
    from GBC import *
    from GBC_new import *
    from hybrid_DReductionC import *
    features_selected = pd.DataFrame(None)



    class KBest_features(BaseEstimator):
        """
         KBest_features class is wrapper class that make use of all the algorithms in our assignment
        and fit them into use in the gridsearch, as it do the same job as ClfSwitcher
        its inherit from the BaseEstimator of sklearn        """
        global features_selected

        def __init__(self, score_func=f_classif, *, k=10, alpha=5e-2, estimator=SVC(kernel="linear")):
            self.k = k
            self.score_func = score_func
            self.alpha = alpha
            self.estimator = estimator
            self._is_fitted = False
            # self._selector

        def fit(self, X, y):
            """
            description:
            a class method that uses the features and labels in dataframe type, passes them to the
            appropriate algorithms that performs FS on the features.
            X param: dataframe type of features to be FS performed on
            y param: the labels of the samples passed in X
            return:
            self
            """
            global features_selected
            start = timer()
            if self.score_func == mrmr or self.score_func == reliefF:
                best_featuresSelector = SelectKBest(score_func=self.score_func, k=self.k)
                best_featuresSelector = best_featuresSelector.fit(X, y)
                self._is_fitted = True
                binaric_vector = np.array(list(map(int, best_featuresSelector.get_support())))
                self.best_kfeatures = X.columns[np.where(binaric_vector == 1)]
                self.total_scores = best_featuresSelector.scores_
                end = timer()
                self._selector = best_featuresSelector
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{"Features_selected": self.best_kfeatures, "scoring_features": self.total_scores,"k": self.k, "score_func":self.score_func, "time:":end-start}])], axis=0, ignore_index=True)
                return self
            elif self.score_func == RFE:
                selector = RFE(estimator=self.estimator, n_features_to_select=self.k)
                selector = selector.fit(X, y)
                self._is_fitted = True
                binaric_vector = np.array(list(map(int, selector.support_)))
                X = pd.DataFrame(X)
                self.best_kfeatures = X.columns[np.where(binaric_vector == 1)]
                self.total_scores = selector.ranking_
                end = timer()
                self._selector = selector
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{"Features_selected": self.best_kfeatures, "scoring_features": self.total_scores,"k": self.k, "score_func":self.score_func, "time:":end-start}])], axis=0, ignore_index=True)
                return self
            elif self.score_func == SelectFdr:
                FDRSelector = SelectFdr(alpha=self.alpha)
                FDRSelector = FDRSelector.fit(X, y)
                self._is_fitted = True
                self.total_scores = FDRSelector.scores_
                med = pd.DataFrame([pd.Series(X.columns), pd.Series(self.total_scores)]).T
                med.columns = ["column_name", "rating"]
                sorted_med = med.sort_values(by=["rating"], ascending=False)
                self.best_kfeatures = sorted_med.iloc[0:self.k, 0].values
                end = timer()
                self._selector = FDRSelector
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{"Features_selected": self.best_kfeatures, "scoring_features": self.total_scores,"k": self.k, "score_func":self.score_func, "time:":end-start}])], axis=0, ignore_index=True)
                return self
            elif self.score_func == hybrid_DReductionC :
                hybrid_DReductionC_selector = hybrid_DReductionC("class", 0.35, 1, True, k=self.k)
                hybrid_DReductionC_selector = hybrid_DReductionC_selector.fit(X,y)
                self._is_fitted = True
                self.total_scores = hybrid_DReductionC_selector.all_features
                self.best_kfeatures = hybrid_DReductionC_selector.final.columns
                end = timer()
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{"Features_selected": self.best_kfeatures, "scoring_features": self.total_scores,"k": self.k, "score_func":self.score_func, "time:":end-start}])], axis=0, ignore_index=True)
                self._selector = hybrid_DReductionC_selector
                return self
            elif self.score_func == GBC:
                m = int(0.6 * X.shape[1])
                mrmr_out = mrmr(X.to_numpy(), y.to_numpy())[:m]
                X = X.iloc[:, mrmr_out].copy()
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
                evaluator = FitnessEvaluator(X_train, X_test, y_train, y_test)
                K = self.k
                max_trials = 5
                seed = 1
                SN = 80
                lower = [0 for i in range(K)]
                upper = [X_train.shape[1] - 1 for i in range(K)]
                gbc_selector = GBC(lower, upper, fun=evaluator, seed=seed, max_itrs=100, max_trials=max_trials,
                                       numb_bees=SN, K=K)
                output_GBC = gbc_selector.run()
                self.best_kfeatures = output_GBC["features"]
                self.total_scores = output_GBC["score"]
                end = timer()
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{
                                                                                              "Features_selected": self.best_kfeatures,
                                                                                              "scoring_features": self.total_scores,
                                                                                              "k": self.k,
                                                                                              "score_func": self.score_func,
                                                                                              "time:": end - start}])],
                                              axis=0, ignore_index=True)
                self._is_fitted = True
                self._selector = gbc_selector
                return self
            elif self.score_func == GBC_new:
                m = int(0.6 * X.shape[1])
                mrmr_out = mrmr(X.to_numpy(), y.to_numpy())[:m]
                X = X.iloc[:, mrmr_out].copy()
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
                evaluator = FitnessEvaluator(X_train, X_test, y_train, y_test)
                K = self.k
                max_trials = 5
                seed = 1
                SN = 80
                lower = [0 for i in range(K)]
                upper = [X_train.shape[1] - 1 for i in range(K)]
                gbc_selector = GBC_new(lower, upper, fun=evaluator, seed=seed, max_itrs=100, max_trials=max_trials,
                                 numb_bees=SN, K=K)
                output_GBC = gbc_selector.run()
                self.best_kfeatures = output_GBC["features"]
                self.total_scores = output_GBC ["score"]
                end = timer()
                features_selected = pd.concat([features_selected, pd.DataFrame.from_dict([{ "Features_selected": self.best_kfeatures,"scoring_features": self.total_scores,"k": self.k, "score_func":self.score_func, "time:": end - start}])],axis=0, ignore_index=True)
                self._is_fitted = True
                self._selector = gbc_selector
                return self
            else:
                raise ValueError("function is not defined")

        def transform(self, X, y=None):
            """

            X param:
            the features to selected in FS algorithm according to the algorithm passed in the constructor

            return:
            the features selected in dataframed type
            """
            if self._is_fitted:
                if self.score_func == GBC or self.score_func == GBC_new or self.score_func == SelectFdr:
                    to_return = X.loc[:, self.best_kfeatures].copy()
                else:
                    to_return = pd.DataFrame(self._selector.transform(X), columns=self.best_kfeatures)
                return to_return
            else:
                raise ValueError("not fitted")

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self

    class ClfSwitcher(BaseEstimator):

        def __init__(
                self,
                estimator=SGDClassifier(),
        ):
            """
            A Custom BaseEstimator that can switch between classifiers.
            :param estimator: sklearn object - The classifier
            """
            self.estimator = estimator

        def fit(self, X, y=None, **kwargs):
            self.estimator = self.estimator.fit(X, y)
            return self


        def predict(self, X, y=None):
            return self.estimator.predict(X)

        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

        def score(self, X, y):
            return self.estimator.score(X, y)

    class PreProcessor():

        def __int__(self, variance=0):
            self.variance = variance
            self._is_Fitted = False

        def fit(self, X, y=None):
            """
            steps: performing the fit, aka performing the pre-processing of the data:
            1) filling NA values
            2) FS removing features with variance of 0
            3) Transformation of  the data using PowerTransformer
            X param: the features in a dataframe type
            y param: the labels for each sample as passed in the X
            return:
            self
            """
            self.numerical_cols = [i for i in X.columns if X.dtypes[i] not in ['object', 'bool']]
            self.categorical_cols = [i for i in X.columns if X.dtypes[i] in ['object', 'bool']]
            # Preprocessing for numerical data
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean'))
            ])
            # Preprocessing for categorical data
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            self.preprocessor_1 = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, self.numerical_cols),
                    ('cat', categorical_transformer, self.categorical_cols)
                ])
            self.preprocessor_1.fit(X)
            to_perform_further = pd.DataFrame(self.preprocessor_1.transform(X), columns=X.columns)
            self.preprocessor_2 = feature_selection.VarianceThreshold(threshold=0.0)
            self.preprocessor_2.fit(to_perform_further)
            new_step = pd.DataFrame(self.preprocessor_2.transform(to_perform_further),
                                    columns=to_perform_further.columns[self.preprocessor_2._get_support_mask()])
            self.preprocessor_3 = PowerTransformer().fit(new_step)
            transformed = self.preprocessor_3.transform(new_step)
            self.final_preprocessed = pd.DataFrame(transformed, columns=new_step.columns)
            self._is_Fitted = True
            return self

        def transform(self, X, y=None):
            """
           description: returning the output of the fit, aka performing the following preprocessing of the data:
            1) filling NA values
            2) FS removing features with variance of 0
            3) Transformation of  the data using PowerTransformer
            X param: the features in a dataframe type
            y param: the labels for each sample as passed in the X
            return:
            the dataframe of the X with column names saved
            """
            if self._is_Fitted:
                num_col = [i for i in X.columns if X.dtypes[i] not in ['object', 'bool']]
                cat_col = [i for i in X.columns if X.dtypes[i] in ['object', 'bool']]
                if num_col == self.numerical_cols and cat_col==self.categorical_cols:
                    to_perform_further = pd.DataFrame(self.preprocessor_1.transform(X), columns=X.columns)
                    new_step = pd.DataFrame(self.preprocessor_2.transform(to_perform_further),
                                            columns=to_perform_further.columns[self.preprocessor_2._get_support_mask()])
                    transformed = self.preprocessor_3.transform(new_step)
                    self.final_preprocessed_extra = pd.DataFrame(transformed, columns=new_step.columns)
                    return self.final_preprocessed_extra
                else:
                    return self.final_preprocessed
            else:
                raise ValueError("not fitted!")

        def set_params(self, **parameters):
            for parameter, value in parameters.items():
                setattr(self, parameter, value)
            return self

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
        df = pd.read_csv(path,low_memory=False )
        if "Unnamed: 0" in df.columns:
            df.set_index('Unnamed: 0', inplace=True)
        y = df.loc[:,"class"]#.to_numpy()
        df = df.drop("class", axis=1)
        was_filtered = False
        X=df#.to_numpy()
        if (X.shape[1]>1000):
            selector = SelectKBest(k=1000)
            selector.fit(X, y)
            features = X.columns[selector.get_support()]
            X = X[features].copy()
            was_filtered = True

        X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
        return   X_train, X_test, y_train, y_test ,X, was_filtered

    main_path = "C:/Users/elinor/PycharmProjects/ml final project/"#/Users/esraan/Desktop/ML_4/data/clean/all_together/data/"# "" /sise/home/esraan/ML_4/input/
    for dataset in os.listdir(main_path):
        os.chdir(main_path)
        if dataset.endswith("csv"):
            X_train, X_test, y_train, y_test,X , filtered= data_prep(dataset)
            proccessing_instance = PreProcessor()
            proccessing_instance.fit(X_train)
            X_train = proccessing_instance.transform(X_train)
            my_pipeline = Pipeline (steps=[('reduce_dim', KBest_features()),
                                                  ('clf', ClfSwitcher()),
                                                  ])

            param_grid = {'clf__estimator': [GaussianNB(),RandomForestClassifier(), LogisticRegression(), KNeighborsClassifier(), SVC()],  # ,,RandomForestClassifier()
                          # SVM if hinge loss / logreg if log loss
                          'reduce_dim__k': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 50, 100],
                          'reduce_dim__score_func': [mrmr,reliefF, RFE, SelectFdr, hybrid_DReductionC, GBC, GBC_new],#, reliefF #, hybrid_DReductionC
                          }

            scoring = {  # 'roc_auc_score':make_scorer(roc_auc_score),
                'MCC': make_scorer(MCC),
                'ACC': make_scorer(ACC),
                'AUC': make_scorer(multiclass_roc_auc_score),
                'precision_recall': make_scorer(PR_AUC)
            }

            if X.shape[0] < 50:
                cv = LeavePOut(2)
                cv_method = "LeavePOut(2)"
            elif 50 <= X.shape[0] <= 100:
                cv = LeaveOneOut()
                cv_method = "LeaveOneOut"
            elif X.shape[0] > 100:
                cv = StratifiedKFold(n_splits=10)
                cv_method = "10 folds cv"
            elif X.shape[0] > 1000:
                cv = StratifiedKFold(n_splits=5)
                cv_method = "5 folds cv"

            gscv = GridSearchCV(my_pipeline, param_grid, cv=cv,  scoring=scoring, verbose=5, refit=False)
            try:
                sup = gscv.fit(X_train, y_train)
                df = pd.DataFrame(gscv.cv_results_)
                df = pd.concat([pd.DataFrame.from_dict([{"dataset_name":str(dataset)}]),df], axis=1)
                df = pd.concat([pd.DataFrame.from_dict([{"filtered": str(filtered)}]), df], axis=1)
                os.chdir("/sise/home/esraan/ML_4/output/") #/Users/esraan/Desktop/ML_4/data/clean/all_together/")  # "" /sise/home/esraan/ML_4/output
                dataset = dataset.replace(".csv", "")
                df.to_csv(str(dataset) + "_result.csv")
                features_selected.to_csv(str(dataset) + "_FS_result.csv")

            except Exception:
                file1 = open("our_log_file.txt", "w+")
                file1.write(dataset + "caused error !")

