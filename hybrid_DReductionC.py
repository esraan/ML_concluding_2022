import pandas as pd
import math
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
import skdim
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.decomposition import PCA

features_selected = pd.DataFrame(None)


class hybrid_DReductionC():
    """
    the algorithms perform the following step for FS
    1) calculating the data dimension according to MLE
    2) primary FS using Fisher score and IG
    3) hierarchically clustering the rest of feature into the output of step 1 clusters
    4)performing PCA for each cluster, and sorting the features according to thies var in the first compent
    to choose the best k features
    """

    def __init__(self, target_class=0, Worst_N=0.2, n_comps=1, Scaled=True, k=None):
        self.targetclass = target_class
        self.WorstN = Worst_N
        self.n_comp = n_comps
        self.scaled = Scaled
        self.MLE_dim_val = 1
        self.Fitted = False
        self.k = k
        self._feature_clusters = {}

    def set_params(self, **parameters):
        """ setting parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self):
        """ printing the non private parameters of the class"""
        print("WorstN is" + str(self.WorstN) +
              "n_comp is" + str(self.n_comp) +
              "scaled is" + str(self.scaled) +
              "targetclass is" + self.targetclass)

    def _MLE_dim(self, X):
        """
        input: dataframe or array-like
        output: the dimension calculated according MLE rounded up
        """
        try:
            danco = skdim.id.MLE(neighborhood_based=False)
            dim_num = danco.fit_transform(X)
            if math.ceil(dim_num) == 0:
                self.MLE_dim_val = math.ceil(dim_num) + 1
            else:
                self.MLE_dim_val = math.ceil(dim_num)
        except (ValueError):
            print("MLE dimension of your data is NaN,check your dimension")

    def _fisher_feature_ranker(self, X, y):
        """
        input:
        the dataframe we are performing the feature selection on it
        targetclass is the name of the class
        worstN- is the percentage we want to drop
        output:
        it returns the worst N% feature names sorted according to fisher parameter
        """
        score = fisher_score.fisher_score(X.to_numpy(),
                                          y.to_numpy(), mode='rank')
        fisher_rfeature = X.iloc[:, score].columns  # feature names ranked according to fisher
        to_drop_fisher = int(round(len(score) * self.WorstN, 0))
        return pd.Series(fisher_rfeature).tail(to_drop_fisher)

    def _IG_feature_ranker(self, X, y):
        """
        input:
        the dataframe we are performing the feature selection on it
        targetclass is the name of the class
        worstN- is the percentage we want to drop
        output:
        it returns the worst N% feature names sorted according to IG parameter
        """
        importance = mutual_info_classif(X, y)
        feature_id = pd.DataFrame(X.columns)
        importance_IG = feature_id.join(pd.Series(importance).rename('importance_IG'))
        importance_IG.columns = ["feature_id", "feature_IG"]
        importance_IG.sort_values(by=["feature_IG"], ascending=False, inplace=True)
        to_drop_IG = int(round(importance_IG.shape[0] * self.WorstN, 0))
        return importance_IG["feature_id"].tail(to_drop_IG)

    def _multi_stra_fselection(self, X, y):
        """
        input:
        the dataframe along with the target class and the percentage of worst to drop
        output:
        name of features after dropping the worst features
        """
        whatToDropFisher = self._fisher_feature_ranker(X, y)
        whatToDropIG = self._IG_feature_ranker(X, y)
        featuresUnifedToDrop = whatToDropFisher.append(whatToDropIG, ignore_index=True).drop_duplicates()
        featuresUnifedToDrop
        return pd.Series(X.drop(featuresUnifedToDrop, axis=1).columns)

    def _MICI(self, f_feature, s_feature):
        """
        arguments:
        f_feature:
        one column representing a feature column- column in pandas dataframe.
        s_feature:
        second column representing a feature column- column in pandas dataframe.
        returns:
        the function returns the MICI similarity measure value for the two features.
        """
        try:
            if isinstance(f_feature, np.ndarray) or isinstance(s_feature, np.ndarray):
                f_feature = pd.Series(f_feature)
                s_feature = pd.Series(s_feature)
                var_x = f_feature.var()
                var_y = s_feature.var()
                bothfeatures = pd.concat([f_feature, s_feature], axis=1)
                if var_y == 0 or var_x == 0:
                    p = 0
                else:
                    p = bothfeatures.iloc[:, 0].cov(bothfeatures.iloc[:, 1]) / math.sqrt(var_x * var_y)
                # corr_xy = f_feature.corr(s_feature)
                medgamma = var_x + var_y - math.sqrt((var_x + var_y) ** 2 - 4 * var_x * var_y * (1 - p) ** 2)
                return medgamma / 2
            else:
                var_x = f_feature.var()
                var_y = s_feature.var()
                bothfeatures = pd.concat([f_feature, s_feature], axis=1)
                if var_y == 0 or var_x == 0:
                    p = 0
                else:
                    p = bothfeatures.iloc[:, 0].cov(bothfeatures.iloc[:, 1]) / math.sqrt(var_x * var_y)
                # corr_xy = f_feature.corr(s_feature)
                medgamma = var_x + var_y - math.sqrt((var_x + var_y) ** 2 - 4 * var_x * var_y * (1 - p) ** 2)
                return medgamma / 2
        except ValueError:
            return 0

    def _Heir_clustering(self, X):
        """
        describtion: hidden class method that perform the hierarchical clustering of the features based on
        MICI measurements as a distance parameter.
        X param:the Features to be clustered in AgglomerativeClustering
        return:
        returns the clustering labels in dict, the labels would be the keys and for each cluster label
        the items would be the features names in that cluster.
        """
        agg_clustering = AgglomerativeClustering(affinity='precomputed', n_clusters=self.MLE_dim_val,
                                                 linkage='single')
        # predicting the labels
        self._my_distances = X.corr(method=self._MICI)
        labels = agg_clustering.fit_predict(self._my_distances)
        return labels

    def _pre_PCA(self, label, X):
        column_names = X.columns
        X = X.append(pd.Series(label, index=column_names),
                     ignore_index=True)
        X = X.transpose()
        clustering_dict = {}
        for value in X.iloc[:, -1].unique():
            if value not in clustering_dict.keys():
                clustering_dict[value] = [] + list(X[X.iloc[:, -1] == value].index)
        self.feature_clusters = clustering_dict

    def _PCA_FS(self, X):
        """
        calculates the PCA first component for each cluster and returns the "scoring" for each
        feature in the whole cluster among the others
        X param:
        dataframe type of the features
        return:
        array-like/dataframe type with each feature in all clustering with it perspective "scoring"
        according to the var of first component in the PCA
        """
        if self.scaled:
            totaldataframe = pd.DataFrame()
            for key, value in self.feature_clusters.items():
                pca = PCA(n_components=self.n_comp)
                pca.fit(X.loc[:, value])
                all_comp = pd.DataFrame(abs(pca.components_))
                all_comp.columns = X.loc[:, value].columns
                totaldataframe = pd.concat([totaldataframe, all_comp], axis=1)
            totaldataframe = totaldataframe.T
            self.par = totaldataframe.shape[0]
            self.chosen = totaldataframe[totaldataframe.iloc[:, 0] > 0.2].T
            return totaldataframe[totaldataframe.iloc[:, 0] > 0.2].T
        else:
            sc = StandardScaler()
            X = sc.fit_transform(X)
            for key, value in self.feature_clusters.items():
                pca = PCA(n_components=self.n_comp)
                pca.fit(X.loc[:, value])
                all_comp = pd.DataFrame(abs(pca.components_))
                all_comp.columns = X.loc[:, value].columns
                totaldataframe = pd.concat([totaldataframe, all_comp], axis=1)
            totaldataframe = totaldataframe.T
            self.par = totaldataframe.shape[0]
            self.chosen = totaldataframe[totaldataframe.iloc[:, 0] > 0.2].T
            return totaldataframe[totaldataframe.iloc[:, 0] > 0.2].T

    def _hybrid_DReduction(self, X, y):
        """
        description:
        wrapper method that call all the other method inside it, it perfomrs the whole processes of the hybrid
        dimension reduction algorithm in the reference.
        X param: the features in a dataframe type
        y param: the labels for each sample as passed in the X
        return: None, it updates the attribute of the class
        """
        mydataframe = pd.concat([X, y], axis=1)
        self._MLE_dim(X)
        featuresToKeep = self._multi_stra_fselection(X, y)
        X = X[list(featuresToKeep)].copy()
        label = self._Heir_clustering(X)
        self._pre_PCA(label, X)
        final = self._PCA_FS(X)
        self.final = final
        not_chose_columns = list(mydataframe.drop(self.targetclass, axis=1).columns.difference(self.final.columns))
        not_chosen = pd.DataFrame(0, index=np.arange(1), columns=not_chose_columns)
        self.final = pd.concat([self.final, not_chosen], axis=1).T.sort_values(0,ascending=False).T
        self.all_features = self.final.reindex(sorted(self.final.columns), axis=1).values
        self.Fitted = True

    def fit(self, X, y):
        """
        description: performing the fit, aka calling the wrapper function _hybrid_DReduction
        X param: the features in a dataframe type
        y param: the labels for each sample as passed in the X
        return:
        self
        """
        try:
            if self.k != None:
                self._hybrid_DReduction(X, y)
                self.final = self.final.iloc[:, 0:self.k].copy()
                return self
            else:
                self._hybrid_DReduction(X, y)
                return self
        except:
            raise ("i dont know what happend")

    def transform(self, X, y=None):
        """
        description: performing the fit, aka calling the wrapper function _hybrid_DReduction
        X param: the features in a dataframe type
        y param: None or the labels for each sample as passed in the X
        return:
        the features selected in dataframe type after FS
        """
        if self.Fitted:
            if self.k != None:
                return X.loc[:, self.final.columns]
            else:
                return X.loc[:, self.chosen.columns]
        else:
            raise Exception("not fitted!")
