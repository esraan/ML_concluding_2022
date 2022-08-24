import pandas as pd
import numpy as np
from scipy.sparse import *
import sys
import copy
import random
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skfeature.function.information_theoretical_based.MRMR import mrmr
from scipy.special import logsumexp
from sklearn.feature_selection import SelectKBest
from timeit import default_timer as timer


class Bee_solution(object):
    """ Creates a bee_solution object.Which is random solution vector"""

    def __init__(self, lower, upper, fun):
        """
        Instantiates a bee object randomly.
        Parameters:
        ----------
            :param list lower  : lower bound of solution vector , denotes in the paper as Lj
            :param list upper  : upper bound of solution vector , denotes in the paper as Uj
            :param def  fun    : evaluation function , the SVM accuracy
            :param def  funcon : constraints function, must return a boolean
        """
        self._random_sol(lower, upper)
        if (fun != None):
            self.value = fun(self.vector)
        else:
            self.value = sys.float_info.max
        self.counter = 0

    def _random_sol(self, lower, upper):
        """ Initialises a solution vector randomly. """
        self.vector = []
        for i in range(len(lower)):
            self.vector.append(int(lower[i] + random.random() * (upper[i] - lower[i])))


class GBC(object):
    def run(self):
        """ Runs an the algorithm. """
        cost = {}
        cost["score"] = []
        for itr in range(self.max_itrs):

            for index in range(self.size):
                self.send_employee(index)
            self.find_best()
            self.send_onlookers()
            self.find_best()
            self.send_scout()
            self.send_second_scout()

        cost["features"] = self.columns[self.solution]
        vector_ = pd.Series(index=self.columns, data=0)
        vector_[cost["features"]] = 1
        cost["score"] = vector_.values.tolist()
        return cost

    def __init__(self,
                 lower, upper,
                 fun=None,
                 numb_bees=30,
                 max_itrs=30,
                 max_trials=None,
                 seed=None,
                 K=None):
        """
        Instantiates a bee hive object.
        1. INITIALISATION PHASE.
        Parameters:
        ----------
            :param list lower          : lower bound of solution vector
            :param list upper          : upper bound of solution vector
            :param def fun             : evaluation function of the optimal problem
            :param def numb_bees       : number of active bees within the hive
            :param int max_trials      : max number of trials without any improvment (i.e abandonment number)
            :param int seed            : seed of random number generator

        """
        assert len(upper) == len(lower), "'lower' and 'upper' must be a list of the same length."
        if seed == None:
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)
        self.size = numb_bees
        self.dim = len(lower)
        self.max_itrs = max_itrs
        if (max_trials == None):
            self.max_trials = 0.6 * self.size * self.dim
        else:
            self.max_trials = max_trials
        self.evaluate = fun
        self.lower = lower
        self.upper = upper
        self.K = K
        self.columns = fun.cols
        # initialises current best and its a solution vector
        self.best = sys.float_info.max
        self.best_index = None
        self.solution = None
        self.population = [Bee_solution(lower, upper, fun) for i in range(self.size)]
        # initialises best solution vector to food nectar
        self.find_best()
        # computes selection probability
        self.compute_probability()
        self.solution_probas = None

    def find_best(self):
        """ Finds current best solution bee candidate. """
        values = [bee.value for bee in self.population]
        index = values.index(max(values))
        if values[index] < self.best:
            self.best = values[index]
            self.best_index = index
            self.solution = self.population[index].vector

    def compute_probability(self):
        """
        Computes the relative chance that a given solution vector is
        chosen in the onlooker Phase.
        """
        values = [bee.value for bee in self.population]
        max_values = max(values)
        self.probas = [v / sum(values) for v in values]
        return self.probas

    def uniform_crossover(self, index):
        """
        uniform crossover operation-based model for
        an onlooker bee’s selection of a neighbour.
        Uniform crossover treats each gene independently
        and making a random choice as to which parent it should to be
        inherited from within a specific index and position.
        """

        offspring1 = copy.deepcopy(self.population[index])
        offspring2 = copy.deepcopy(self.population[index])
        # selects another bee
        bee_ix = index
        while (bee_ix == index):
            bee_ix = random.randint(0, self.size - 1)
        for i in range(0, self.dim - 1):
            RN = random.uniform(0, 1)
            if RN > 0.6:
                offspring1.vector[i] = self.solution[i]
                j = random.randint(0, self.size - 1)
                offspring2.vector[i] = self.population[j].vector[i]
            else:
                j = random.randint(0, self.size - 1)
                offspring1.vector[i] = self.population[j].vector[i]
                offspring2.vector[i] = self.solution[i]
            offspring1.value = self.evaluate(offspring1.vector)
            offspring2.value = self.evaluate(offspring2.vector)
            # deterministic crowding
            if offspring1.value > offspring2.value:
                NewSolution=copy.deepcopy(offspring1)
            else:
                NewSolution=copy.deepcopy(offspring2)
            if NewSolution.value > self.population[index].value:
                self.population[index] = NewSolution
                self.population[index].counter = 0
            else:
                self.population[index].counter+=1
    def send_employee(self, index):
        """
        2. SEND EMPLOYED BEES PHASE.
        ---------------------------
        A new candidate solutions are produced for
        each employed bee by cross-over and mutation of the employees.
        If the modified vector of the mutant bee solution is better than
        that of the original bee, the new vector is assigned to the bee.
        """
        zombee = copy.deepcopy(self.population[index])
        d = random.randint(0, self.dim - 1)
        bee_ix = index
        while (bee_ix == index):
            bee_ix = random.randint(0, self.size - 1)
        zombee.vector[d] = int(self._mutate(d, index, bee_ix))
        zombee.vector = self._check(zombee.vector, dim=d)
        zombee.value = self.evaluate(zombee.vector)
        if (zombee.value > self.population[index].value):
            self.population[index] = copy.deepcopy(zombee)
            self.population[index].counter = 0
        else:
            self.population[index].counter += 1

    def send_onlookers(self):
        """

        3. SEND ONLOOKERS PHASE.
        -----------------------
    The onlooker bees learn the location of the solution examining the probabilities
    of a given solution to bee chosen and assign to itself the better solutions gene vector.

        """
        numb_onlookers = 0
        while (numb_onlookers < self.size):
            phi = random.random()
            index = self.select(phi)

            # sends new onlooker
            self.uniform_crossover(index)

            # increments number of onlookers
            numb_onlookers += 1

    def select(self, phi):
        """
        Parameter(s):
        ------------
            :param float beta : "roulette wheel selection" parameter - i.e. 0 <= beta <= max(probas)
        """
        probas = self.compute_probability()
        for index in range(self.size):
            if phi < probas[index]:
                return index
            else:
                return 0

    def send_scout(self):
        """
        4. SEND SCOUT PHASE.
        -----------------------
        if the fitness (I.E SVM accuracy) value associated with
        a solution is not improved for a limited number of specified trials,
        then the employee bee becomes a scout to which a crossover value
        is assigned for finding the new solution.
        """
        trials = [self.population[i].counter for i in range(self.size)]
        index = trials.index(max(trials))
        if (trials[index] > self.max_trials):
            self.population[index] = Bee_solution(self.lower, self.upper, self.evaluate)
            self.send_employee(index)

    #

    def send_second_scout(self):
        """ 5. SEND SCOUT PHASE.
        -----------------------
        In the second scout bee, we look for a place around the highest
        fitness solution generated so far, which is denoted as the Queen
        Bee,The crossover process happens relative to the Queen be solution
        """
        trials = [self.population[i].counter for i in range(self.size)]
        index = trials.index(max(trials))
        if (trials[index] > self.max_trials):
            self.population[index].counter = 0
            self.population[index].vector = self.solution
            MPR = 0.01
            if random.uniform(0, 1) < MPR:
                zombee = copy.deepcopy(self.population[index])
                d = random.randint(0, self.dim - 1)
                #bee_ix = index
                #while (bee_ix == index):
                bee_ix = random.randint(0, self.size - 1)
                zombee.vector[d] = int(self.solution[d] + (random.random() - 0.5) * 2 *(self.population[bee_ix].vector[d] - self.solution[d]))
                zombee.vector = self._check(zombee.vector, dim=d)
                zombee.value = self.evaluate(zombee.vector)
                if (zombee.value > self.population[index].value):
                    self.population[index] = copy.deepcopy(zombee)
                    self.population[index].counter = 0
                else:
                    self.population[index].counter += 1
                self.send_employee(index)

    def _mutate(self, dim, current_bee, other_bee):
        """
        mutate the solution by given EQ
        """

        return self.population[current_bee].vector[dim] + \
               (random.random() - 0.5) * 2 * \
               (self.population[current_bee].vector[dim] - self.population[other_bee].vector[dim])

    def _check(self, vector, dim=None):
        """
        Checks that a solution vector is contained within the
        pre-determined lower and upper bounds of the problem.
        """
        if (dim == None):
            range_ = range(self.dim)
        else:
            range_ = [dim]
        for i in range_:
            if (vector[i] < self.lower[i]):
                vector[i] = self.lower[i]
            elif (vector[i] > self.upper[i]):
                vector[i] = self.upper[i]
        return vector





class FitnessEvaluator():
    """
    Evaluates the fitness of a solution vector.
    """

    def __init__(self, X_train, X_test, y_train, y_test, C=1):
        """
        Initializes the fitness evaluator.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.C = C
        self.cols = self.X_train.columns

    def __call__(self, gene_indexes):
        """
        Evaluates the fitness of a solution vector.
        """
        svm = SVC(C=self.C)
        X_train_subset = self.X_train.iloc[:, gene_indexes]
        X_test_subset = self.X_test.iloc[:, gene_indexes]
        svm.fit(X_train_subset, self.y_train)
        y_pred = svm.predict(X_test_subset)
        return accuracy_score(self.y_test, y_pred)

#
# def data_prep(path):
#     df = pd.read_csv(path, low_memory=False)
#     if "Unnamed: 0" in df.columns:
#         df.set_index('Unnamed: 0', inplace=True)
#     y = df.loc[:, "class"]  # .to_numpy()
#     df = df.drop("class", axis=1)
#     X = df  # .to_numpy()
#     if (X.shape[1] > 1000):
#         selector = SelectKBest(k=1000)
#         selector.fit(X, y)
#         features = X.columns[selector.get_support()]
#         X = X[features].copy()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18, stratify=y)
#     return X_train, X_test, y_train, y_test, X ,y
#
#
# #X_train, X_test, y_train, y_test, X ,y = data_prep("C:/Users/elinor/PycharmProjects/ml final project/Qin2014_Cirrhosis-20220731T141442Z-001.csv")
#
# m = int(0.6 * X.shape[1])
# mrmr_out = mrmr(X.to_numpy(), y.to_numpy())[:m]
# X = X.iloc[:, mrmr_out].copy()
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
# evaluator = FitnessEvaluator(X_train, X_test, y_train, y_test)
# K = 5
# max_trials = 5
# seed = 1
# SN = 80
# lower = [0 for i in range(K)]
# upper = [X_train.shape[1] - 1 for i in range(K)]
# gbc_selector = GBC(lower, upper, fun=evaluator, seed=seed, max_trials=max_trials,
#                    numb_bees=SN, K=K)
# start = timer()
# gbc_selector.run()
# end = timer()
# print(end-start)