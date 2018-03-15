"""
This file show the use of several algorithms to solve following problem:
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data

This Project is only for academic proposes and it was made
by Normando Zubia, college professor of Universidad La Salle in
Chihuahua.

Bibliography and References are going to be upload in the future
"""

from utils import utils
import pandas

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn import metrics
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    # Open data
    # data = pandas.read_csv("data/train.csv", nrows=10000000)
    data = pandas.read_csv("data/train.csv", nrows=10000)

    # Drop columns
    data = data.drop(['click_time', 'attributed_time'], axis=1)

    # Convert data
    numpy_data = data.values

    feature_vector = numpy_data[:, 1:-1]
    targets = numpy_data[:, -1]

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        feature_vector,
        targets,
        0.25)

    # Algorithms declaration
    names = [
        "Bagging_Regressor",
        "AdaBoost_Regressor",
        "Random_Forest_Regressor",
        "Neural_Network_Regressor",
        "Decision_Tree_Regressor",
        "Support_Vector_Machine_Regressor",
        "K-Neighbor_Regressor"
    ]

    models = [
        BaggingClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='entropy',
                max_depth=10)
        ),
        AdaBoostClassifier(
            base_estimator=tree.DecisionTreeClassifier(
                criterion='entropy',
                max_depth=10)
        ),
        RandomForestClassifier(
            criterion='entropy',
            max_depth=10
        ),
        MLPClassifier(
            hidden_layer_sizes=(50),
            activation="relu",
            solver="adam"
        ),
        tree.DecisionTreeClassifier(
            criterion='entropy'
        )
    ]

    # Algorithm implementation
    for name, em_clf in zip(names, models):
        print("###################---" + name + "---###################")

        em_clf.fit(data_features_train, data_targets_train)

        # Model evaluation
        test_data_predicted = em_clf.predict(data_features_test)

        score = metrics.accuracy_score(data_targets_test, test_data_predicted)
        conf_matrix = confusion_matrix(data_targets_test, test_data_predicted)

        print("Model Score: " + str(score))
        print("Confusion Matrix: ")
        print(conf_matrix)

