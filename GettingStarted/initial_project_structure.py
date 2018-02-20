"""
This script show a initial structure for machine learning projects

Author: Normando Zubia
Universidad de La Salle
"""

import pandas
import numpy
import logging
from sklearn import datasets
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ##################################################################
# --------------------- Data Utils ---------------------------------
# ##################################################################

def load_data(file_path):
	"""
    This function loads a csv file and return its numpy representation
    :param file_path: File Path
    :return: numpy array data
    """
	data = pandas.read_csv(file_path)

	return data


def convert_data_to_numeric(data, columns_to_convert):
	"""
    This function convert a nominal representation to number to use the data with
    sklearn algorithms
    :param data: pandas feature vector
    :param columns_to_convert: array with nominals columns to convert
    :return: numpy array with numeric data
	"""
	numpy_data = data.values

	for i in range(len(numpy_data[0])):
		temp = numpy_data[:, i]
		if i in columns_to_convert:
			dict = numpy.unique(numpy_data[:, i])
			for j in range(len(dict)):
				temp[numpy.where(numpy_data[:, i] == dict[j])] = j

			numpy_data[:, i] = temp

	return numpy_data


if __name__ == '__main__':
	# ##################################################################
	# ------------------------ 1.- Open File ---------------------------
	# ##################################################################

	# Open CSV File
	# file_path = ""
	# data = load_data(file_path)

	# Test data
	iris = datasets.load_iris()

	# ##################################################################
	# ---------------- 2.- Pre-processing Phase ------------------------
	# ##################################################################

	# Separate features and targets
	data_features = iris.data
	data_targets = iris.target

	# logger.debug("Data Features: \n %s", data_features.describe)
	# logger.debug("Data Targets: \n %s", data_targets.describe)

	# Data splitting
	data_features_train, data_features_test, data_targets_train, data_targets_test = train_test_split(data_features, data_targets, test_size=0.25)

	# ##################################################################
	# ------------------ 3.- Model Training ----------------------------
	# ##################################################################

	#Model declaration
	"""
	Parameters to select:
	criterion: "entropy" or "gini": default: gini
	max_depth: maximum depth of tree, default: None
	"""
	dec_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
	dec_tree.fit(data_features_train, data_targets_train)

	# ##################################################################
	# ----------------- 4.- Data Evaluation ----------------------------
	# ##################################################################

	#Model evaluation
	test_data_predicted = dec_tree.predict(data_features_test)
	score = metrics.accuracy_score(data_targets_test, test_data_predicted)

	logger.debug("Model Score: " + str(score))
	logger.debug("Probability of each class: \n")
	#Measure probability of each class
	prob_class = dec_tree.predict_proba(data_features_test)
	logger.debug(prob_class)
