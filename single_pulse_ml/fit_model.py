""" Tools for fitting ML models to data
"""


import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#from . import reader
import reader

nfreq = 16

def fit_svm(fn_training_data, n_components=10):
	data_train, y_train = reader.read_data(fn_training_data)

	ntimes = data_train.shape[-1] // 16

	# Perform a PCA to reduce data dimensionality
	pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(data_train)

	eigenmodes = pca.components_.reshape((n_components, nfreq, ntimes))

	data_train_pca = pca.transform(data_train)

	param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
	clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
	clf = clf.fit(data_train_pca, y_train)

	return clf, pca

def fit_kneighbors(fn_training_data):
	data_train, y_train = reader.read_data(fn_training_data)

	clf = KNeighborsClassifier()
	clf.fit(data_train, y_train)

	return clf

def predict_test(data_test, model, y_test=None, pca=None):

	n_classes = 2
	target_names = np.array(['RFI', 'Pulse'],
      dtype='|S17')

	if pca is not None:
		data_test = pca.transform(data_test)


	y_pred = model.predict(data_test)

	class_report, conf_matrix = None, None

	if y_test is not None:
		class_report = classification_report(y_test, y_pred, target_names=target_names)
		conf_matrix = confusion_matrix(y_test, y_pred, labels=range(n_classes))
		print class_report
		print conf_matrix
		

	return y_pred, class_report, conf_matrix


if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('fn_training_data', help='training data filename')
	parser.add_argument('-fnout', default='out', help='training data filename')
	parser.add_argument('-n_components', default=10, type=int, help='number of SVD modes to keep')	

	args = parser.parse_args()

	clf, pca = fit_svm(args.fn_training_data, n_components=args.n_components)
	reader.write_pkl(clf, args.fnout + '_model')
	reader.write_pkl(pca, args.fnout + '_pca')








