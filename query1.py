import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Queries to run
QUERYA = True
QUERYB_1 = True
QUERYB_2 = True
QUERYB_3 = True
QUERYB_4 = True

# Constants
TRAIN = 0.75
AUTOTUNE = False
TUNING_JOBS = 4
C = 0.655
COEF0 = 1.795
DEGREES = 4
KERNEL = "poly"

# Load the dataset
data = pd.read_csv("winequality-red.csv")

# Split the dataset (random seed creates small abnormalities in the results)
y = data.quality
x = data.drop("quality", axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=TRAIN)
x_train_bak = None
x_test_bak = None

# Scale the dataset as it's highly recommended on SVMs
scaler = preprocessing.MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Create an SVM classifier and report
def predict():
	svclassifier = SVC(kernel=KERNEL, C=C, degree=DEGREES, coef0=COEF0)
	svclassifier.fit(x_train, y_train)
	y_pred = svclassifier.predict(x_test)
	report = classification_report(y_test, y_pred, output_dict=True, zero_division=False)["weighted avg"]
	precision = report["precision"]
	recall = report["recall"]
	f1 = report["f1-score"]
	print("Precision: {:.2f}\t\tRecall: {:.2f}\t\tF1: {:.2f}\n".format(precision, recall, f1))
	return (precision, recall, f1)

# Query A
if QUERYA:
	if AUTOTUNE:
		Cs = np.arange(0.63, 0.67, 0.001)
		DEGREES = [4]
		COEF0S = np.arange(1.78, 1.81, 0.001)
		kernels = ["poly"]
		param_grid = {'C': Cs, 'degree': DEGREES, 'coef0': COEF0S, 'kernel': kernels}
		grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 1, n_jobs=TUNING_JOBS)
		grid.fit(x_train_scaled, y_train)
		print("\nFinished tuning\n")
		print("Best params: ")
		print(grid.best_params_)
		print("\nResults:\n")
		grid_predictions = grid.predict(x_test_scaled)
		report = classification_report(y_test, grid_predictions, output_dict=True, zero_division=False)["weighted avg"]
		precision = report["precision"]
		recall = report["recall"]
		f1 = report["f1-score"]
		print("Precision: {:.2f}\t\tRecall: {:.2f}\t\tF1: {:.2f}\n".format(precision, recall, f1))

	print("Query A: ")
	predict()

def backup():
	# Backup the dataset for the sub-queries
	global x_train_bak
	global x_test_bak 
	x_train_bak = x_train.copy()	
	x_test_bak = x_test.copy()

def restore():
	# Restore the dataset for the sub-queries
	global x_train
	global x_test
	x_train = x_train_bak.copy()
	x_test = x_test_bak.copy()

# Query B

# Remove 1/3 of the pH column
for index, row in x_train.iterrows():
	if index % 3:
		row["pH"] = np.nan

backup()

# Sub-Query 1
if QUERYB_1:
	restore()
	# Remove the whole column
	x_train.drop("pH", axis=1, inplace=True)
	x_test.drop("pH", axis=1, inplace=True)
	print("Query B 1: ")
	predict()

# Sub-Query 2
if QUERYB_2:
	restore()
	# Fill the missing values with the column average
	x_train["pH"].fillna(x_train["pH"].mean(), inplace= True)
	print("Query B 2: ")
	predict()

# Sub-Query 3
if QUERYB_3:
	restore()
	# Fill the missing values using Logistic Regression
	print("Query B 3: ")
	predict()

if QUERYB_4:
	restore()
	# Fill the missing values using K-Means
	print("Query B 4: ")
	predict()
