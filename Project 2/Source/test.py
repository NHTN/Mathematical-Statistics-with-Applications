# import
import sys
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

# read command line arguments
argv_len = len(sys.argv)
out_model_path = sys.argv[1] if argv_len > 1 else "output.model"
data_path = sys.argv[2] if argv_len > 2 else "train.csv"

# read data
data = np.loadtxt(data_path, dtype=np.uint8, delimiter=',', skiprows=1)
X = data[:, 1:]
Y = data[:, 0]

# normalize
X = X / 255


# data transform
def data_transform(X, use_poly=False):
    Z = np.zeros((len(X), 7 * 7))
    for k, row in enumerate(X):
        pic = row.reshape(28, 28)
        new_row = np.zeros(7 * 7)
        for i in range(7):
            for j in range(7):
                new_row[i * 7 + j] = np.mean(pic[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4])
        Z[k] = new_row

    if use_poly:
        poly = PolynomialFeatures(2)
        Z = poly.fit_transform(Z)

    return Z


# load_model()
with open(out_model_path, 'rb') as file:
    model = pickle.load(file)

print('Accuracy:', model.score(data_transform(X)[:, 7:-7], Y)*100)