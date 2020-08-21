# import
import sys
import pickle
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

# read command line arguments
argv_len = len(sys.argv)
data_path = sys.argv[1] if argv_len > 1 else "train.csv"
out_model_path = sys.argv[2] if argv_len > 2 else "output.model"

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


model = XGBClassifier()
model.fit(data_transform(X)[:, 7:-7], Y)

with open(out_model_path, 'wb') as file:
    pickle.dump(model, file)

print("Da train xong!")
print("In-sample accuracy:", model.score(data_transform(X)[:, 7:-7], Y))