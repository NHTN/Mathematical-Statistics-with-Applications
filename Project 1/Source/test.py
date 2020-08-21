# import
import sys
import pickle
import pandas as pd
import numpy as np
import xgboost
from sklearn import preprocessing
from sklearn import metrics

# read command line arguments
argv_len = len(sys.argv)
out_model_path = sys.argv[1] if argv_len > 1 else "test.model"
X_path = sys.argv[2] if argv_len > 2 else "X.csv"
Y_path = sys.argv[3] if argv_len > 3 else "predictY.csv"

# read data
dataX = pd.read_csv(X_path)
del dataX['id']

# pre-process
# handle missing values
NaNRows = dataX[dataX.isna().any(axis=1)]
dataX.drop(index=NaNRows.index, errors='ignore', inplace=True)

# delete duplicates
# duplicateRows = dataX[dataX.duplicated(keep='first')]
# dataX.drop(index=duplicateRows.index, errors='ignore', inplace=True)
# dataY.drop(index=duplicateRows.index, errors='ignore', inplace=True)

# Features Selection
del dataX['model']
del dataX['engineFuel']

# Standardize dataX
numericFeatures = ['odometer', 'year', 'engineCapacity', 'photos']  # Get column names first
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(dataX[numericFeatures])
scaled_df = pd.DataFrame(scaled_df, index=dataX[numericFeatures].index, columns=numericFeatures)
dataX.update(scaled_df)
dataX[numericFeatures]

# Encode categorical
ctg_cols = [col for col in dataX.columns if dataX[col].dtype == object]
with open("category.encoder", 'rb') as file:
    enc = pickle.load(file)
enc_cols = pd.DataFrame(data=enc.transform(dataX[ctg_cols]).toarray(), index=dataX.index, columns=enc.get_feature_names(ctg_cols))
for col in enc_cols.columns:
    dataX[col] = enc_cols[col]
dataX.drop(columns=ctg_cols, inplace=True)

# test
def test(X_test):
    # load_model()
    with open(out_model_path, 'rb') as file:
        model = pickle.load(file)

    Y_pred = model.predict(X_test)

    # save Y_pred
    np.savetxt(Y_path, Y_pred, delimiter=",")

test(dataX)