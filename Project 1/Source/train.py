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
X_path = sys.arg[1] if argv_len > 1 else "X_train.csv"
Y_path = sys.arg[2] if argv_len > 2 else "Y_train.csv"
out_model_path = sys.arg[3] if argv_len > 3 else "output.model"

# read data
dataX = pd.read_csv(X_path)
dataY = pd.read_csv(Y_path)
del dataX['id']
del dataY['id']

# pre-process
# handle missing values
NaNRows = dataX[dataX.isna().any(axis=1)]
dataX.drop(index=NaNRows.index, errors='ignore', inplace=True)
dataY.drop(index=NaNRows.index, errors='ignore', inplace=True)

# delete duplicates
duplicateRows = dataX[dataX.duplicated(keep='first')]
dataX.drop(index=duplicateRows.index, errors='ignore', inplace=True)
dataY.drop(index=duplicateRows.index, errors='ignore', inplace=True)

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
# ctg_col = [(col, len(dataX[col].unique())) for col in dataX.columns if dataX[col].dtype == object]
# for col, _ in ctg_col:
#     enc = preprocessing.OneHotEncoder(handle_unknown='ignore')  # dummy
#     enc.fit(dataX[col].unique().reshape(-1, 1))
#     enc_cols = enc.transform(dataX[col].to_numpy().reshape(-1, 1)).toarray()
#     dataX[[col + str(i) for i in range(len(enc_cols[0]))]] = pd.DataFrame(enc_cols, index=dataX.index)
#     dataX.drop(columns=col, inplace=True)
ctg_cols = [col for col in dataX.columns if dataX[col].dtype == object]
enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
enc.fit(dataX[ctg_cols])

with open("category.encoder", 'wb') as file:
    pickle.dump(enc, file)

enc_cols = pd.DataFrame(data=enc.transform(dataX[ctg_cols]).toarray(), index=dataX.index, columns=enc.get_feature_names(ctg_cols))
for col in enc_cols.columns:
    dataX[col] = enc_cols[col]
dataX.drop(columns=ctg_cols, inplace=True)

# train
def train(X_train, Y_train):
    xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=8)
    model = xgb.fit(X_train, Y_train)

    # save_model()
    with open(out_model_path, 'wb') as file:
        pickle.dump(model, file)


# run
train(dataX, dataY)
print('Da train xong!')

def test(X_test, Y_test):
    # load_model()
    with open(out_model_path, 'rb') as file:
        model = pickle.load(file)
    #
    Y_pred = model.predict(X_test)
    rmse = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred))
    return rmse;

# RMSE
print("RMSE:", test(dataX, dataY))