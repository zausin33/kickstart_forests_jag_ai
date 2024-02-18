import warnings

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from step_04.src.data import Dataset, COLS_CATEGORICAL
from step_04.src.model_factory import ModelFactory

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    data = Dataset()

    # define target and features
    X, y = data.load_xy()
    print('Xshape: \n{}'.format(X.shape))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    gbm_model = ModelFactory.lgbm_regressor()
    cat_features_indices = [X_train.columns.get_loc(c) for c in COLS_CATEGORICAL if c in X_train]
    gbm_model.fit(X_train, y_train)
    y_pred_train = gbm_model.predict(X_train)

    mse = mean_squared_error(y_train, y_pred_train)

    print("Train MSE:", mse, "Train r2:", r2_score(y_train, y_pred_train))

    y_pred_val = gbm_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred_val)

    print("Validation MSE:", mse, "Validation r2:", r2_score(y_val, y_pred_val))

    y_pred_test = gbm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)

    print("Test MSE:", mse, "Test r2:", r2_score(y_test, y_pred_test))
