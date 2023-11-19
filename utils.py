import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from datetime import datetime





class Utils:

  def load_data(path):
    return pd.read_csv('Sleep_Efficiency.csv', index_col = 0)



  def missing_info(df):
    num_null = df.shape[0] - df.dropna().shape[0]
    null_percent = ((df.shape[0] - df.dropna().shape[0]) / df.shape[0]) * 100
    columns_with_missing = df.columns[df.isnull().any()].tolist()

    df.info()

    print(f'\nNull rows: {num_null}')
    print(f'{null_percent :.2f}% of data')
    print("Columns with missing values:", columns_with_missing)

    return columns_with_missing



  def label_encode(data, columns_to_encode):
    encoded_data = data.copy()
    label_encoder = LabelEncoder()
    
    for column in columns_to_encode:
      encoded_data[column] = label_encoder.fit_transform(data[column])
    
    return encoded_data



  def find_binary_columns(df, threshold=5):
    categorical_columns = []

    for column in df.columns:
      unique_values = df[column].unique()
      
      if len(unique_values) <= threshold:
        categorical_columns.append(column)

    return categorical_columns



  def date_to_decimal(date):
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    time_decimal = date_time.hour + date_time.minute / 60 + date_time.second / 3600

    return time_decimal
  





class Imputation:
  def evaluate(self, y_test, y_test_predicted, scaler=None):
    if scaler is not None:
            y_test = scaler.inverse_transform(y_test)
            y_test_predicted = scaler.inverse_transform(y_test_predicted)

    mae = mean_absolute_error(y_test, y_test_predicted)
    r2 = r2_score(y_test, y_test_predicted)
    rmse = mean_squared_error(y_test, y_test_predicted, squared=False)

    return {'MAE': mae, 'R-squared': r2, 'RMSE': rmse}



  def split(self, df, column, columns_with_missing):
    df_drop = df.dropna()
    y = df_drop[column]

    df_drop = df_drop.drop(columns=columns_with_missing)
    X = df_drop

    return train_test_split(X, y, test_size=0.2)
  


  def model_imputation(self, model, df, column, columns_with_missing):
    df_ret = df.copy()

    X_train, X_test, y_train, y_test = self.split(df_ret, column, columns_with_missing)

    model.fit(X_train, y_train)

    y_test_predicted = model.predict(X_test)
    evaluation_results = self.evaluate(y_test, y_test_predicted)

    X_missing = df_ret[df_ret[column].isnull()].drop(columns=columns_with_missing)
    y_missing_predicted = model.predict(X_missing)

    df_ret.loc[df_ret[column].isnull(), column] = y_missing_predicted

    return df_ret, evaluation_results

  

  def imputer_imputation(self, imputer, df, column, columns_with_missing):
      df_ret = df.copy()

      X_train, X_test, y_train, y_test = self.split(df_ret, column, columns_with_missing)

      scaler = StandardScaler()

      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.fit_transform(X_test)
      X_missing_scaled = scaler.transform(df_ret[df_ret[column].isnull()].drop(columns=columns_with_missing))

      imputer.fit(X_train_scaled, y_train)

      y_test_predicted_scaled = imputer.transform(X_test_scaled)
      y_test_predicted = scaler.inverse_transform(y_test_predicted_scaled)
      evaluation_results = self.evaluate(y_test, y_test_predicted, scaler)

      y_missing_predicted_scaled = imputer.transform(X_missing_scaled)
      y_missing_predicted = scaler.inverse_transform(y_missing_predicted_scaled)

      df_ret.loc[df_ret[column].isnull(), column] = y_missing_predicted

      return df_ret, evaluation_results



  def evaluate_best_methods(self, results, method='RMSE'):
    best_methods = {}

    for column, model_results in results.items():
      best_method = min(model_results, key=lambda x: model_results[x][method])
      best_methods[column] = best_method

    return best_methods



  def try_imputation_models(self, df, columns_with_missing):
    models = {
      'Linear Regression': LinearRegression(),
      'Decision Tree': DecisionTreeRegressor(),
      'Random Forest': RandomForestRegressor(),
      'SVR': SVR(),
      'Gradient Boosting Regressor': GradientBoostingRegressor(),
      'KNeighbors': KNeighborsRegressor(),
      'Lasso': Lasso(),
      'Ridge': Ridge(),
      'HuberRegressor': HuberRegressor(),
      'ElasticNet': ElasticNet(),
      'SGDRegressor': SGDRegressor(),
    }

    imputers = {
      'Simple Imputer (Mean)': SimpleImputer(strategy='mean'),
      'KNN Imputer': KNNImputer(n_neighbors=5),
    }

    results = {column: {} for column in columns_with_missing}

    for column in columns_with_missing:
      for model_name, model in models.items():
        print(model_name)
        _, evaluation_results = self.model_imputation(model, df, column, columns_with_missing)
        results[column][model_name] = evaluation_results

      for imputer_name, imputer in imputers.items():
        print(imputer_name)
        _, evaluation_results = self.imputer_imputation(imputer, df, column, columns_with_missing)
        results[column][imputer_name] = evaluation_results

    return results