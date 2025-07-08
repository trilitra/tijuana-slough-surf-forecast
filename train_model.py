import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Чтение обучающих данных
files = ['data2021.txt', 'data2022.txt', 'data2023.txt', 'data2024.txt', 'data2025.txt']
dataframes = []
for file in files:
    df = pd.read_csv(file, sep=r'\s+', skiprows=2,
                     names=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT',
                            'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE'])
    dataframes.append(df)
data = pd.concat(dataframes, ignore_index=True)

# Преобразование типов и замена пропусков
special_values = [99.0, 99.00, 999.0, 9999.0]
numeric_columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col] = data[col].replace(special_values, np.nan)

# Удаление строк с пропусками в WVHT
data_cleaned = data.dropna(subset=['WVHT'])

# Удаление VIS и TIDE
data_cleaned = data_cleaned.drop(columns=['VIS', 'TIDE'])

# Создание столбца datetime
data_cleaned['datetime'] = pd.to_datetime(data_cleaned[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
    columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))

# Сортировка по времени
data_cleaned = data_cleaned.sort_values('datetime')

# Новые признаки
data_cleaned['WSPD_GST'] = data_cleaned['WSPD'] * data_cleaned['GST']
data_cleaned['APD_DPD'] = data_cleaned['APD'] * data_cleaned['DPD']
data_cleaned['WSPD_APD'] = data_cleaned['WSPD'] * data_cleaned['APD']
data_cleaned['GST_APD'] = data_cleaned['GST'] * data_cleaned['APD']

# Обработка выбросов с IQR=5.0
def remove_outliers(df, column, iqr_factor=5.0):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['WVHT', 'WSPD', 'GST']:
    data_cleaned = remove_outliers(data_cleaned, col, iqr_factor=5.0)
    print(f"После удаления выбросов в {col}: {len(data_cleaned)} строк")

# Создание лаговых признаков
lag_columns = ['WSPD', 'GST', 'APD', 'WVHT']
for col in lag_columns:
    for lag in [1, 2, 3]:
        data_cleaned[f'{col}_lag{lag}'] = data_cleaned[col].shift(lag)

# Удаление ненужных столбцов
data_cleaned = data_cleaned.drop(columns=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP'])

# Разделение данных
train_data, test_data = train_test_split(data_cleaned, test_size=0.2, random_state=42)
train_data = train_data.dropna()
test_data = test_data.dropna()

# Выбор признаков
features = ['APD', 'WSPD_GST', 'APD_DPD', 'WSPD_APD', 'GST_APD', 'APD_lag1', 'APD_lag2', 'APD_lag3', 'WVHT_lag1', 'WVHT_lag2', 'WVHT_lag3']
target = 'WVHT'

# Обновление выборок
X_train = train_data[features]
y_train = train_data[target]
X_test = test_data[features]
y_test = test_data[target]

# Интерполяция пропусков
X_train = X_train.infer_objects(copy=False).interpolate(method='linear', limit_direction='forward').fillna(X_train.mean())
X_test = X_test.infer_objects(copy=False).interpolate(method='linear', limit_direction='forward').fillna(X_test.mean())

# Нормализация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
mask = y_test > 2
lr_mae_high = mean_absolute_error(y_test[mask], lr_pred[mask]) if mask.sum() > 0 else np.nan
lr_rmse_high = np.sqrt(mean_squared_error(y_test[mask], lr_pred[mask])) if mask.sum() > 0 else np.nan
lr_r2_high = r2_score(y_test[mask], lr_pred[mask]) if mask.sum() > 0 else np.nan

print("Линейная регрессия:")
print(f"RMSE: {lr_rmse:.3f}")
print(f"MAE: {lr_mae:.3f}")
print(f"R²: {lr_r2:.3f}")
print(f"RMSE (WVHT>2): {lr_rmse_high:.3f}")
print(f"MAE (WVHT>2): {lr_mae_high:.3f}")
print(f"R² (WVHT>2): {lr_r2_high:.3f}")

# Сохранение модели и скейлера
joblib.dump(lr_model, 'lr_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
