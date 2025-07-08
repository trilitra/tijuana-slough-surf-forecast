from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import io
import base64
import requests
import os
from datetime import timedelta

app = Flask(__name__)

# Загрузка модели и скейлера
model = joblib.load('lr_model.pkl')
scaler = joblib.load('scaler.pkl')


# Функция для загрузки данных с NDBC
def load_ndbc_data(url='https://www.ndbc.noaa.gov/data/5day2/46086_5day.txt'):
    try:
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.splitlines()

        data = pd.read_csv(
            io.StringIO('\n'.join(lines[2:])),
            sep=r'\s+',
            names=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT',
                   'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
        )

        special_values = [99.0, 99.00, 999.0, 9999.0]
        for col in ['WSPD', 'GST', 'WVHT', 'DPD', 'APD']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].replace(special_values, np.nan)

        data['datetime'] = pd.to_datetime(data[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
            columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))

        # Сдвиг времени на UTC+5
        data['datetime'] = data['datetime'] + timedelta(hours=5)

        data = data[['datetime', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD']].dropna()

        # Агрегация до почасовых данных
        data = data.set_index('datetime').resample('h').mean(numeric_only=True).reset_index()

        if len(data) < 4:
            raise ValueError("Недостаточно данных после агрегации (нужно минимум 4 часа)")

        data = data.tail(4)  # Последние 4 часа для лагов
        return data
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных NDBC: {str(e)}")


# Функция для создания признаков
def create_features(current_data, lag_columns=['WSPD', 'GST', 'APD', 'WVHT'], lags=3):
    current_data = current_data.copy()

    current_data['WSPD_GST'] = current_data['WSPD'] * current_data['GST']
    current_data['APD_DPD'] = current_data['APD'] * current_data['DPD']
    current_data['WSPD_APD'] = current_data['WSPD'] * current_data['APD']
    current_data['GST_APD'] = current_data['GST'] * current_data['APD']

    for col in lag_columns:
        for lag in range(1, lags + 1):
            current_data[f'{col}_lag{lag}'] = current_data[col].shift(lag)

    feature_columns = ['APD', 'WSPD_GST', 'APD_DPD', 'WSPD_APD', 'GST_APD',
                       'APD_lag1', 'APD_lag2', 'APD_lag3', 'WVHT_lag1', 'WVHT_lag2', 'WVHT_lag3']

    current_data = current_data.dropna()
    current_data[feature_columns] = current_data[feature_columns].infer_objects(copy=False).interpolate(method='linear',
                                                                                                        limit_direction='forward').fillna(
        current_data[feature_columns].mean())
    return current_data[feature_columns]


# Функция для прогноза на заданное количество часов
def forecast_hours(model, scaler, initial_data, n_steps=24):
    predictions = []
    current_data = initial_data.copy()

    for _ in range(n_steps):
        X = create_features(current_data)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled[-1:])[0]
        predictions.append(pred)

        new_row = current_data.iloc[-1].copy()
        new_row['WVHT'] = pred
        new_row['datetime'] = current_data['datetime'].iloc[-1] + timedelta(hours=1)
        current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        current_data = current_data.iloc[-10:]  # Ограничение для оптимизации

    return predictions


# Маршрут для главной страницы
@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_24_data = None
    plot_24_url = None
    forecast_week_data = None
    plot_week_url = None

    if request.method == 'POST':
        try:
            # Загрузка данных с NDBC
            initial_data = load_ndbc_data()

            if 'forecast_24' in request.form:
                # Прогноз на 24 часа
                predictions = forecast_hours(model, scaler, initial_data, n_steps=24)
                forecast_dates = pd.date_range(start=initial_data['datetime'].iloc[-1] + timedelta(hours=1), periods=24,
                                               freq='h')
                forecast_df = pd.DataFrame({
                    'datetime': forecast_dates,
                    'WVHT_pred': predictions,
                    'WVHT_min': [max(0, p - 0.5) for p in predictions]
                })

                # График для 24 часов
                plt.figure(figsize=(10, 5))
                plt.plot(initial_data['datetime'], initial_data['WVHT'], label='Исторические данные (WVHT)',
                         color='green')
                plt.plot(forecast_df['datetime'], forecast_df['WVHT_pred'], label='Прогноз WVHT', color='blue')
                plt.fill_between(forecast_df['datetime'], forecast_df['WVHT_min'], forecast_df['WVHT_pred'],
                                 color='blue', alpha=0.1, label='Диапазон прогноза (-0.5 м)')
                plt.xlabel('Дата и время (UTC+5)')
                plt.ylabel('Высота волн (м)')
                plt.title('Tijuana Slough: Прогноз высоты волн на 24 часа')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_24_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                # Форматирование таблицы для 24 часов
                forecast_df['datetime'] = forecast_df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
                forecast_df['WVHT_range'] = forecast_df.apply(lambda x: f"{x['WVHT_min']:.2f}–{x['WVHT_pred']:.2f}",
                                                              axis=1)
                forecast_24_data = forecast_df[['datetime', 'WVHT_range']].to_dict('records')

            elif 'forecast_week' in request.form:
                # Прогноз на неделю (168 часов)
                predictions = forecast_hours(model, scaler, initial_data, n_steps=168)
                forecast_dates = pd.date_range(start=initial_data['datetime'].iloc[-1] + timedelta(hours=1),
                                               periods=168, freq='h')
                forecast_df = pd.DataFrame({
                    'datetime': forecast_dates,
                    'WVHT_pred': predictions,
                    'WVHT_min': [max(0, p - 0.5) for p in predictions]
                })

                # Агрегация по дням для таблицы
                forecast_df['date'] = forecast_df['datetime'].dt.date
                daily_forecast = forecast_df.groupby('date').agg({
                    'WVHT_pred': 'mean',
                    'WVHT_min': 'mean'
                }).reset_index()
                daily_forecast['WVHT_range'] = daily_forecast.apply(
                    lambda x: f"{x['WVHT_min']:.2f}–{x['WVHT_pred']:.2f}", axis=1)
                forecast_week_data = daily_forecast[['date', 'WVHT_range']].to_dict('records')

                # График для недели
                plt.figure(figsize=(10, 5))
                plt.plot(initial_data['datetime'], initial_data['WVHT'], label='Исторические данные (WVHT)',
                         color='green')
                plt.plot(forecast_df['datetime'], forecast_df['WVHT_pred'], label='Прогноз WVHT', color='blue')
                plt.fill_between(forecast_df['datetime'], forecast_df['WVHT_min'], forecast_df['WVHT_pred'],
                                 color='blue', alpha=0.1, label='Диапазон прогноза (-0.5 м)')
                plt.xlabel('Дата и время (UTC+5)')
                plt.ylabel('Высота волн (м)')
                plt.title('Tijuana Slough: Прогноз высоты волн на неделю')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_week_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

            return render_template('index.html', forecast_24_data=forecast_24_data, plot_24_url=plot_24_url,
                                   forecast_week_data=forecast_week_data, plot_week_url=plot_week_url)
        except Exception as e:
            return render_template('index.html', error=f"Не удалось загрузить данные с NDBC: {str(e)}")

    return render_template('index.html', forecast_24_data=None, plot_24_url=None,
                           forecast_week_data=None, plot_week_url=None)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Используем порт из переменной окружения или 5000 по умолчанию
    app.run(host='0.0.0.0', port=port, debug=True)