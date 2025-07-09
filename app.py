# Импорт необходимых библиотек
from flask import Flask, request, render_template  # Flask для создания веб-сервиса, request для обработки запросов, render_template для отображения HTML-шаблонов
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Для создания графиков прогнозов
import io
import base64  # Для конвертации графиков в изображения для отображения в браузере
import requests  # Для загрузки данных с сайта NDBC
import os
from datetime import timedelta  # Для работы с временными интервалами
import joblib  # Для загрузки сохраненных моделей и скейлера

# Создаем приложение Flask
app = Flask(__name__)

# Загрузка модели и скейлера
# Линейная регрессия выбрана за высокую точность (R²=0.981) и быстродействие, что идеально для веб-сервиса
model = joblib.load('lr_model.pkl')  # Загружаем обученную модель линейной регрессии
scaler = joblib.load('scaler.pkl')  # Загружаем скейлер для нормализации входных данных


# Функция для загрузки данных с буя NDBC 46086
# Данные берутся из 5-дневного архива (https://www.ndbc.noaa.gov/data/5day2/46086_5day.txt)
def load_ndbc_data(url='https://www.ndbc.noaa.gov/data/5day2/46086_5day.txt'):
    try:
        # Загружаем данные с сайта NDBC (Национальный центр данных о буях)
        response = requests.get(url)
        response.raise_for_status()  # Проверяем, что запрос успешен
        lines = response.text.splitlines()  # Разбиваем ответ на строки

        # Читаем данные, пропуская первые 2 строки (заголовки), и задаем имена столбцов
        data = pd.read_csv(
            io.StringIO('\n'.join(lines[2:])),
            sep=r'\s+',  # Разделитель — пробелы
            names=['YY', 'MM', 'DD', 'hh', 'mm', 'WDIR', 'WSPD', 'GST', 'WVHT',
                   'DPD', 'APD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'PTDY', 'TIDE']
        )

        # Описание переменных датасета:
        # YY: Год (например, 2025)
        # MM: Месяц (1–12)
        # DD: День (1–31)
        # hh: Час (0–23)
        # mm: Минута (0 или 30, данные собираются каждые 30 минут)
        # WDIR: Направление ветра (градусы, относительно севера)
        # WSPD: Средняя скорость ветра (м/с, влияет на формирование волн)
        # GST: Скорость порывов ветра (м/с, важна для экстремальных условий)
        # WVHT: Значительная высота волны (м, целевая переменная для прогноза)
        # DPD: Доминантный период волны (с, определяет частоту волн)
        # APD: Средний период волны (с, показывает стабильность волн)
        # MWD: Среднее направление волны (градусы)
        # PRES: Атмосферное давление (гПа)
        # ATMP: Температура воздуха (°C)
        # WTMP: Температура воды (°C)
        # DEWP: Точка росы (°C)
        # VIS: Видимость (мили, исключена из модели из-за низкой информативности)
        # PTDY: Изменение давления (гПа, исключено из модели)
        # TIDE: Уровень прилива (м, исключен из-за пропусков или низкой корреляции)

        # Обработка пропусков: заменяем специальные значения (99, 999, 9999) на NaN
        special_values = [99.0, 99.00, 999.0, 9999.0]
        for col in ['WSPD', 'GST', 'WVHT', 'DPD', 'APD']:
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Преобразуем в числовой формат
            data[col] = data[col].replace(special_values, np.nan)  # Заменяем специальные значения на NaN

        # Создаем столбец datetime для временной сортировки
        data['datetime'] = pd.to_datetime(data[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
            columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}))

        # Сдвиг времени на UTC+5 для соответствия местному времени Tijuana Slough
        data['datetime'] = data['datetime'] + timedelta(hours=5)

        # Оставляем только нужные столбцы и удаляем строки с пропусками
        data = data[['datetime', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD']].dropna()

        # Агрегируем данные до почасовых значений (усредняем за час)
        data = data.set_index('datetime').resample('h').mean(numeric_only=True).reset_index()

        # Проверяем, достаточно ли данных для создания лагов (нужно минимум 4 часа)
        if len(data) < 4:
            raise ValueError("Недостаточно данных после агрегации (нужно минимум 4 часа)")

        # Берем последние 4 часа для создания лаговых признаков
        data = data.tail(4)
        return data
    except Exception as e:
        raise Exception(f"Ошибка загрузки данных NDBC: {str(e)}")


# Функция для создания признаков
# Создаем те же признаки, что использовались при обучении модели, для консистентности
def create_features(current_data, lag_columns=['WSPD', 'GST', 'APD', 'WVHT'], lags=3):
    current_data = current_data.copy()

    # Комбинированные признаки для учета взаимодействия параметров
    current_data['WSPD_GST'] = current_data['WSPD'] * current_data['GST']  # Скорость ветра × порывы
    current_data['APD_DPD'] = current_data['APD'] * current_data['DPD']  # Средний период × доминантный период
    current_data['WSPD_APD'] = current_data['WSPD'] * current_data['APD']  # Скорость ветра × средний период
    current_data['GST_APD'] = current_data['GST'] * current_data['APD']  # Порывы ветра × средний период

    # Лаговые признаки: значения за предыдущие 1–3 часа для учета временных зависимостей
    for col in lag_columns:
        for lag in range(1, lags + 1):
            current_data[f'{col}_lag{lag}'] = current_data[col].shift(lag)

    # Список признаков, используемых моделью
    feature_columns = ['APD', 'WSPD_GST', 'APD_DPD', 'WSPD_APD', 'GST_APD',
                       'APD_lag1', 'APD_lag2', 'APD_lag3', 'WVHT_lag1', 'WVHT_lag2', 'WVHT_lag3']

    # Интерполяция пропусков и заполнение средними значениями для надежности
    current_data = current_data.dropna()
    current_data[feature_columns] = current_data[feature_columns].infer_objects(copy=False).interpolate(
        method='linear', limit_direction='forward').fillna(current_data[feature_columns].mean())

    return current_data[feature_columns]


# Функция для прогноза на заданное количество часов
# Прогнозируем высоту волн (WVHT) на n_steps часов вперед (например, 24 часа или 168 часов для недели)
def forecast_hours(model, scaler, initial_data, n_steps=24):
    predictions = []
    current_data = initial_data.copy()

    # Итеративно делаем прогноз для каждого часа
    for _ in range(n_steps):
        X = create_features(current_data)  # Создаем признаки
        X_scaled = scaler.transform(X)  # Нормализуем данные
        pred = model.predict(X_scaled[-1:])[0]  # Прогноз для последнего часа
        predictions.append(pred)

        # Добавляем прогноз в данные как новое значение WVHT
        new_row = current_data.iloc[-1].copy()
        new_row['WVHT'] = pred
        new_row['datetime'] = current_data['datetime'].iloc[-1] + timedelta(hours=1)
        current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        current_data = current_data.iloc[-10:]  # Ограничиваем данные для оптимизации (храним последние 10 часов)

    return predictions


# Маршрут для главной страницы
# Обрабатываем GET и POST запросы для отображения формы и результатов прогноза
@app.route('/', methods=['GET', 'POST'])
def index():
    forecast_24_data = None
    plot_24_url = None
    forecast_week_data = None
    plot_week_url = None

    if request.method == 'POST':
        try:
            # Загружаем последние данные с буя NDBC 46086
            initial_data = load_ndbc_data()  # Исправлено: используем функцию load_ndbc_data

            if 'forecast_24' in request.form:
                # Прогноз на 24 часа
                predictions = forecast_hours(model, scaler, initial_data, n_steps=24)
                # Создаем временные метки для прогноза (начинаем с последнего времени + 1 час)
                forecast_dates = pd.date_range(start=initial_data['datetime'].iloc[-1] + timedelta(hours=1), periods=24,
                                               freq='h')
                # Формируем DataFrame с прогнозами и диапазоном (прогноз ± 0.5 м для учета погрешности)
                forecast_df = pd.DataFrame({
                    'datetime': forecast_dates,
                    'WVHT_pred': predictions,
                    'WVHT_min': [max(0, p - 0.5) for p in predictions]  # Минимальная высота волны (не отрицательная)
                })

                # График для прогноза на 24 часа
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

                # Конвертируем график в изображение для отображения в браузере
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_24_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

                # Форматируем данные для таблицы на 24 часа
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

                # Агрегируем данные по дням для недельного прогноза
                forecast_df['date'] = forecast_df['datetime'].dt.date
                daily_forecast = forecast_df.groupby('date').agg({
                    'WVHT_pred': 'mean',
                    'WVHT_min': 'mean'
                }).reset_index()
                daily_forecast['WVHT_range'] = daily_forecast.apply(
                    lambda x: f"{x['WVHT_min']:.2f}–{x['WVHT_pred']:.2f}", axis=1)
                forecast_week_data = daily_forecast[['date', 'WVHT_range']].to_dict('records')

                # График для недельного прогноза
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

                # Конвертируем график в изображение
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_week_url = base64.b64encode(img.getvalue()).decode()
                plt.close()

            # Отображаем результаты в HTML-шаблоне
            return render_template('index.html', forecast_24_data=forecast_24_data, plot_24_url=plot_24_url,
                                   forecast_week_data=forecast_week_data, plot_week_url=plot_week_url)
        except Exception as e:
            # В случае ошибки отображаем сообщение об ошибке
            return render_template('index.html', error=f"Не удалось загрузить данные с NDBC: {str(e)}")

    # Для GET-запросов отображаем пустую форму
    return render_template('index.html', forecast_24_data=None, plot_24_url=None,
                           forecast_week_data=None, plot_week_url=None)


# Запуск приложения
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Используем порт из переменной окружения или 5000 по умолчанию
    app.run(host='0.0.0.0', port=port, debug=True)