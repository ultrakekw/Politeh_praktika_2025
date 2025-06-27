import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from catboost import CatBoostRegressor
from prophet import Prophet

def generate_timeseries(start_date, end_date, trend_coef=0.0, seasonality_amplitude=0.0, season_length=30, noise_level=1.0):
    """
    Генерирует синтетический временной ряд с трендом, сезонностью и шумом.
    
    Параметры:
    - start_date (str): начальная дата ('YYYY-MM-DD').
    - end_date (str): конечная дата ('YYYY-MM-DD').
    - trend_coef (float): коэффициент тренда (0 — без тренда, >0 — рост, <0 — спад).
    - seasonality_amplitude (float): амплитуда сезонности.
    - season_length (int): длина одного цикла сезонности (в днях).
    - noise_level (float): уровень шума (стандартное отклонение).
    
    Возвращает:
    - pd.DataFrame с колонками ['timestamp', 'value'].
    """

    # Даты
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)

    # Тренд
    trend = trend_coef * np.arange(n)

    # Сезонность
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(n) / season_length)

    # Шум
    noise = np.random.normal(loc=0, scale=noise_level, size=n)

    # Финальный ряд
    values = trend + seasonality + noise

    return pd.DataFrame({'timestamp': dates, 'value': values})

    

def split_train_test(df, time_col='timestamp', ratio=0.7):
    """
    Делит DataFrame на обучающую и тестовую выборки по времени в заданной пропорции.
    
    Параметры:
    - df (pd.DataFrame): исходный DataFrame.
    - time_col (str): имя колонки с датами.
    - ratio (float): доля train выборки (по умолчанию 0.7).
    
    Возвращает:
    - train_df, test_df (pd.DataFrame, pd.DataFrame).
    """

    df = df.sort_values(by=time_col).reset_index(drop=True)
    split_index = int(len(df) * ratio)
    
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    return train_df, test_df


def describe_timeseries(df, target_col='value'):
    """
    Вычисляет описательные статистики временного ряда:
    среднее, медиану, std, min, max, range, квантили.
    
    Параметры:
    - df (pd.DataFrame): DataFrame с данными.
    - target_col (str): колонка с числовыми значениями.
    
    Возвращает:
    - dict с вычисленными статистиками.
    """

    stats = {}
    series = df[target_col]

    stats['mean'] = series.mean()
    stats['median'] = series.median()
    stats['std'] = series.std()
    stats['var'] = series.var()
    stats['min'] = series.min()
    stats['max'] = series.max()
    stats['range'] = series.max() - series.min()
    
    # Квантильные значения
    quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    for q in quantiles.index:
        stats[f'quantile_{int(q*100)}'] = quantiles[q]
    
    return stats

def plot_forecast(full_df, pred_df, time_col='timestamp', value_col='value'):
    """
    Строит график истинных и предсказанных значений на одном графике по датам.
    
    Параметры:
    - full_df (pd.DataFrame): DataFrame с истинными значениями.
    - pred_df (pd.DataFrame): DataFrame с предсказаниями.
    - time_col (str): колонка с датами.
    - value_col (str): колонка с числовыми значениями.
    
    Возвращает:
    - None (строит график).
    """

    # Оставим только даты из предсказаний
    mask = full_df[time_col].isin(pred_df[time_col])
    actual = full_df[mask].sort_values(by=time_col)
    predicted = pred_df.sort_values(by=time_col)

    # Построение
    plt.figure(figsize=(12, 4))
    plt.plot(actual[time_col], actual[value_col], label='Истинные значения', linewidth=2)
    plt.plot(predicted[time_col], predicted[value_col], label='Прогноз', linewidth=2)

    plt.xlabel("Дата")
    plt.ylabel("Значение")
    plt.title("Истинные значения vs Прогноз (плавная линия)")
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_forecast(full_df, pred_df, time_col='timestamp', value_col='value'):
    """
    Считает метрики качества прогноза:
    MAE, RMSE, MAPE, R2.
    
    Параметры:
    - full_df (pd.DataFrame): DataFrame с истинными значениями.
    - pred_df (pd.DataFrame): DataFrame с предсказанными значениями.
    - time_col (str): колонка с датами.
    - value_col (str): колонка с числовыми значениями.
    
    Возвращает:
    - dict с метриками.
    """

    # Совпадающие даты
    merged = pd.merge(pred_df, full_df, on=time_col, suffixes=('_pred', '_true'))

    y_true = merged[f'{value_col}_true']
    y_pred = merged[f'{value_col}_pred']

    # Метрики
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred, squared=False),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'R2': r2_score(y_true, y_pred)
    }

    return metrics

##########дальше идут функции с обучением различных алгоритмов##########

def forecast_with_ar(train_df, test_df, time_col='timestamp', target_col='value', lags=10):
    """
    Прогнозирование с помощью авторегрессии (AR) из statsmodels.
    
    Параметры:
    - train_df (pd.DataFrame): обучающая выборка.
    - test_df (pd.DataFrame): тестовая выборка.
    - time_col (str): колонка с датами.
    - target_col (str): колонка с числовыми значениями.
    - lags (int): количество лагов.
    
    Возвращает:
    - pd.DataFrame с колонками [timestamp, value] с предсказанными значениями.
    """

    # Сортировка по времени
    train_df = train_df.sort_values(by=time_col)
    test_df = test_df.sort_values(by=time_col)

    # Обучение модели AR
    model = AutoReg(train_df[target_col], lags=lags, old_names=True)
    model_fit = model.fit()

    # Прогнозирование на тестовый период
    forecast = model_fit.predict(
        start=len(train_df),
        end=len(train_df) + len(test_df) - 1,
        dynamic=False
    ).reset_index(drop=True)

    # Формируем результат
    result_df = pd.DataFrame({
        time_col: test_df[time_col].values,
        target_col: forecast
    })

    return result_df


def forecast_with_sarima(train_df, test_df, time_col='timestamp', target_col='value', order=(1,1,1), seasonal_order=(0,0,0,0)):

    """
    Прогнозирование с помощью модели SARIMA.
    
    Параметры:
    - train_df (pd.DataFrame): обучающая выборка.
    - test_df (pd.DataFrame): тестовая выборка.
    - time_col (str): колонка с датами.
    - target_col (str): колонка с числовыми значениями.
    - order (tuple): параметры ARIMA (p,d,q).
    - seasonal_order (tuple): параметры сезонности (P,D,Q,s).
    
    Возвращает:
    - pd.DataFrame с колонками [timestamp, value] с предсказанными значениями.
    """

    train_df = train_df.sort_values(by=time_col)
    test_df = test_df.sort_values(by=time_col)

    model = SARIMAX(train_df[target_col], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)

    forecast = model_fit.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)

    return pd.DataFrame({time_col: test_df[time_col].values, target_col: forecast.values})



def forecast_with_catboost(train_df, test_df, time_col='timestamp', target_col='value', lags=[1,2,3]):
    """
    Прогнозирование с помощью CatBoostRegressor с использованием лагов как признаков.
    
    Параметры:
    - train_df (pd.DataFrame): обучающая выборка.
    - test_df (pd.DataFrame): тестовая выборка.
    - time_col (str): колонка с датами.
    - target_col (str): колонка с числовыми значениями.
    - lags (list): список лагов, которые использовать как признаки.
    
    Возвращает:
    - pd.DataFrame с колонками [timestamp, value] с предсказанными значениями.
    """

    df_all = pd.concat([train_df, test_df]).sort_values(by=time_col).reset_index(drop=True)
    
    # Генерация лагов
    for lag in lags:
        df_all[f'lag_{lag}'] = df_all[target_col].shift(lag)
    
    df_all.dropna(inplace=True)
    
    train_idx = df_all[time_col] < test_df[time_col].min()
    train = df_all[train_idx]
    test = df_all[~train_idx]

    features = [col for col in df_all.columns if col.startswith('lag_')]

    model = CatBoostRegressor(verbose=0)
    model.fit(train[features], train[target_col])

    preds = model.predict(test[features])

    return pd.DataFrame({time_col: test[time_col].values, target_col: preds})


def forecast_with_prophet(train_df, test_df, time_col='timestamp', target_col='value'):
    """
    Прогнозирование с помощью модели Prophet.
    
    Параметры:
    - train_df (pd.DataFrame): обучающая выборка.
    - test_df (pd.DataFrame): тестовая выборка.
    - time_col (str): колонка с датами.
    - target_col (str): колонка с числовыми значениями.
    
    Возвращает:
    - pd.DataFrame с колонками [timestamp, value] с предсказанными значениями.
    """

    df = train_df[[time_col, target_col]].rename(columns={time_col: 'ds', target_col: 'y'})
    future = test_df[[time_col]].rename(columns={time_col: 'ds'})

    model = Prophet()
    model.fit(df)

    forecast = model.predict(future)
    
    return pd.DataFrame({
        time_col: test_df[time_col].values,
        target_col: forecast['yhat'].values
    })

#TODO forecast_with_fedot