import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,GRU,Bidirectional
import xgboost as xgb
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
import joblib
import random
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_all_seeds(seed_value=42):

    # Set Python seed
    random.seed(seed_value)

    # Set NumPy seed
    np.random.seed(seed_value)

    # Set TensorFlow seed
    tf.random.set_seed(seed_value)

    # Set environment variables for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # TensorFlow session configuration if using older versions of TF
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        print("TensorFlow deterministic operations not available in this version")

    print(f"All seeds have been set to {seed_value}")


# set_all_seeds(42)

ticker = "ZOMATO.NS"  # Apple stock
start_date = "2015-01-01"
end_date = "2025-04-06"
plot_ind = ['Close','EMA_20']
ind_begin = "2022-01-01"
ind_end="2024-01-01"
corr_val = 'Close'
seq_len = 60
Predictor = 'Close'
test_size = 0.6
model_controller = 2  # New model type (5 for Bi-LSTM, 6 for Bi-GRU)
trainable = True
model_path = 'tcs.keras'

def fetch_stock_data(ticker: str, start_date: str, end_date: str):

    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # stock_data1 = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def fetch_vix_data(start_date: str, end_date: str):

    vix_data = yf.download('^VIX', start=start_date, end=end_date)
    return vix_data



# data = fetch_stock_data(ticker, start_date, end_date)
# data2 = fetch_vix_data(start_date, end_date)

# print(data.head())
# df = pd.DataFrame(data)
# df2 = pd.DataFrame(data2)
# df["Vix_Close"]=df2["Close"]
# df



def calculate_rsi(data, period=14):

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# df['RSI']=calculate_rsi(df,14)
# df.iloc[:40]


def calculate_macd(data, short_window=12, long_window=26, signal_window=9):

    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal


# df['MACD'],df["Signal"]=calculate_macd(df)

# df

def calculate_ema(data, period=20):

    ema = data['Close'].ewm(span=period, adjust=False).mean()
    return ema

def fisher_transform(series, period=9):

    min_low = series.rolling(window=period).min()
    max_high = series.rolling(window=period).max()
    value = 2 * ((series - min_low) / (max_high - min_low) - 0.5)

    fisher = np.zeros(len(value))  # Use NumPy array for better efficiency
    for i in range(1, len(value)):
        if np.isnan(value[i]):  # Skip NaN values at the beginning
            continue
        fisher[i] = 0.5 * np.log((1 + value[i]) / (1 - value[i])) + 0.5 * fisher[i - 1]

    return pd.Series(fisher, index=series.index)


# df['Fisher']=fisher_transform(df['Close'])

# df['EMA_20']=calculate_ema(df,20)
# df["EMA_50"]=calculate_ema(df,50)
# df.fillna(0,inplace=True)
# df

def calculate_support_resistance(data, window=20):

    data['Support'] = data['Low'].rolling(window=window).min()
    data['Resistance'] = data['High'].rolling(window=window).max()
    return data

df = calculate_support_resistance(df)
df.fillna(0,inplace=True)
df

def calculate_profit_loss(data):

    data['P&L%'] = data['Close'].pct_change() * 100
    return data

# calculate_profit_loss(df)
# df.fillna(0,inplace=True)
# df

def calculate_atr(data, period=14):

    high = data['High'].values
    low = data['Low'].values
    close = np.roll(data['Close'].values, 1)
    close[0] = close[1]  # Fix the first element after roll

    # Calculate the three different methods for true range
    tr1 = high - low
    tr2 = np.abs(high - close)
    tr3 = np.abs(low - close)

    # Find the maximum of the three methods for each day
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # Calculate ATR using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period

    return atr

# df['ATR']=calculate_atr(df)
# df

def calculate_adx(data, period=14):

    # Convert to numpy arrays for faster computation
    high = data['High'].values
    low = data['Low'].values

    # Calculate True Range
    tr = calculate_atr(data, period)

    # Calculate high and low differences
    high_diff = np.zeros_like(high)
    low_diff = np.zeros_like(low)

    # Calculate differences manually (equivalent to np.diff with proper handling of first element)
    for i in range(1, len(high)):
        high_diff[i] = high[i] - high[i-1]
        low_diff[i] = low[i-1] - low[i]

    # Initialize +DM and -DM arrays
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(low)

    # Calculate +DM and -DM values
    for i in range(1, len(high)):
        # +DM occurs when current high - previous high > previous low - current low
        # -DM occurs when previous low - current low > current high - previous high
        if high_diff[i] > low_diff[i] and high_diff[i] > 0:
            plus_dm[i] = high_diff[i]
        else:
            plus_dm[i] = 0

        if low_diff[i] > high_diff[i] and low_diff[i] > 0:
            minus_dm[i] = low_diff[i]
        else:
            minus_dm[i] = 0

    # Smooth +DM, -DM and TR using Wilder's smoothing technique
    smoothed_plus_dm = np.zeros_like(plus_dm)
    smoothed_minus_dm = np.zeros_like(minus_dm)
    smoothed_tr = np.zeros_like(tr)

    # Initialize the first value with simple average
    if len(plus_dm) >= period:
        smoothed_plus_dm[period-1] = np.sum(plus_dm[0:period])
        smoothed_minus_dm[period-1] = np.sum(minus_dm[0:period])
        smoothed_tr[period-1] = np.sum(tr[0:period])

        # Apply Wilder's smoothing
        for i in range(period, len(plus_dm)):
            smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1]/period) + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1]/period) + minus_dm[i]
            smoothed_tr[i] = smoothed_tr[i-1] - (smoothed_tr[i-1]/period) + tr[i]

    # Calculate +DI and -DI
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(low)

    # Avoid division by zero
    nonzero_indices = smoothed_tr > 0
    plus_di[nonzero_indices] = 100 * (smoothed_plus_dm[nonzero_indices] / smoothed_tr[nonzero_indices])
    minus_di[nonzero_indices] = 100 * (smoothed_minus_dm[nonzero_indices] / smoothed_tr[nonzero_indices])

    # Calculate DX
    dx = np.zeros_like(high)
    sum_di = plus_di + minus_di
    nonzero_sum = sum_di > 0
    dx[nonzero_sum] = 100 * np.abs(plus_di[nonzero_sum] - minus_di[nonzero_sum]) / sum_di[nonzero_sum]

    # Calculate ADX with Wilder's smoothing
    adx = np.zeros_like(high)

    # Initialize ADX with simple average of DX
    if len(dx) >= 2*period:
        adx[2*period-1] = np.mean(dx[period:2*period])

        # Apply Wilder's smoothing to calculate ADX
        for i in range(2*period, len(adx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx, plus_di, minus_di

# Calculate ADX, +DI, and -DI
# adx_values, plus_di_values, minus_di_values = calculate_adx(df)
# df['ADX'] = adx_values
# df['Plus_DI'] = plus_di_values
# df['Minus_DI'] = minus_di_values

# # Fill any NaN values
# df.fillna(0, inplace=True)
# df

print("DataFrame index type:", type(df.index))
if isinstance(df.index, pd.MultiIndex):
    print("MultiIndex levels:", df.index.nlevels)
    print("Level names:", df.index.names)

def plot_indicators(data, indicators, start_date=None, end_date=None):

    if start_date and end_date:
        data = data.loc[start_date:end_date]

    plt.figure(figsize=(12, 6))
    for indicator in indicators:
        if indicator in data.columns:
            plt.plot(data.index, data[indicator], label=indicator)

    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Stock Indicators")
    plt.grid()
    plt.show()

# plot_indicators(df,plot_ind,ind_begin,ind_end)
# plot_indicators(df,['MACD'],"2022-01-01","2023-10-01")

def plot_indicator_correlation(data, main_indicator):


    data.columns = data.columns.droplevel(1)  # Drops the second level ('Ticker')
    correlation = data.corr()[main_indicator].drop(main_indicator)
    # correlation.sort_values(ascending=False, inplace=True)
    correlation = correlation.sort_values( ascending=False)


    plt.figure(figsize=(10, 5))
    plt.bar(correlation.index, correlation.values, color='skyblue')
    plt.xlabel("Indicators")
    plt.ylabel("Correlation")
    plt.title(f"Correlation of {main_indicator} with Other Indicators")
    plt.xticks(rotation=45)
    plt.show()

    return correlation




# df.columns

# sample1 = df.copy()
# plot_indicator_correlation(sample1,corr_val)

# df.columns

# date_hold = pd.DataFrame()
# date_hold = df['Date']
# df.drop(columns=["Date"],inplace=True)
# df.head()
# index_values = pd.DataFrame(df.index.tolist())            #In this dataframe index are of date so we are storing them for future use

# df.reset_index(drop=True, inplace=True)     #Resetting the index which was date to integer
# df



df.shape

def scale_data(data):

    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return scaled_data,scaler


def inverse_scale_data(scaled_data, scaler):
    inverse_data = pd.DataFrame(scaler.inverse_transform(scaled_data), columns=scaled_data.columns, index=scaled_data.index)
    return inverse_data

scaled_data,scaler = scale_data(df)
scaled_2d = np.array(scaled_data)
scaled_2d_label = np.array(scaled_data['Close'])
inverse_scale_data(scaled_data,scaler)
scaled_data["P&L%"].max()

scaled_data

y =df['Close']
scaled_close,closed_scaler=scale_data(y)
closed_scaler

import numpy as np

def create_sequences(data,Predictor, sequence_length=60):

    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i+sequence_length].values)
        labels.append(data.iloc[i+sequence_length][Predictor])
    return np.array(sequences), np.array(labels)
    # return pd.DataFrame(sequences[-1]),pd.DataFrame(labels[-1])




sequence,label = create_sequences(scaled_data,Predictor,seq_len)
# sequence,label

scaled_2d=scaled_2d[-(scaled_2d.shape[0]-seq_len):]
scaled_2d_label=scaled_2d_label[-(scaled_2d.shape[0]):]
scaled_2d.shape,sequence.shape,label.shape,scaled_2d_label.shape

def train_test_val_split(sequences, labels, test_size=0.2, val_size=0.1):


    Xgb_train, Xgb_test, Ygb_train, Ygb_test = train_test_split(scaled_2d, scaled_2d_label, test_size=test_size, shuffle=False,random_state=42)
    Xgb_train, Xgb_val, Ygb_train, Ygb_val = train_test_split(Xgb_train, Ygb_train, test_size=val_size, shuffle=False,random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, shuffle=False,random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False,random_state=42)
    return X_train, X_test, X_val, y_train, y_test, y_val,Xgb_train, Xgb_test, Xgb_val, Ygb_train, Ygb_test, Ygb_val


X_train, X_test, X_val, y_train, y_test, y_val,Xgb_train, Xgb_test, Xgb_val, Ygb_train, Ygb_test, Ygb_val= train_test_val_split(sequence, label,test_size=test_size)
X_train.shape, X_test.shape, X_val.shape, y_train.shape, y_test.shape, y_val.shape,Xgb_train.shape, Xgb_test.shape, Xgb_val.shape, Ygb_train.shape, Ygb_test.shape, Ygb_val.shape

def build_lstm_model(input_shape):

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru_model(input_shape):

    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def build_xgboost_model():

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    return model

def build_bidirectional_lstm_model(input_shape):

    model = Sequential([
        # First bidirectional LSTM layer
        Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),

        # Second bidirectional LSTM layer
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),

        # Dense layers

        Dense(25, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Use Adam optimizer with a smaller learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse')
    return model

def build_bidirectional_gru_model(input_shape):

    model = Sequential([
        # First bidirectional GRU layer
        Bidirectional(GRU(50, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),

        # Second bidirectional GRU layer
        Bidirectional(GRU(50, return_sequences=False)),
        Dropout(0.2),

        # Dense layers

        Dense(25, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Use Adam optimizer with a smaller learning rate
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',patience=10,restore_best_weights=True
    )

    # Model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        model_path,monitor='val_loss',save_best_only=True, mode='min', verbose=1
    )

    # Learning rate reducer
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=0.0001,verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,validation_data=(X_val, y_val),epochs=50,batch_size=32,callbacks=[early_stopping, checkpoint, reduce_lr],verbose=1,shuffle=False
    )

    return history


def train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

    return train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val, model_path)

def train_lstm_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint], verbose=1)


def train_gru_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint],verbose=1)


def train_xgboost_on_residuals(lstm_model, X_train, y_train, X_test, y_test,varient):

    if varient == 1:
      lstm_predictions_train = lstm_model.predict(X_train)
      lstm_predictions_test = lstm_model.predict(X_test)
      residuals_train = y_train - (lstm_predictions_train )
      print(residuals_train.shape)
      residuals_test = y_test - (lstm_predictions_test)

    elif varient == 2:
      gru_predictions_train = lstm_model.predict(X_train)
      gru_predictions_test = lstm_model.predict(X_test)
      residuals_train = y_train - (gru_predictions_train )
      print(residuals_train.shape)
      residuals_test = y_test - (gru_predictions_test)

    # return residuals_train.shape,residuals_test.shape,y_train.shape,y_test.shape

    xgb_model = build_xgboost_model()
    xgb_model.fit(Xgb_train, residuals_train)
    joblib.dump(xgb_model, 'xgb_model.pkl')  # Save the XGBoost model

    return xgb_model, residuals_test

Xgb_train.shape

# train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test)


def load_best_model(model_path):

    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    else:
        print(f"Model file {model_path} not found.")
        return None



# if trainable:
#     model = 0
#     if model_controller == 1:
#         model = build_lstm_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_lstm_model(model, X_train, y_train, X_val, y_val)

#     elif model_controller == 2:
#         model = build_gru_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_gru_model(model, X_train, y_train, X_val, y_val)

#     elif model_controller == 3:
#         model = build_lstm_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_lstm_model(model, X_train, y_train, X_val, y_val)
#         xgb_model, residuals_test = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 1)

#     elif model_controller == 4:
#         model = build_gru_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_gru_model(model, X_train, y_train, X_val, y_val)
#         xgb_model, residuals_test = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 2)

#     elif model_controller == 5:
#         model = build_bidirectional_lstm_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val)

#     elif model_controller == 6:
#         model = build_bidirectional_gru_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val)

#     elif model_controller == 7:
#         model = build_bidirectional_lstm_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val)
#         xgb_model, residuals_test = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 5)

#     elif model_controller == 8:
#         model = build_bidirectional_gru_model((seq_len, X_train.shape[2]))
#         model.summary()
#         train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val)
#         xgb_model, residuals_test = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 6)

# if model_controller == 1:
#     model = load_best_model(model_path)
# elif model_controller == 2:
#     model = load_best_model(model_path)
# elif model_controller == 3:
#     model = load_best_model(model_path)
#     xgb_model = joblib.load('xgb_model.pkl')
# elif model_controller == 4:
#     model = load_best_model(model_path)
#     xgb_model = joblib.load('xgb_model.pkl')
# elif model_controller == 5:
#     model = load_best_model(model_path)
# elif model_controller == 6:
#     model = load_best_model(model_path)
# elif model_controller == 7:
#     model = load_best_model(model_path)
#     xgb_model = joblib.load('xgb_model.pkl')
# elif model_controller == 8:
#     model = load_best_model(model_path)
#     xgb_model = joblib.load('xgb_model.pkl')

X_test.shape,y_test.shape

Xgb_test[-1:].shape

def calculate_rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))

def evaluate_model(model, X_test, y_test, model_controller):

    if model_controller in [1, 2, 5, 6]:
        # Standard models without XGBoost
        predictions = model.predict(X_test)
    elif model_controller in [3, 4, 7, 8]:
        # Hybrid models with XGBoost
        predictions = model.predict(X_test) + xgb_model.predict(Xgb_test[-1:])

    predictions1=inverse_scale_data(pd.DataFrame(predictions),closed_scaler)
    y_test1=inverse_scale_data(pd.DataFrame(y_test),closed_scaler)

    mse = mean_squared_error(y_test1, predictions1)
    mae = mean_absolute_error(y_test1, predictions1)
    r2 = r2_score(y_test1, predictions1)
    rmse = calculate_rmse(y_test1, predictions1)
    mape = calculate_mape(y_test1, predictions1)


    return mse, mae, r2, rmse,mape,predictions

def calculate_rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

if model_controller == 1:
    mse, mae, r2 ,rmse,mape,predictions= evaluate_model(model, X_test, y_test,model_controller)
    print(f"LSTM Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2}, RMSE={rmse}, MAPE={mape}")

elif model_controller==2:
    mse, mae, r2 ,rmse,mape,predictions= evaluate_model(model, X_test, y_test,model_controller)
    print(f"GRU Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller==3:
    mse, mae, r2 ,rmse,mape,predictions= evaluate_model(model, X_test, y_test,model_controller)
    print(f"LSTM + Xgboost Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller==4:
    mse, mae, r2 ,rmse,mape,predictions= evaluate_model(model, X_test, y_test,model_controller)
    print(f"GRU + Xgboost Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller == 5:
    mse, mae, r2 ,rmse,mape,predictions = evaluate_model(model, X_test, y_test, model_controller)
    print(f"Bidirectional LSTM Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller == 6:
    mse, mae, r2 ,rmse,mape,predictions = evaluate_model(model, X_test, y_test, model_controller)
    print(f"Bidirectional GRU Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller == 7:
    mse, mae, r2 ,rmse,mape,predictions = evaluate_model(model, X_test, y_test, model_controller)
    print(f"Bidirectional LSTM + XGBoost Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")

elif model_controller == 8:
    mse, mae, r2 ,rmse,mape,predictions = evaluate_model(model, X_test, y_test, model_controller)
    print(f"Bidirectional GRU + XGBoost Model Performance: MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")








def directional_accuracy(y_true, y_pred, horizon=1, threshold=0.0):

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)} elements")

    if len(y_true) <= horizon:
        raise ValueError(f"Input arrays must have more than {horizon} elements")

    # Calculate true and predicted directions
    true_diff = y_true[horizon:] - y_true[:-horizon]
    pred_diff = y_pred[horizon:] - y_pred[:-horizon]

    # Apply threshold filter
    true_directions = np.where(np.abs(true_diff) > threshold, np.sign(true_diff), 0)
    pred_directions = np.where(np.abs(pred_diff) > threshold, np.sign(pred_diff), 0)

    # Count correct predictions
    correct_predictions = (true_directions == pred_directions)
    accuracy = np.sum(correct_predictions) / len(true_directions) * 100

    # Calculate direction-specific metrics
    up_days = np.sum(true_directions > 0)
    down_days = np.sum(true_directions < 0)
    flat_days = np.sum(true_directions == 0)

    # Calculate accuracy for up and down days separately
    up_correct = np.sum((true_directions > 0) & correct_predictions)
    down_correct = np.sum((true_directions < 0) & correct_predictions)
    flat_correct = np.sum((true_directions == 0) & correct_predictions)

    up_accuracy = (up_correct / up_days * 100) if up_days > 0 else 0
    down_accuracy = (down_correct / down_days * 100) if down_days > 0 else 0
    flat_accuracy = (flat_correct / flat_days * 100) if flat_days > 0 else 0

    # Compile detailed metrics
    metrics = {
        'accuracy': accuracy,
        'up_days': int(up_days),
        'down_days': int(down_days),
        'flat_days': int(flat_days),
        'up_accuracy': up_accuracy,
        'down_accuracy': down_accuracy,
        'flat_accuracy': flat_accuracy,
        'correct_predictions': int(np.sum(correct_predictions)),
        'total_predictions': len(true_directions)
    }

    print(f"Directional Accuracy: {accuracy:.2f}%")
    print(f"Up Days: {up_days} (Accuracy: {up_accuracy:.2f}%)")
    print(f"Down Days: {down_days} (Accuracy: {down_accuracy:.2f}%)")

    return accuracy, metrics

true = np.array(scaled_data["Close"])[-int(test_size * (len(scaled_data)-seq_len))-1:]

# Basic usage
accuracy, metrics = directional_accuracy(true, predictions)

# With custom parameters - for 5-day horizon and ignoring movements less than 1%
accuracy, metrics = directional_accuracy(true, predictions, horizon=20, threshold=0.01)

# Access specific metrics
print(f"Up movement accuracy: {metrics['up_accuracy']:.2f}%")
print(f"Down movement accuracy: {metrics['down_accuracy']:.2f}%")

yop = pd.DataFrame(model.predict(X_test))
pre=inverse_scale_data(yop, closed_scaler)

k = pd.DataFrame(y_test)
act=inverse_scale_data(k, closed_scaler)

plt.plot(pre)
plt.plot(act)

def simulate_trading_returns_for_daily_predictions_fixed(y_true, y_pred, prices,
                                             holding_period=20, prediction_threshold=0.005,
                                             initial_capital=100000, position_size=0.1,
                                             stop_loss=0.07, take_profit=0.15, commission=0.001,
                                             slippage=0.001, risk_free_rate=0.02, verbose=True):

    import numpy as np

    # Input validation
    if len(y_true) != len(y_pred) or len(y_true) != len(prices):
        raise ValueError("All input arrays must have the same length")


    prices = np.array(prices, dtype=float)


    capital = float(initial_capital)
    holdings = 0
    trades = []
    positions = []
    daily_returns = []
    daily_capital = [float(initial_capital)]


    signals = np.zeros(len(y_pred))


    def log(message):
        if verbose:
            print(message)

    log(f"Generating trading signals based on predictions...")


    for i in range(len(y_pred) - 2):

        if y_pred[i] > y_true[i-1] + prediction_threshold:
            signals[i] = 1
        elif y_pred[i] < y_true[i-1] - prediction_threshold:
            signals[i] = -1

    log(f"Generated {np.sum(np.abs(signals) > 0)} trading signals ({np.sum(signals > 0)} buy, {np.sum(signals < 0)} sell)")


    active_positions = []


    log(f"Starting simulation with ₹{initial_capital:.2f} capital...")

    for i in range(len(prices)):
        current_price = float(prices[i])


        new_active_positions = []
        for pos in active_positions:
            entry_price, pos_size, stop_price, tp_price, entry_day = pos
            days_held = i - entry_day


            close_position = False
            close_reason = ""

            if current_price <= stop_price:
                close_position = True
                close_reason = "stop_loss"
                log(f"Day {i}: Stop loss triggered at ₹{current_price:.2f} (entry: ₹{entry_price:.2f})")
            elif tp_price is not None and current_price >= tp_price:
                close_position = True
                close_reason = "take_profit"
                log(f"Day {i}: Take profit triggered at ₹{current_price:.2f} (entry: ₹{entry_price:.2f})")
            elif days_held >= holding_period:
                close_position = True
                close_reason = "holding_period_reached"
                log(f"Day {i}: Holding period reached, closing at ₹{current_price:.2f} (entry: ₹{entry_price:.2f})")

            if close_position:

                sale_proceeds = pos_size * current_price * (1 - commission - slippage)
                capital += sale_proceeds
                holdings -= pos_size
                profit_pct = (current_price / entry_price - 1) * 100
                entry_cost = pos_size * entry_price * (1 + commission + slippage)
                profit_amount = sale_proceeds - entry_cost

                trades.append({
                    'day': i,
                    'type': close_reason,
                    'price': current_price,
                    'entry_price': entry_price,
                    'size': pos_size,
                    'profit_pct': profit_pct,
                    'profit_amount': profit_amount,
                    'days_held': days_held,
                    'sale_proceeds': sale_proceeds,
                    'entry_cost': entry_cost
                })
            else:
                new_active_positions.append(pos)

        active_positions = new_active_positions


        if i < len(signals):

            if signals[i] > 0 and capital > 1000:
                trade_value = min(capital * position_size, capital * 0.95)


                if trade_value >= 1000:

                    shares_to_buy = trade_value / (current_price * (1 + commission + slippage))
                    actual_trade_cost = shares_to_buy * current_price * (1 + commission + slippage)


                    if actual_trade_cost <= capital:
                        stop_price = current_price * (1 - stop_loss)
                        tp_price = None if take_profit is None else current_price * (1 + take_profit)

                        capital -= actual_trade_cost
                        holdings += shares_to_buy
                        active_positions.append([current_price, shares_to_buy, stop_price, tp_price, i])

                        log(f"Day {i}: BUY at ₹{current_price:.2f}, shares: {shares_to_buy:.2f}, " +
                           f"cost: ₹{actual_trade_cost:.2f}, stop: ₹{stop_price:.2f}")

                        trades.append({
                            'day': i,
                            'type': 'buy',
                            'price': current_price,
                            'size': shares_to_buy,
                            'cost': actual_trade_cost,
                            'stop_price': stop_price,
                            'take_profit_price': tp_price
                        })


        holdings_value = holdings * current_price
        day_end_capital = float(capital) + holdings_value


        if i < len(daily_capital):
            daily_capital[i] = day_end_capital
        else:
            daily_capital.append(day_end_capital)


        positions.append({
            'day': i,
            'cash': capital,
            'holdings': holdings,
            'holdings_value': holdings_value,
            'total_value': day_end_capital,
            'price': current_price
        })


        if i > 0:
            daily_return = (daily_capital[i] / daily_capital[i-1]) - 1
            daily_returns.append(daily_return)


    if holdings > 0 and len(prices) > 0:
        final_price = float(prices[-1])
        liquidation_value = holdings * final_price * (1 - commission - slippage)


        log(f"End of simulation: Liquidating {float(holdings):.2f} shares at ₹{final_price:.2f} for ₹{liquidation_value:.2f}")

        capital += liquidation_value
        holdings = 0

        final_capital = capital
        daily_capital[-1] = capital
    else:
        final_capital = daily_capital[-1] if daily_capital else initial_capital


    total_return = (final_capital / initial_capital - 1) * 100


    sharpe = 0
    sortino = 0
    max_drawdown = 0
    win_rate = 0

    if len(daily_returns) > 1:
        daily_returns_array = np.array(daily_returns, dtype=float)
        daily_capital_array = np.array(daily_capital, dtype=float)


        volatility = daily_returns_array.std() * np.sqrt(252)


        excess_returns = daily_returns_array - risk_free_rate/252
        sharpe = (excess_returns.mean() / daily_returns_array.std()) * np.sqrt(252) if daily_returns_array.std() > 0 else 0


        downside_returns = daily_returns_array[daily_returns_array < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            sortino = (excess_returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0


        peak = np.maximum.accumulate(daily_capital_array)
        drawdown = (peak - daily_capital_array) / peak
        max_drawdown = drawdown.max() * 100


    profitable_trades = [t for t in trades if 'profit_pct' in t and t['profit_pct'] > 0]
    completed_trades = [t for t in trades if 'profit_pct' in t]

    if len(completed_trades) > 0:
        win_rate = len(profitable_trades) / len(completed_trades) * 100
    else:
        win_rate = 0
        if verbose:
            print("Warning: No completed trades in simulation - win rate is 0%")


    if len(completed_trades) > 0:
        avg_profit_per_trade = sum([t['profit_amount'] for t in completed_trades]) / len(completed_trades)


        gross_profits = sum([t['profit_amount'] for t in completed_trades if t['profit_amount'] > 0])
        gross_losses = abs(sum([t['profit_amount'] for t in completed_trades if t['profit_amount'] < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    else:
        avg_profit_per_trade = 0
        profit_factor = 0


    holding_periods = [t['days_held'] for t in completed_trades] if completed_trades else [0]
    avg_holding_period = sum(holding_periods) / len(holding_periods) if len(holding_periods) > 0 else 0


    metrics = {
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': total_return,
        'number_of_trades': len(trades),
        'completed_trades': len(completed_trades),
        'avg_holding_period': avg_holding_period,
        'avg_profit_per_trade': avg_profit_per_trade,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown_pct': max_drawdown,
        'win_rate': win_rate,
        'trades': trades,
        'daily_capital': daily_capital,
        'positions': positions
    }


    if verbose:
        print(f"\nTrading Simulation Results:")
        print(f"Initial Capital: ₹{initial_capital:,.2f}")
        print(f"Final Capital: ₹{final_capital:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Number of Trades: {len(trades)}")
        print(f"Completed Trades: {len(completed_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Profit per Trade: ₹{avg_profit_per_trade:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Holding Period: {avg_holding_period:.1f} days")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Sortino Ratio: {sortino:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Total Trading Days: {len(prices)}")

    return metrics

def split_time_series_array(series, ratio=1-test_size):


    if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
        series = series.squeeze()

    split_index = int(len(series) * ratio)
    arr1 = series.iloc[:split_index].to_numpy()
    arr2 = series.iloc[split_index:].to_numpy()
    return arr1, arr2

train_Date,test_Date=split_time_series_array(index_values)
test_Date.shape





def get_original_prices(scaled_values, scaler, column_idx=0):


    dummy = np.zeros((len(scaled_values), scaler.scale_.shape[0]))


    dummy[:, column_idx] = scaled_values

    original_data = scaler.inverse_transform(dummy)

    return original_data[:, column_idx]




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta



def plot_trading_results_improved(metrics, title="Trading Simulation Results", figsize=(14, 12)):


    dates = test_Date.flatten().tolist()


    if len(dates) > 0 and not isinstance(dates[0], pd.Timestamp):
        dates = [pd.to_datetime(date) if not pd.isna(date) else None for date in dates]
        dates = [d for d in dates if d is not None]


    portfolio_values = metrics['daily_capital']


    min_length = min(len(dates), len(portfolio_values))
    dates = dates[:min_length]
    portfolio_values = portfolio_values[:min_length]


    df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Value': portfolio_values
    })


    df['Peak'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = 100 * (df['Peak'] - df['Portfolio_Value']) / df['Peak']


    df['Daily_Return'] = df['Portfolio_Value'].pct_change() * 100


    fig = plt.figure(figsize=figsize)


    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['Date'], df['Portfolio_Value'], label='Portfolio Value', linewidth=2)
    ax1.set_title(f"{title} - Equity Curve", fontsize=14)
    ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')


    if 'trades' in metrics and len(metrics['trades']) > 0:

        first_trade_day = min([t['day'] for t in metrics['trades']])


        for trade in metrics['trades']:
            day_idx = trade['day']
            if day_idx < len(df):
                marker_date = df['Date'].iloc[day_idx]

                if trade['type'] == 'buy':
                    ax1.axvline(x=marker_date, color='green', alpha=0.3, linestyle='--')
                elif trade['type'] in ['sell', 'stop_loss', 'take_profit', 'holding_period_reached']:
                    ax1.axvline(x=marker_date, color='red', alpha=0.3, linestyle='--')


    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)


    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.fill_between(df['Date'], df['Drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
    ax2.plot(df['Date'], df['Drawdown'], color='red', linewidth=1)
    ax2.set_title('Drawdown Analysis', fontsize=14)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_ylim(bottom=0, top=max(df['Drawdown'])*1.1)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3)


    max_dd = df['Drawdown'].max()
    max_dd_date = df.loc[df['Drawdown'].idxmax(), 'Date']
    ax2.axhline(y=max_dd, color='darkred', linestyle='--', alpha=0.8)
    ax2.text(df['Date'].iloc[0], max_dd, f'Max Drawdown: {max_dd:.2f}%',
             verticalalignment='bottom', horizontalalignment='left', color='darkred')


    ax3 = plt.subplot(3, 1, 3)

    if 'trades' in metrics and len(metrics['trades']) > 0:

        profit_trades = [t for t in metrics['trades'] if 'profit_pct' in t]

        if len(profit_trades) > 0:

            win_trades = [t for t in profit_trades if t['profit_pct'] > 0]
            loss_trades = [t for t in profit_trades if t['profit_pct'] <= 0]
            win_rate = len(win_trades) / len(profit_trades) if len(profit_trades) > 0 else 0


            profits = [t['profit_pct'] for t in profit_trades]
            ax3.hist(profits, bins=20, alpha=0.7, color='skyblue', label='Profit/Loss Distribution')
            ax3.axvline(x=0, color='black', linestyle='--')
            ax3.set_title('Trade Profit/Loss Distribution', fontsize=14)
            ax3.set_xlabel('Profit/Loss (%)', fontsize=12)
            ax3.set_ylabel('Number of Trades', fontsize=12)
            ax3.grid(True, alpha=0.3)


            stats_text = (f"Win Rate: {win_rate:.2%}\n"
                         f"Total Trades: {len(profit_trades)}\n"
                         f"Avg Win: {sum([t['profit_pct'] for t in win_trades])/len(win_trades):.2f}% (n={len(win_trades)})\n"
                         f"Avg Loss: {sum([t['profit_pct'] for t in loss_trades])/len(loss_trades):.2f}% (n={len(loss_trades)})" if len(loss_trades) > 0 else "No losses")


            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, "No trades with profit data available",
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax3.transAxes, fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No trade data available",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax3.transAxes, fontsize=12)


    if 'total_return_pct' in metrics and 'sharpe_ratio' in metrics:
        performance_text = (f"Total Return: {metrics['total_return_pct']:.2f}%  |  "
                           f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}  |  "
                           f"Max Drawdown: {max_dd:.2f}%")
        fig.text(0.5, 0.97, performance_text, horizontalalignment='center',
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)

    if 'trades' in metrics and len(metrics['trades']) > 0:
        profit_trades = [t for t in metrics['trades'] if 'profit_pct' in t and 'days_held' in t]

        if len(profit_trades) > 5:
            fig2 = plt.figure(figsize=(14, 6))


            ax4 = plt.subplot(1, 2, 1)
            days_held = [t['days_held'] for t in profit_trades]
            profits = [t['profit_pct'] for t in profit_trades]

            ax4.scatter(days_held, profits, alpha=0.7, c=profits, cmap='coolwarm')
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Holding Period vs Profit', fontsize=14)
            ax4.set_xlabel('Days Held', fontsize=12)
            ax4.set_ylabel('Profit/Loss (%)', fontsize=12)
            ax4.grid(True, alpha=0.3)


            if len(df) > 30:
                ax5 = plt.subplot(1, 2, 2)


                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                monthly_returns = df.groupby(['Year', 'Month'])['Portfolio_Value'].apply(
                    lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
                ).reset_index()


                if len(monthly_returns) > 1:
                    try:
                        pivot_table = monthly_returns.pivot(index='Year', columns='Month', values='Portfolio_Value')


                        im = ax5.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto')
                        ax5.set_title('Monthly Returns (%)', fontsize=14)


                        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        ax5.set_xticks(np.arange(len(months)))
                        ax5.set_xticklabels(months)


                        ax5.set_yticks(np.arange(len(pivot_table.index)))
                        ax5.set_yticklabels(pivot_table.index)


                        cbar = plt.colorbar(im, ax=ax5)
                        cbar.set_label('Return (%)')


                        for i in range(len(pivot_table.index)):
                            for j in range(len(pivot_table.columns)):
                                if not np.isnan(pivot_table.values[i, j]):
                                    text = ax5.text(j, i, f"{pivot_table.values[i, j]:.1f}",
                                                  ha="center", va="center", color="black", fontsize=8)
                    except:
                        ax5.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax5.transAxes, fontsize=12)
                else:
                    ax5.text(0.5, 0.5, "Insufficient data for monthly heatmap",
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax5.transAxes, fontsize=12)
            else:
                ax5 = plt.subplot(1, 2, 2)
                ax5.text(0.5, 0.5, "Insufficient data for monthly analysis",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax5.transAxes, fontsize=12)

            plt.tight_layout()

    plt.show()
    return fig



true.shape,predictions.shape



original_y_true = get_original_prices(true, scaler, df.columns.get_loc('Close'))
original_y_pred = get_original_prices(predictions, scaler, df.columns.get_loc('Close'))
original_prices = original_y_true


results = simulate_trading_returns_for_daily_predictions_fixed(
    y_true=true,              # Scaled actual values
    y_pred=predictions,       # Scaled predictions
    prices=original_y_true,   # Original market prices
    holding_period=100,        # Shorter holding period
    prediction_threshold=0.005, # 0.5% threshold
    initial_capital=100000,   # Starting capital
    position_size=0.5,       # Smaller position size (5% of capital)
    stop_loss=0.1,           # 5% stop loss
    take_profit=0.30,          # 10% take profit
    verbose=True              # Print detailed logs
)
# Access the results
print("Total no of Trading days",len(y_test))
print(f"Total return: {results['total_return_pct']:.2f}%")
print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")





plot_trading_results_improved(results, "Trading Simulation with Correct Dates")

list(df.columns)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.dates as mdates

def forecast_n_days_scaled(model, scaled_data, index_values, n_days=30, seq_len=60):


    last_date = index_values.iloc[-1, 0]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)


    future_dates = []
    date = last_date
    days_added = 0

    while days_added < n_days:
        date = date + timedelta(days=1)
        if date.weekday() < 5:
            future_dates.append(date)
            days_added += 1


    sequence = scaled_data.iloc[-seq_len:].values


    close_idx = scaled_data.columns.get_loc('Close')


    scaled_predictions = []


    for _ in range(n_days):

        x_input = sequence.reshape(1, seq_len, sequence.shape[1])


        pred = model.predict(x_input, verbose=0)[0][0]
        scaled_predictions.append(pred)


        new_row = sequence[-1].copy()
        new_row[close_idx] = pred
        sequence = np.vstack([sequence[1:], new_row])


    result_df = pd.DataFrame()
    result_df['Date'] = future_dates
    result_df['Scaled_Prediction'] = scaled_predictions

    return result_df

def plot_scaled_forecast(df, scaled_data, index_values, forecast_df, days_to_show=60):


    hist_dates = index_values.iloc[-days_to_show:, 0].values


    hist_scaled_close = scaled_data['Close'][-days_to_show:].values

    forecast_dates = forecast_df['Date']
    forecast_values = forecast_df['Scaled_Prediction']

    plt.figure(figsize=(12, 6))


    plt.plot(hist_dates, hist_scaled_close, color='blue', label='Historical (Scaled)')


    plt.plot(forecast_dates, forecast_values, color='red', linestyle='--', label='Forecast (Scaled)')


    plt.axvline(x=hist_dates[-1], color='green', linestyle='-', alpha=0.5,
                label='Forecast Start')


    plt.title('Stock Price Forecast (MinMax Scaled)')
    plt.xlabel('Date')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True, alpha=0.3)


    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()




scaled_forecast = forecast_n_days_scaled(
    model=model,
    scaled_data=scaled_data,  # Use the scaled DataFrame for prediction
    index_values=index_values,  # DataFrame with the dates
    n_days=30,
    seq_len=60
)


print(scaled_forecast)

plot_scaled_forecast(df, scaled_data, index_values, scaled_forecast)