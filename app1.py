# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
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
from datetime import date
import time
import requests

# === SESSION STATE INITIALIZATION ===
if 'model' not in st.session_state:
    st.session_state.model = None
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'forecast_df' not in st.session_state:
    st.session_state.forecast_df = None

# === SIDEBAR ===
st.sidebar.title("Model Controls")

ticker = st.sidebar.text_input("Ticker Symbol", value="MSFT")
start_date = st.sidebar.date_input("📅 Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("📅 End Date", value=date.today().strftime("%Y-%m-%d"))
model_controller = st.sidebar.selectbox("🧠 Model Type", options=list(range(1, 9)), format_func=lambda x: {
    1: "LSTM", 2: "GRU", 3: "LSTM + XGBoost", 4: "GRU + XGBoost",
    5: "Bi-LSTM", 6: "Bi-GRU", 7: "Bi-LSTM + XGBoost", 8: "Bi-GRU + XGBoost"
}[x])
model_path = st.sidebar.text_input("Name of model",value="msft_gru2.keras")
trainable = st.sidebar.checkbox("Train Model (If not selected pretrained model will be used for prediction)", value=False)
seq_len = st.sidebar.slider("Sequence Length", min_value=30, max_value=150, value=60,step=10)
test_size = st.sidebar.slider("Test Size", min_value=0.01, max_value=0.9, value=0.2,step=0.05)
plot_indicators = st.sidebar.multiselect("Indicators to Plot", ["Close", "EMA_20", "MACD", "RSI","Vix_Close","Support","Resistance"], default=["Close"])
st.title("Stock Price Prediction & Trading Strategy")
tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Dashboard", "🎯 Accuracy Metrics","📈 Trading Simulation","📰 Sentiments","🧭 Forecasting"])
if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = 0
# === FETCH AND PROCESS DATA ===
if "ckecker" not in st.session_state:
    st.session_state.ckecker = 0
if "ckecker2" not in st.session_state:
    st.session_state.ckecker2 = 0
with tab1:
    if st.button("Run Model"):
        if trainable:
            st.success("Using Newly Trained Model")
        else:
            st.success("Using Pre Saved Model")
        st.session_state.ckecker = 1
        # === PLACEHOLDER: Add seed setting ===
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
        set_all_seeds()

        # === PLACEHOLDER: Add data fetch logic ===
        def fetch_stock_data1(ticker: str, start_date: str, end_date: str):

            stock_data = yf.download(ticker, start=start_date, end=end_date)
            # stock_data1 = yf.download(ticker, start=start_date, end=end_date)
            return stock_data
        
        # Newly added code
        def fetch_stock_data2(ticker: str, start_date: str, end_date: str):
            for attempt in range(5):
                try:
                    return yf.download(ticker, start=start_date, end=end_date, progress=False)
                except Exception as e:
                    print(f"Error fetching {ticker}: {e}. Retry {attempt+1}/5...")
                    time.sleep(1)
            print(f"Failed after retries: {ticker}")
            return None
        

        def fetch_stock_data3(ticker, start_date, end_date):
            for attempt in range(5):
                try:
                    value = yf.Ticker(ticker).history(start=start_date, end=end_date)
                    print(value)
                    return value
                except Exception as e:
                    print(f"Error: {e}, retry {attempt+1}/5")
                    time.sleep(1)
            return None
        

        def fetch_stock_data(ticker, start_date, end_date):
            for attempt in range(5):
                try:
                    data = yf.download(
                        ticker, 
                        start=start_date, 
                        end=end_date,
                        interval="1d",
                        progress=False
                    )
                    if not data.empty:
                        return data
                    else:
                        raise Exception("Empty response")
                except Exception as e:
                    print(f"Error: {e}. Retry {attempt+1}/5...")
                    time.sleep(3)

            print(f"Failed after retries: {ticker}")
            return None




        def fetch_alpha_vantage_range(symbol: str, start_date: str, end_date: str, api_key: str):
            """
            Fetch full Alpha Vantage daily data and filter by date range.
            start_date & end_date format: 'YYYY-MM-DD'
            """

            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",      # IMPORTANT: full gives all 20+ years
                "apikey": api_key
            }

            for attempt in range(5):  # retry loop
                try:
                    response = requests.get(url, params=params, timeout=10)
                    data = response.json()

                    # Rate limit
                    if "Note" in data:
                        print("Rate limited by Alpha Vantage. Waiting 60 seconds...")
                        time.sleep(60)
                        continue

                    # API errors
                    if "Error Message" in data:
                        print("API Error:", data["Error Message"])
                        return None

                    ts = data.get("Time Series (Daily)")
                    print(data)
                    if ts is None:
                        print("No time series data available.")
                        return None

                    # Convert JSON to DataFrame
                    df = pd.DataFrame(ts).T
                    df.index = pd.to_datetime(df.index)

                    df = df.rename(columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. volume": "Volume"
                    })

                    # Convert types
                    df = df.astype({
                        "Open": float,
                        "High": float,
                        "Low": float,
                        "Close": float,
                        "Volume": float
                    })

                    # Sort oldest → newest
                    df = df.sort_index()

                    # 🔥 Filter date range
                    df = df.loc[start_date:end_date]

                    return df

                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {e}")
                    time.sleep(2)

            print("Failed after max retries.")
            return None


        def fetch_vix_data(start_date: str, end_date: str):

            vix_data = yf.download('^VIX', start=start_date, end=end_date)
            return vix_data
        # df = fetch_stock_data(ticker, str(start_date), str(end_date))

        Api_Key_Alpha_Vintage = "IGLXO3R0C3I1BI5F"
        # Api_Key_Alpha_Vintage = "52F13IH6YQDWDJPE"
        
        # data = fetch_stock_data(ticker, start_date, end_date)
        df = fetch_alpha_vantage_range(ticker,start_date,end_date,Api_Key_Alpha_Vintage)
        # data2 = fetch_vix_data(start_date, end_date)

        # df = pd.DataFrame(data)
        # df2 = pd.DataFrame(data2)
        # df["Vix_Close"]=df2["Close"]
        

        # === PLACEHOLDER: Add technical indicator calculations ===
        def calculate_rsi(data, period=14):

            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        

        def calculate_macd(data, short_window=12, long_window=26, signal_window=9):

            short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
            long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=signal_window, adjust=False).mean()
            return macd, signal
        

        def calculate_ema(data, period=20):

            ema = data['Close'].ewm(span=period, adjust=False).mean()
            return ema
        

        def calculate_support_resistance(data, window=20):

            data['Support'] = data['Low'].rolling(window=window).min()
            data['Resistance'] = data['High'].rolling(window=window).max()
            return data
        
        def calculate_profit_loss(data):

            data['P&L%'] = data['Close'].pct_change() * 100
            return data
        
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

        # df = calculate_indicators(df)
        df['RSI']=calculate_rsi(df,14)
        df['MACD'],df["Signal"]=calculate_macd(df)
        df['EMA_20']=calculate_ema(df,20)
        df["EMA_50"]=calculate_ema(df,50)
        df.fillna(0,inplace=True)
        df = calculate_support_resistance(df)
        df.fillna(0,inplace=True) 
        calculate_profit_loss(df)
        df.fillna(0,inplace=True)
        df['ATR']=calculate_atr(df)
        adx_values, plus_di_values, minus_di_values = calculate_adx(df)
        df['ADX'] = adx_values
        df['Plus_DI'] = plus_di_values
        df['Minus_DI'] = minus_di_values
        df.fillna(0, inplace=True)

        st.write(df)

        index_values = pd.DataFrame(df.index.tolist())            #In this dataframe index are of date so we are storing them for future use
        st.session_state.index_values = index_values
        # df.reset_index(drop=True, inplace=True)
        st.session_state.df = df

        
        
        

        def plot_indicators_streamlit(data, indicators, start_date=None, end_date=None):
                   
                    if start_date and end_date:
                        data = data.loc[start_date:end_date]

                    fig, ax = plt.subplots(figsize=(12, 6))

                    for indicator in indicators:
                        if indicator in data.columns:
                            ax.plot(data.index, data[indicator], label=indicator)

                    ax.set_xlabel("Date")
                    ax.set_ylabel("Value")
                    ax.set_title("Stock Indicators")
                    ax.legend()
                    ax.grid(True)

                    st.pyplot(fig)

        def plot_indicator_correlation_streamlit(data, main_indicator):
            # Drop multi-index if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Compute correlation
            correlation = data.corr()[main_indicator].drop(main_indicator)
            correlation = correlation.sort_values(ascending=False)

            # Plot using matplotlib and Streamlit
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(correlation.index, correlation.values, color='skyblue')
            ax.set_xlabel("Indicators")
            ax.set_ylabel("Correlation")
            ax.set_title(f"Correlation of {main_indicator} with Other Indicators")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

            # Optionally return for use elsewhere
            return correlation
        

        # plot_indicators = ['Close','EMA_20']
        ind_begin = "2022-01-01"
        ind_end="2024-01-01"            
        st.success("Indicator plotted successfully")
        plot_indicators_streamlit(df, plot_indicators, str(ind_begin), str(ind_end))
        correlation_df = plot_indicator_correlation_streamlit(df.copy(), main_indicator='Close')
        # st.dataframe(correlation_df)


        # === PLACEHOLDER: Add scaling logic, sequence creation, train-test split ===
        def scale_data(data):

            scaler = MinMaxScaler()
            scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
            return scaled_data,scaler
        
        
        def inverse_scale_data(scaled_data, scaler):
            inverse_data = pd.DataFrame(scaler.inverse_transform(scaled_data), columns=scaled_data.columns, index=scaled_data.index)
            return inverse_data
        
        scaled_data,scaler = scale_data(df)
        st.session_state.scaled_data=scaled_data
        scaled_2d = np.array(scaled_data)
        scaled_2d_label = np.array(scaled_data['Close'])
        y =df[['Close']]
        scaled_close,closed_scaler=scale_data(y)
        # scaled_data, scaler = scale_data(df)

        st.session_state.scaler = scaler

        corr_val = 'Close'
        # seq_len = 60
        Predictor = 'Close'
      

        def create_sequences(data,Predictor, sequence_length=60):

            sequences = []
            labels = []
            for i in range(len(data) - sequence_length):
                sequences.append(data.iloc[i:i+sequence_length].values)
                labels.append(data.iloc[i+sequence_length][Predictor])
            return np.array(sequences), np.array(labels)
        
        sequence,label = create_sequences(scaled_data,Predictor,seq_len)
    # sequence,label

        scaled_2d=scaled_2d[-(scaled_2d.shape[0]-seq_len):]
        scaled_2d_label=scaled_2d_label[-(scaled_2d.shape[0]):]
        # return pd.DataFrame(sequences[-1]),pd.DataFrame(labels[-1])
        # sequence, label = create_sequences(scaled_data, sequence_length)

        def train_test_val_split(sequences, labels, test_size=0.2, val_size=0.1):

            Xgb_train, Xgb_test, Ygb_train, Ygb_test = train_test_split(scaled_2d, scaled_2d_label, test_size=test_size, shuffle=False,random_state=42)
            Xgb_train, Xgb_val, Ygb_train, Ygb_val = train_test_split(Xgb_train, Ygb_train, test_size=val_size, shuffle=False,random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, shuffle=False,random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, shuffle=False,random_state=42)
            return X_train, X_test, X_val, y_train, y_test, y_val,Xgb_train, Xgb_test, Xgb_val, Ygb_train, Ygb_test, Ygb_val

        X_train, X_test, X_val, y_train, y_test, y_val,Xgb_train, Xgb_test, Xgb_val, Ygb_train, Ygb_test, Ygb_val= train_test_val_split(sequence, label,test_size=test_size)
        
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
                GRU(75, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                GRU(75, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            return model
        

        def build_xgboost_model():

            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            return model
        

        def build_bidirectional_lstm_model(input_shape):

            model = Sequential([
               
                Bidirectional(LSTM(50, return_sequences=True), input_shape=input_shape),
                Dropout(0.4),

                Bidirectional(LSTM(50, return_sequences=False)),
                Dropout(0.4),

                # Dense layers

                Dense(50, activation='relu'),
                Dense(1)  # Output layer
            ])

            
            optimizer = keras.optimizers.Adam(learning_rate=0.0001)

            model.compile(optimizer=optimizer, loss='mse')
            return model

        def build_bidirectional_gru_model(input_shape):

            model = Sequential([
                # First bidirectional GRU layer
                Bidirectional(GRU(75, return_sequences=True), input_shape=input_shape),
                Dropout(0.2),

                # Second bidirectional GRU layer
                Bidirectional(GRU(75, return_sequences=False)),
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
                X_train, y_train,validation_data=(X_val, y_val),epochs=100,batch_size=32,callbacks=[early_stopping, checkpoint, reduce_lr],verbose=1,shuffle=False
            )

            return history


        def train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

            return train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val, model_path)

        def train_lstm_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

            checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=70, batch_size=32, callbacks=[checkpoint], verbose=1)


        def train_gru_model(model, X_train, y_train, X_val, y_val, model_path=model_path):

            checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[checkpoint],verbose=1)


      
        st.session_state.Xgb_test1 = X_test.reshape((X_test.shape[0], -1))
        def train_xgboost_on_residuals(lstm_model, X_train, y_train, X_test, y_test, variant):
  
            base_preds_train = lstm_model.predict(X_train)
            base_preds_test = lstm_model.predict(X_test)

            if variant == 1:
                print("Using LSTM residuals...")
            elif variant == 2:
                print("Using GRU residuals...")

        
            residuals_train = y_train - base_preds_train
            residuals_test = y_test - base_preds_test

            print("Residuals shape:", residuals_train.shape)

        
            Xgb_train = X_train.reshape((X_train.shape[0], -1))
            Xgb_test = X_test.reshape((X_test.shape[0], -1))

        
            residuals_train = residuals_train.flatten()

            xgb_model = build_xgboost_model()
            xgb_model.fit(Xgb_train, residuals_train)

        
            joblib.dump(xgb_model, 'xgb_model.pkl')

            return xgb_model, residuals_test,Xgb_test
            # , Xgb_train, Xgb_test
        

        def load_best_model(model_path):

            if os.path.exists(model_path):
                return keras.models.load_model(model_path)
            else:
                print(f"Model file {model_path} not found.")
                return None

        if trainable:
            model = 0
            if model_controller == 1:
                model = build_lstm_model((seq_len, X_train.shape[2]))
                model.summary()
                train_lstm_model(model , X_train, y_train, X_val, y_val)

            elif model_controller == 2:
                model = build_gru_model((seq_len, X_train.shape[2]))
                model.summary()
                train_gru_model(model, X_train, y_train, X_val, y_val)
                if st.button("Model Complexity"):
                    st.write()

            elif model_controller == 3:
                model = build_lstm_model((seq_len, X_train.shape[2]))
                model.summary()
                train_lstm_model(model, X_train, y_train, X_val, y_val)
                xgb_model, residuals_test,st.session_state.Xgb_test1 = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 1)

            elif model_controller == 4:
                model = build_gru_model((seq_len, X_train.shape[2]))
                model.summary()
                train_gru_model(model, X_train, y_train, X_val, y_val)
                xgb_model, residuals_test,st.session_state.Xgb_test1 = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 2)

            elif model_controller == 5:
                model = build_bidirectional_lstm_model((seq_len, X_train.shape[2]))
                model.summary()
                train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val)

            elif model_controller == 6:
                model = build_bidirectional_gru_model((seq_len, X_train.shape[2]))
                model.summary()
                train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val)

            elif model_controller == 7:
                model = build_bidirectional_lstm_model((seq_len, X_train.shape[2]))
                model.summary()
                train_bidirectional_lstm_model(model, X_train, y_train, X_val, y_val)
                xgb_model, residuals_test,st.session_state.Xgb_test1 = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 5)

            elif model_controller == 8:
                model = build_bidirectional_gru_model((seq_len, X_train.shape[2]))
                model.summary()
                train_bidirectional_gru_model(model, X_train, y_train, X_val, y_val)
                xgb_model, residuals_test,st.session_state.Xgb_test1 = train_xgboost_on_residuals(model, X_train, y_train, X_test, y_test, 6)

        if model_controller == 1:
            st.session_state.model = load_best_model(model_path)
        elif model_controller == 2:
            st.session_state.model = load_best_model(model_path)
        
        elif model_controller == 3:
            st.session_state.model = load_best_model(model_path)
            st.session_state.xgb_model= joblib.load('xgb_model.pkl')
        elif model_controller == 4:
            st.session_state.model = load_best_model(model_path)
            st.session_state.xgb_model= joblib.load('xgb_model.pkl')
        elif model_controller == 5:
            st.session_state.model = load_best_model(model_path)
        elif model_controller == 6:
            st.session_state.model = load_best_model(model_path)
        elif model_controller == 7:
            st.session_state.model = load_best_model(model_path)
            st.session_state.xgb_model= joblib.load('xgb_model.pkl')
        elif model_controller == 8:
            st.session_state.model = load_best_model(model_path)
            st.session_state.xgb_model= joblib.load('xgb_model.pkl')

       
        def calculate_rmse(y_true, y_pred):

            return np.sqrt(mean_squared_error(y_true, y_pred))

        def calculate_mape(y_true, y_pred):

            y_true, y_pred = np.array(y_true), np.array(y_pred)
            # Avoid division by zero
            mask = y_true != 0
            return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
        
        import numpy as np

        def min_max_percentage_error(y_true, y_pred):

            y_true = np.array(y_true).flatten()
            y_pred = np.array(y_pred).flatten()

            non_zero_indices = y_true != 0
            percentage_errors = np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices]) * 100

            min_error = np.min(percentage_errors)
            max_error = np.max(percentage_errors)

            return min_error, max_error

       


        def evaluate_model(model, X_test, y_test, model_controller, xgb_model=None, Xgb_test=None):
  
            base_preds = model.predict(X_test).flatten()

        
            if model_controller in [3, 4, 7, 8] and xgb_model is not None and Xgb_test is not None:
                xgb_preds = xgb_model.predict(Xgb_test).flatten()
                print("Xgboost used")
                hybrid_preds = base_preds + xgb_preds
            else:
                hybrid_preds = base_preds
                print("Not used")

        
            predictions_inv = inverse_scale_data(pd.DataFrame(hybrid_preds), closed_scaler)
            y_test_inv = inverse_scale_data(pd.DataFrame(y_test), closed_scaler)

        
            predictions_inv = np.array(predictions_inv).flatten()
            y_test_inv = np.array(y_test_inv).flatten()

            
            mse = mean_squared_error(y_test_inv, predictions_inv)
            mae = mean_absolute_error(y_test_inv, predictions_inv)
            r2 = r2_score(y_test_inv, predictions_inv)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
            min_error,max_error = min_max_percentage_error(y_test_inv, predictions_inv)
            print(f"Minimum Percentage Error: {min_error:.2f}%")
            print(f"Maximum Percentage Error: {max_error:.2f}%")

            return mse, mae, r2, rmse, mape, hybrid_preds, min_error, max_error


       
        try:
            st.success("Model run complete. Check results below.") 
            if model_controller == 1:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error= evaluate_model(st.session_state.model, X_test, y_test,model_controller)
                st.success(f"LSTM Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2}, RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")
            elif model_controller==2:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error= evaluate_model(st.session_state.model, X_test, y_test,model_controller)
                st.success(f"GRU Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller==3:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error= evaluate_model(st.session_state.model, X_test, y_test,model_controller,st.session_state.xgb_model,st.session_state.Xgb_test1)
                st.success(f"LSTM + Xgboost Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller==4:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error= evaluate_model(st.session_state.model, X_test, y_test,model_controller,st.session_state.xgb_model,st.session_state.Xgb_test1)
                st.success(f"GRU + Xgboost Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller == 5:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error = evaluate_model(st.session_state.model, X_test, y_test, model_controller)
                st.success(f"Bidirectional LSTM Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller == 6:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error = evaluate_model(st.session_state.model, X_test, y_test, model_controller)
                st.success(f"Bidirectional GRU Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller == 7:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error = evaluate_model(st.session_state.model, X_test, y_test, model_controller,st.session_state.xgb_model,st.session_state.Xgb_test1)
                st.success(f"Bidirectional LSTM + XGBoost Model Performance({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")

            elif model_controller == 8:
                mse, mae, r2 ,rmse,mape,predictions, min_error, max_error = evaluate_model(st.session_state.model, X_test, y_test, model_controller,st.session_state.xgb_model,st.session_state.Xgb_test1)
                st.success(f"Bidirectional GRU + XGBoost Model Performance ({model_path}): MSE={mse}, MAE={mae}, R2 Score={r2},RMSE={rmse}, MAPE={mape}")
                st.success(f"Minimum Error: {min_error}%  Maximum Error: {max_error}% ")
            predictions = predictions.astype(np.float64)
            predictions = predictions.reshape(-1,1)
            st.session_state.predictions = predictions  
            
        except:
            st.error("Named Model does not exsist. Please recheck name or train a new model")
    

        
        

        
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.closed_scaler = closed_scaler

# === METRICS DISPLAY ===
if st.session_state.metrics:
    st.subheader("Evaluation Metrics")
    st.write(st.session_state.metrics)


with tab2:
# === PLOT RESULTS ===
    if st.button("Directional Accuracy"):
        if st.session_state.ckecker==0:
            st.error("Please, First run the model")
        else:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            closed_scaler = st.session_state.closed_scaler
            model = st.session_state.model
            def inverse_scale_data(scaled_data, scaler):
                inverse_data = pd.DataFrame(scaler.inverse_transform(scaled_data), columns=scaled_data.columns, index=scaled_data.index)
                return inverse_data
            yop = pd.DataFrame(model.predict(X_test))
            pre=inverse_scale_data(yop, closed_scaler)
            st.session_state.ckecker2 = 1
            k = pd.DataFrame(y_test)
            act=inverse_scale_data(k, closed_scaler)
            st.subheader("Test Data VS Model Prediction")
            fig, ax = plt.subplots()
            
            ax.plot(pre, label='Predicted')  # Add label
            ax.plot(act, label='Actual')     # Add label

            ax.set_xlabel('Time (Days)')      # X-axis label
            ax.set_ylabel('Price')            # Y-axis label
            ax.set_title('Predicted vs Actual Prices')  # Title
            ax.legend()                       # Show legend

            st.pyplot(fig)

            plt.plot(pre)
            plt.plot(act)

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

                st.write(f"Directional Accuracy: {accuracy:.2f}%")
               

                return accuracy, metrics
            scaled_data= st.session_state.scaled_data
            predictions = st.session_state.predictions
            true = np.array(scaled_data[["Close"]])[-int(test_size * (len(scaled_data)-seq_len))-1:]
            st.session_state.true = true


# Basic usage
            st.subheader("Directional Accuracy For Next Day")
            accuracy, metrics = directional_accuracy(true, predictions,horizon=1,threshold=0.01)
            st.subheader("Directional Accuracy For Next 20 Day")
            accuracy, metrics = directional_accuracy(true, predictions, horizon=20, threshold=0.01)
     

with tab3:
    col1,col2 = st.columns(2)
    with col1:
        holding_period = st.number_input("Holding Period",value=80)
        position_size = (st.slider("Position Size %",10,100,step=10,value=25))/100
        take_profit = (st.slider("Take Profit %",5,90,step=5,value=15))/100
        verbose = st.checkbox("Verbose",value=False)
    with col2:
        initial_capital = st.number_input("Your initial Capital",value=100000)
        prediction_threshold = (st.slider("Prediction Threshold %",0.1,20.0,step=0.5,value=0.5))/1000
        stoploss = (st.slider("Stoploss %",0.1,40.0,step=1.0,value=10.0))/100
        
   
    if st.button("Simulate Trading"):
        if st.session_state.ckecker==0 or st.session_state.ckecker2==0:
            st.error("First, Please run Model and Directional Accuracy")
   
        else:
            st.info("Trading simulation results plotted.")
            def simulate_trading_returns_for_daily_predictions_fixed(y_true, y_pred=0, prices=0,
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
                        st.write(message)

                log(f"Generating trading signals based on predictions...")


                for i in range(len(y_pred) - 2):

                    if y_pred[i] > y_pred[i-1] + prediction_threshold:
                        # st.write(y_pred[i-1] + prediction_threshold)
                        signals[i] = 1
                    elif y_pred[i] < y_pred[i-1] - prediction_threshold:
                        # st.write(y_pred[i-1] - prediction_threshold)
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


                if True:
                    with st.expander("Detailed Decription"):
                        st.write(f"\nTrading Simulation Results:")
                        st.write(f"Initial Capital: ₹{initial_capital:,.2f}")
                        st.write(f"Final Capital: ₹{final_capital:,.2f}")
                        st.write(f"Total Return: {total_return:.2f}%")
                        st.write(f"Number of Trades: {len(trades)}")
                        # st.write(f"Completed Trades: {len(completed_trades)}")
                        st.write(f"Win Rate: {win_rate:.2f}%")
                        st.write(f"Average Profit per Trade: ₹{avg_profit_per_trade:.2f}")
                        # st.write(f"Profit Factor: {profit_factor:.2f}")
                        st.write(f"Average Holding Period: {avg_holding_period:.1f} days")
                        # st.write(f"Sharpe Ratio: {sharpe:.2f}")
                        # st.write(f"Sortino Ratio: {sortino:.2f}")
                        st.write(f"Max Drawdown: {max_drawdown:.2f}%")
                        st.write(f"Total Trading Days: {len(prices)}")

                return metrics


           
        

            def split_time_series_array(series, ratio=1-test_size):


                if isinstance(series, pd.DataFrame) and series.shape[1] == 1:
                    series = series.squeeze()

                split_index = int(len(series) * ratio)
                arr1 = series.iloc[:split_index].to_numpy()
                arr2 = series.iloc[split_index:].to_numpy()
                return arr1, arr2

            index_values = st.session_state.index_values
            train_Date,test_Date=split_time_series_array(index_values)
            st.session_state.test_Date = test_Date



            import streamlit as st
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            
            def plot_trading_results_improved(metrics,test_Date, title="Trading Simulation Results", figsize=(14, 12)):


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
                                    # f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}  |  "
                                    f"Max Drawdown: {max_dd:.2f}%")
                    fig.text(0.5, 0.97, performance_text, horizontalalignment='center',
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.subplots_adjust(hspace=0.3)

                if 'trades' in metrics and len(metrics['trades']) > 0:
                    profit_trades = [t for t in metrics['trades'] if 'profit_pct' in t and 'days_held' in t]

                    if len(profit_trades) > 0:
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

                st.pyplot(fig)
                return fig2

        # Optional: Add second figure for holding period vs profit and monthly heatmap (similar logic can be adapted)

            def get_original_prices(scaled_values, scaler, column_idx=0):

                scaled_values = scaled_values.flatten()

                dummy = np.zeros((len(scaled_values), scaler.scale_.shape[0]))


                dummy[:, column_idx] = scaled_values

                original_data = scaler.inverse_transform(dummy)

                return original_data[:, column_idx]

            scaler = st.session_state.scaler
            df = st.session_state.df
            test_Date = st.session_state.test_Date
            true = st.session_state.true
            predictions = st.session_state.predictions

            
            y_test = st.session_state.y_test
            original_y_true = get_original_prices(true, scaler, df.columns.get_loc('Close'))
            original_y_pred = get_original_prices(predictions, scaler, df.columns.get_loc('Close'))
            original_prices = original_y_true

        
            
            results = simulate_trading_returns_for_daily_predictions_fixed(
                  
                holding_period=holding_period,        # Shorter holding period
                prediction_threshold=prediction_threshold, # 0.5% threshold
                initial_capital=initial_capital,   # Starting capital
                position_size=position_size,       # Smaller position size (5% of capital)
                stop_loss=stoploss,           # 5% stop loss
                take_profit=take_profit,          # 10% take profit
                verbose=verbose,
                y_true=true,              # Scaled actual values
                y_pred=predictions,       # Scaled predictions
                prices=original_y_true                           # Print detailed logs
            )
            # Access the results
            print("Total no of Trading days",len(y_test))
            print(f"Total return: {results['total_return_pct']:.2f}%")
            print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")


            print(test_Date)


            st.pyplot( plot_trading_results_improved(results,test_Date=test_Date))

import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

with tab4:
    import requests
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from collections import Counter
    from datetime import date
    import plotly.graph_objects as go

    st.subheader("📰 Stock News Sentiment Dashboard")

    if st.button("Get Sentiments"):
        # Sidebar / user input
        # stock = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, ITC):", "AAPL").upper()
        stock = ticker
        # current_date = date.today().strftime("%Y-%m-%d")
        current_date = end_date

        # Load FinBERT model
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained("./finbert_local")
            model = AutoModelForSequenceClassification.from_pretrained("./finbert_local")
            labels = ["positive", "negative", "neutral"]  # ensure correct label order
            return tokenizer, model, labels

        tokenizer, model, labels = load_model()

        # Fetch market news
        api_token = "1wdBM7gpzQ2zUGPKzbGxIzDiSDo8ZOYHxLlFKySd"
        url = (
            f"https://api.marketaux.com/v1/news/all?"
            f"symbols={stock}&filter_entities=true&language=en&published_on={current_date}&api_token={api_token}"
        )

        st.write(f"📅 Showing news for **{stock}** on {current_date}")
        response = requests.get(url)
        data = response.json()

        if "data" not in data or len(data["data"]) == 0:
            st.warning("No news found for this stock today.")
        else:
            results = []
            for article in data["data"]:
                text = article["title"] + ". " + article.get("description", "")
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs).item()
                sentiment_label = labels[pred]
                sentiment_score = probs[0][pred].item()

                results.append({
                    "title": article["title"],
                    "description": article["description"],
                    "sentiment": sentiment_label,
                    "confidence": round(sentiment_score, 3),
                    "source": article["source"],
                    "url": article["url"]
                })

            # Newlly added code
            sentiment_weight = {
                "positive": 1,
                "negative": -1,
                "neutral": 0.15
            }


            weighted_sum = 0
            total_confidence = 0

            for r in results:
                label = r["sentiment"].lower()
                conf = r["confidence"]
                weight = sentiment_weight[label]

                weighted_sum += weight * conf
                total_confidence += conf

            # Avoid division by zero
            overall_sentiment_score = weighted_sum / total_confidence if total_confidence > 0 else 0

            # Round for neatness
            overall_sentiment_score = round(overall_sentiment_score, 3)

            if overall_sentiment_score not in st.session_state:
                st.session_state.overall_sentiment_score = overall_sentiment_score


            if overall_sentiment_score > 0.2:
                final_label = "Bullish"
            elif overall_sentiment_score < -0.2:
                final_label = "Bearish"
            else:
                final_label = "Neutral"

            # Sentiment summary
            count = Counter([r["sentiment"] for r in results])
            total = sum(count.values())
            sentiment_summary = {k: round(v / total, 2) for k, v in count.items()}

           

            #Newlly added code
            # Display weighted sentiment score
            st.markdown("### 📊 Overall Sentiment Meter (Weighted by Confidence)")

            gauge_value = overall_sentiment_score   # -1 to +1

            # Convert -1..+1 → 0..100 for the gauge
            meter_percentage = (gauge_value + 1) * 50


            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=meter_percentage,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "red"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "green"},
                    ],
                },
                title={'text': f"Sentiment: {final_label}"}
            ))

            st.plotly_chart(fig)


            
            st.markdown("### 🗞️ Recent News Articles")
            for r in results:
                color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}[r["sentiment"]]
                st.markdown(
                    f"**{color} [{r['title']}]({r['url']})**  \n"
                    f"{r['description']}  \n"
                    f"**Sentiment:** {r['sentiment'].capitalize()} ({r['confidence']*100:.1f}% confidence)  \n"
                    f"*Source:* {r['source']}"
                )
                st.divider()



with tab5:
    n_days = st.number_input("No of days you want to see direction",max_value=30,value=1)
     
    st.write("### Sentiment Components")
    use_technical = st.toggle("Include Technical Sentiment", value=True)
    use_news = st.toggle("Include News Sentiment", value=True)

    if st.button("Forecast"):
        if st.session_state.ckecker==0:
            st.error("Please first run The model")
        else:
            from datetime import timedelta
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

                # st.markdown("Hello")
                st.markdown(result_df['Scaled_Prediction'])


                return result_df
            

            def predict_next_day_sentiment_old(model, scaled_data, index_values, seq_len=60, neutral_threshold=0.002):
                """
                Predicts the next day's price using the last `seq_len` data points
                and compares it with today's price to infer sentiment:
                    - Positive if next day price > today price
                    - Negative if next day price < today price
                    - Neutral if change within ±neutral_threshold (default 0.2%)
                """

                # --- Prepare last sequence ---
                sequence = scaled_data.iloc[-seq_len:].values
                close_idx = scaled_data.columns.get_loc('Close')

                # --- Predict next day's price ---
                x_input = sequence.reshape(1, seq_len, sequence.shape[1])
                scaled_pred = float(model.predict(x_input, verbose=0)[0][0])

                # --- Extract last (today's) scaled close price ---
                scaled_today = float(scaled_data.iloc[-1, close_idx])

                # --- Compute percentage change ---
                pct_change = ((scaled_pred - scaled_today) / scaled_today) * 100

                # --- Determine sentiment ---
                if abs(pct_change) <= neutral_threshold * 100:
                    sentiment = "Neutral"
                elif scaled_pred > scaled_today:
                    sentiment = "Positive"
                else:
                    sentiment = "Negative"

                # --- Prepare result dataframe ---
                next_date = index_values.iloc[-1, 0] + pd.Timedelta(days=1)
                while next_date.weekday() >= 5:  # skip weekends
                    next_date += pd.Timedelta(days=1)

                result_df = pd.DataFrame({
                    "Date": [next_date],
                    "Scaled_Today": [scaled_today],
                    "Scaled_Prediction": [scaled_pred],
                    "Pct_Change": [pct_change],
                    "Sentiment": [sentiment]
                })

                # --- Optional: display inside Streamlit ---
                st.write("### Next Day Prediction Summary")
                st.dataframe(result_df)

                st.write(st.session_state.pos) #st.session_state.pos give value of positive sentiment from finbert that is feed with live news between 0 and 1 for negative and neutral use .neg and .neu

                return result_df



            #New function for total sentiments
            def predict_next_day_sentiment(model, scaled_data, index_values, seq_len=60, neutral_threshold=0.002):
               


                close_idx = scaled_data.columns.get_loc('Close')

                
                sequence_today = scaled_data.iloc[-(seq_len+1):-1].values   # actual window until yesterday
                x_today = sequence_today.reshape(1, seq_len, sequence_today.shape[1])

                scaled_pred_today = float(model.predict(x_today, verbose=0)[0][0])

             
                sequence_next = scaled_data.iloc[-seq_len:].values          # actual window until today
                x_next = sequence_next.reshape(1, seq_len, sequence_next.shape[1])

                scaled_pred_next = float(model.predict(x_next, verbose=0)[0][0])

                
                diff = scaled_pred_next - scaled_pred_today
                pct_change = (diff / abs(scaled_pred_today)) * 100  # relative % difference

              
                if abs(pct_change) <= neutral_threshold * 100:
                    technical_sentiment_score = 0.04
                    technical_sentiment = "Neutral"
                elif diff > 0:
                    technical_sentiment_score = float(np.tanh(pct_change *2))
                    technical_sentiment = "Positive"
                else:
                    technical_sentiment_score = float(np.tanh((pct_change* 2)))
                    technical_sentiment = "Negative"


               
                # Ensure session state values exist
                pos = st.session_state.get("pos", 0)
                neg = st.session_state.get("neg", 0)
                neu = st.session_state.get("neu", 0)
                news_sentiment_score = st.session_state.get("overall_sentiment_score")  # simple difference
                if news_sentiment_score > 0.1:
                    news_sentiment = "Positive"
                elif news_sentiment_score < -0.1:
                    news_sentiment = "Negative"
                else:
                    news_sentiment = "Neutral"

                

               

                # --- Weighted combination logic ---
                weights = {"technical": 0.6, "news": 0.4}

                total_score = 0
                total_weight = 0

                if use_technical:
                    total_score += weights["technical"] * technical_sentiment_score
                    total_weight += weights["technical"]
                if use_news:
                    total_score += weights["news"] * news_sentiment_score
                    total_weight += weights["news"]
                

                weighted_sentiment_score = total_score / total_weight if total_weight > 0 else 0

                if weighted_sentiment_score > 0.05:
                    total_sentiment = "Positive"
                    recommendation = "Buy"
                elif weighted_sentiment_score < -0.05:
                    total_sentiment = "Negative"
                    recommendation = "Sell"
                else:
                    total_sentiment = "Neutral"
                    recommendation = "Hold"

                # --- Prepare result dataframe ---
                next_date = index_values.iloc[-1, 0] + pd.Timedelta(days=1)
                while next_date.weekday() >= 5:  # skip weekends
                    next_date += pd.Timedelta(days=1)

                result_df = pd.DataFrame({
                    "Date": [next_date],
                    "Predicted Change (%)": [pct_change],
                    "Technical Sentiment": [technical_sentiment],
                    "Technical Score": [technical_sentiment_score],
                    "News Sentiment": [news_sentiment],
                    "News Score": [news_sentiment_score],
                    "Total Sentiment": [total_sentiment],
                    "Recommendation": [recommendation]
                })

                # --- Display results in Streamlit ---
                st.write("### Sentiment Summary")
                st.dataframe(result_df)

                st.metric(label="Final Weighted Sentiment Score", value=f"{weighted_sentiment_score:.3f}")
                st.success(f"**Recommendation:** {recommendation}")

                return result_df

            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import streamlit as st

            def plot_scaled_forecast(df, scaled_data, index_values, forecast_df, days_to_show=60):
                # Extract historical and forecast data
                hist_dates = index_values.iloc[-days_to_show:, 0].values
                hist_scaled_close = scaled_data['Close'][-days_to_show:].values

                forecast_dates = forecast_df['Date']
                forecast_values = forecast_df['Scaled_Prediction']

                # Create a figure object for Streamlit
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot historical data
                ax.plot(hist_dates, hist_scaled_close, color='blue', label='Historical (Scaled)')

                # Plot forecast data
                ax.plot(forecast_dates, forecast_values, color='red', linestyle='--', label='Forecast (Scaled)')

                # Vertical line for forecast start
                ax.axvline(x=hist_dates[-1], color='green', linestyle='-', alpha=0.5, label='Forecast Start')

                # Formatting
                ax.set_title('Stock Price Forecast (MinMax Scaled)')
                ax.set_xlabel('Date')
                ax.set_ylabel('Scaled Price')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Date formatting
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()

                # Show in Streamlit
                st.pyplot(fig)

            model = st.session_state.model
            scaled_data = st.session_state.scaled_data
            index_values = st.session_state.index_values
            df = st.session_state.df


            scaled_forecast = predict_next_day_sentiment(model=model, scaled_data = scaled_data, index_values = index_values, seq_len=60, neutral_threshold=0.002)
           
# === NOTES ===
st.markdown("#### 📌Developer's Note")
st.code("""
Project Title: Stock Market Forecasting and Strategy Simulation Using Deep Learning

This project has been developed with the intent of applying deep learning techniques to forecast stock market prices and simulate trading strategies based on model predictions. 
The core objective is to bridge financial domain knowledge with modern AI methodologies to assess the practical effectiveness of predictive models in real-time trading environments.

✅ Key Features:
Data Preprocessing: Scaled and structured time-series data using technical indicators and historical price movements.
Model Architecture: Implementation of LSTM/GRU-based models for sequential learning and price prediction.
Forecast Visualization: Interactive and intuitive forecast graphs using Streamlit and Matplotlib.
Trading Strategy Simulation: Backtested performance based on configurable strategy parameters such as position size, stop-loss, and take-profit.
Performance Metrics: Portfolio equity curve, drawdown, daily returns, and trade profitability are computed and visualized.

🛠 Technologies Used:
Python, TensorFlow/Keras, XGBoost
Pandas, NumPy, Scikit-learn
Matplotlib, Streamlit
Financial APIs and historical datasets

⚠️ Disclaimer:
This project is developed strictly for educational and research purposes. 
The forecasts and simulated trading strategies are not financial advice and should not be relied upon for real-world trading decisions without further validation and risk assessment. 
Real-world trading involves risk, and past performance is not indicative of future results.
""")
