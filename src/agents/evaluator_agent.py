import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EvaluatorAgent:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.scaler = StandardScaler()

        self.features = [
            'spend', 'impressions', 'clicks',
            'creative_type_code', 'platform_code', 'audience_code'
        ]

    def build_and_train_model(self):
        X = self.df[self.features].values
        y = self.df['calculated_roas'].values

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

    def detect_anomalies(self):
        X = self.df[self.features].values
        X_scaled = self.scaler.transform(X)

        preds = self.model.predict(X_scaled, verbose=0).flatten()

        df = self.df.copy()
        df['predicted_roas'] = preds
        df['roas_deviation'] = df['calculated_roas'] - preds

        return df[df['roas_deviation'] < -1.5].sort_values('roas_deviation')
