import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import Tuple, List

class DataProcessor:
    """
    Advanced data processing pipeline for high-dimensional market data.
    Handles outlier detection, feature engineering, and normalization.
    """
    def __init__(self, config_path: str):
        self.scaler = RobustScaler() # Better for handling outliers in price data
        self.category_embeddings = {}
        
    def process_batch(self, raw_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ingests raw listing data and returns processed tensors for training.
        """
        # 1. Temporal Feature Engineering
        raw_data['listing_age'] = (pd.Timestamp.now() - pd.to_datetime(raw_data['created_at'])).dt.total_seconds()
        raw_data['is_weekend'] = pd.to_datetime(raw_data['created_at']).dt.dayofweek >= 5
        
        # 2. Outlier Detection using IQR
        Q1 = raw_data['price'].quantile(0.25)
        Q3 = raw_data['price'].quantile(0.75)
        IQR = Q3 - Q1
        raw_data = raw_data[~((raw_data['price'] < (Q1 - 1.5 * IQR)) | (raw_data['price'] > (Q3 + 1.5 * IQR)))]
        
        # 3. NLP Feature Extraction (Mock)
        # In production, this would use BERT embeddings
        text_features = self._extract_text_features(raw_data['description'])
        
        # 4. Normalization
        numerical_features = raw_data[['price', 'listing_age', 'seller_rating']].values
        normalized_features = self.scaler.fit_transform(numerical_features)
        
        return np.hstack([normalized_features, text_features])

    def _extract_text_features(self, texts: pd.Series) -> np.ndarray:
        """
        Simulates extraction of semantic density and sentiment scores.
        """
        # Mocking 768-dim BERT embeddings
        return np.random.normal(0, 1, (len(texts), 768))

    def augment_data(self, features: np.ndarray) -> np.ndarray:
        """
        Applies mixup and noise injection for robust training.
        """
        noise = np.random.normal(0, 0.01, features.shape)
        return features + noise
