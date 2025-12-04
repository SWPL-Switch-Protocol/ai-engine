import torch
import torch.nn as nn
import torch.nn.functional as F

class PricePredictionModel(nn.Module):
    """
    A neural network for predicting fair market value based on product features.
    Inputs: [category_embedding, condition_score, seller_reputation, seasonal_index]
    """
    def __init__(self, input_dim=128, hidden_dim=64):
        super(PricePredictionModel, self).__init__()
        
        # Feature processing layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Price estimation head
        self.price_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicted price
        )
        
        # Confidence interval head (predicts variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Variance must be positive
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        predicted_price = self.price_head(features)
        uncertainty = self.uncertainty_head(features)
        return predicted_price, uncertainty

class SemanticSearchEncoder(nn.Module):
    """
    Transformer-based encoder for semantic product search.
    Maps product descriptions to a high-dimensional vector space.
    """
    def __init__(self, vocab_size=10000, embedding_dim=256):
        super(SemanticSearchEncoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(embedding_dim, 128) # Project to 128-d latent space

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        emb = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        emb = emb.permute(1, 0, 2) # Transformer expects [seq_len, batch, dim]
        
        encoded = self.transformer_encoder(emb)
        encoded = encoded.permute(1, 0, 2) # Back to [batch, seq_len, dim]
        
        # Global Average Pooling
        pooled = encoded.mean(dim=1)
        
        # Project to latent space and normalize
        vector = self.projection(pooled)
        return F.normalize(vector, p=2, dim=1)
