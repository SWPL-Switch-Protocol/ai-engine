import torch
import numpy as np
from app.models.neural_net import PricePredictionModel, SemanticSearchEncoder

class AIInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Initializing AI Engine on {self.device}...")
        
        # Load Price Prediction Model
        self.price_model = PricePredictionModel(input_dim=10).to(self.device)
        self.price_model.eval()
        
        # Load Semantic Search Encoder
        self.search_encoder = SemanticSearchEncoder().to(self.device)
        self.search_encoder.eval()
        
        # Mock database of product embeddings (for similarity search)
        self.vector_db = torch.randn(1000, 128).to(self.device) # 1000 products
        self.vector_db = torch.nn.functional.normalize(self.vector_db, p=2, dim=1)

    def predict_price_range(self, features: list):
        """
        Predicts fair market price and confidence interval using the neural net.
        """
        with torch.no_grad():
            input_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
            price, uncertainty = self.price_model(input_tensor)
            
            predicted_price = price.item() * 1000 # Scale up
            variance = uncertainty.item() * 100
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence_interval": [
                    round(predicted_price - variance, 2),
                    round(predicted_price + variance, 2)
                ],
                "confidence_score": round(1.0 / (1.0 + variance), 4)
            }

    def find_similar_products(self, query_tokens: list, top_k=5):
        """
        Performs semantic vector search to find similar historical sales.
        """
        with torch.no_grad():
            # Simulate token IDs
            input_tensor = torch.randint(0, 10000, (1, len(query_tokens))).to(self.device)
            
            # Encode query
            query_vector = self.search_encoder(input_tensor)
            
            # Cosine similarity search
            similarities = torch.mm(query_vector, self.vector_db.t())
            scores, indices = torch.topk(similarities, k=top_k)
            
            return {
                "similar_indices": indices.cpu().numpy().tolist()[0],
                "similarity_scores": scores.cpu().numpy().tolist()[0]
            }

# Singleton instance
engine = AIInferenceEngine()
