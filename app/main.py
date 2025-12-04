from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.core.analysis import analyze_product_listing

app = FastAPI(
    title="Switch Protocol AI Engine",
    description="AI-powered analysis service for the Switch Social Market",
    version="1.0.0"
)

class ProductRequest(BaseModel):
    id: str
    title: str
    price: float
    condition: str
    description: str
    category: str
    seller_id: str

class AnalysisResponse(BaseModel):
    product_id: str
    summary: str
    price_analysis: dict
    seller_trust: dict
    checklist: List[str]

@app.get("/")
async def root():
    return {"message": "Switch AI Engine is running", "status": "active"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_product(product: ProductRequest):
    """
    Analyze a product listing to provide AI insights, price valuation, and safety checks.
    """
    try:
        result = analyze_product_listing(product)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
