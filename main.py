import os
import json
import requests
import numpy as np
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings

# Environment Configurations
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
app = FastAPI(title="Fake Store AI Platform Wrapper")

STORE_URL = "https://escuelajs.co"
shopping_cart = []

# Standard Core Pydantic Models
class Category(BaseModel):
    id: int
    name: str
    image: HttpUrl
    slug: str

class Product(BaseModel):
    id: int
    title: str
    price: float
    description: str
    category: Category
    images: list[HttpUrl]

class CartInput(BaseModel):
    product_id: int
    quantity: int = 1

# Updated/New Pydantic Models for AI Capabilities
class DynamicTranslateInput(BaseModel):
    target_language: str = "Hindi"
    tone: str = "Semi-Professional, friendly, and concise"

class ReviewsInput(BaseModel):
    reviews: list[str]

# Structural Schemas for AI Responses
class CartAnalysisSchema(BaseModel):
    detected_core_category: str
    reasoning: str
    recommended_product_ids: list[int]
    pitch_to_user: str

class ReviewSummarySchema(BaseModel):
    overall_sentiment: str  
    pros: list[str]
    cons: list[str]
    verdict: str

# Represents recommendations for a single specific item in the cart
class ItemRecommendation(BaseModel):
    cart_product_id: int
    cart_product_title: str
    recommended_product_ids: list[int]
    pitch_to_user: str

# The final structure the AI pipeline will return
class MultiCartAnalysisSchema(BaseModel):
    recommendations_by_item: list[ItemRecommendation]
    global_reasoning: str

# Helper Functions
def get_gemini_client():
    """Initializes the official GenAI Client with built-in automatic retry handling."""
    apikey = os.environ.get("GEMINI_API_KEY").strip()
    if not apikey:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is missing."
        )
    return genai.Client(api_key=apikey, vertexai=False)

def fetch_catalog():
    """Fetches the external inventory list directly with failure handling."""
    response = requests.get("https://api.escuelajs.co/api/v1/products")
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch store catalog.")
    return response.json()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Store AI API Wrapper! Go to /docs to test endpoints."}

@app.get("/products")
def list_products():
    return fetch_catalog()

@app.post("/cart/add")
def add_to_cart(item: CartInput):
    all_products = fetch_catalog()
    valid_ids = [p["id"] for p in all_products]
    
    if item.product_id not in valid_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Product with ID {item.product_id} does not exist."
        )
    
    cart_item = {"product_id": item.product_id, "quantity": item.quantity}
    shopping_cart.append(cart_item)
    return {"message": "Successfully added item to cart", "added_item": cart_item}

@app.get("/cart")
def view_cart():
    return {
        "total_items_in_cart": len(shopping_cart), 
        "cart_items": shopping_cart
    }


# FEATURE 1: Combined Cart Analysis, Categorization & Inventory Matching
@app.get("/cart/ai-analyze-and-match")
def analyze_cart_and_match_inventory():
    if not shopping_cart:
        raise HTTPException(
            status_code=400, 
            detail="Your cart is empty! Add products first to run classification and matching."
        )
        
    client = get_gemini_client()
    all_products = fetch_catalog()
    
    cart_ids = [item["product_id"] for item in shopping_cart]
    items_in_cart = [p for p in all_products if p["id"] in cart_ids]
    
    available_catalog = [
        {"id": p["id"], "title": p["title"], "category": p["category"]["name"], "description": p["description"]}
        for p in all_products if p["id"] not in cart_ids
    ][:30]

    #Forces item-by-item analysis
    prompt = f"""
    You are an advanced e-commerce cross-selling and recommendation system.
    
    The user has multiple items in their shopping cart:
    {json.dumps(items_in_cart)}
    
    Look over our available store inventory:
    {json.dumps(available_catalog)}
    
    For EACH individual item currently in the user's cart, look through the available store inventory and select 1 to 2 relevant, complementary, or similar products. 
    Then, write a tailored 1-sentence pitch for that specific pairing.
    """

    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # Use our new itemized schema here
                response_schema=MultiCartAnalysisSchema 
            )
        )
        
        ai_response = json.loads(completion.text)
        item_recommendations = ai_response.get("recommendations_by_item", [])
        
        # Build a complete structured payload with full product objects from catalog
        final_output = []
        for rec in item_recommendations:
            rec_ids = rec.get("recommended_product_ids", [])
            matched_products = [p for p in all_products if p["id"] in rec_ids]
            
            final_output.append({
                "cart_product_id": rec.get("cart_product_id"),
                "cart_product_title": rec.get("cart_product_title"),
                "pitch_to_user": rec.get("pitch_to_user"),
                "similar_matched_products": matched_products
            })
            
        return {
            "global_reasoning": ai_response.get("global_reasoning"),
            "recommendations": final_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Matching pipeline failed: {str(e)}")
    
# FEATURE 2:Smart Search using Vector Embeddings
@app.get("/products/search/smart")
def smart_search(query: str):
    """Finds items conceptually matching a user query, even if keywords mismatch."""
    client = get_gemini_client()
    catalog = fetch_catalog()
    
    query_embed_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_vector = np.array(query_embed_resp.embeddings[0].values) 
    product_texts = [f"{p['title']}: {p['description']}" for p in catalog]
    
    catalog_embed_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=product_texts
    )
    
    scored_products = []
    for idx, emb in enumerate(catalog_embed_resp.embeddings):
        prod_vector = np.array(emb.values)
        similarity = np.dot(query_vector, prod_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(prod_vector))
        scored_products.append((similarity, catalog[idx]))
        
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return {"results": [item[1] for item in scored_products[:5]]}


# FEATURE 3: Cart Context Localization & Translation
@app.post("/cart/translate-descriptions")
def translate_cart_descriptions(data: DynamicTranslateInput):
    """
    Looks inside the active cart, extracts text configurations from items present, 
    and passes them to Gemini for localization without asking the user for text.
    """
    if not shopping_cart:
        raise HTTPException(
            status_code=400, 
            detail="Shopping cart is empty. Nothing to translate."
        )
        
    client = get_gemini_client()
    all_products = fetch_catalog()
    
    cart_ids = [item["product_id"] for item in shopping_cart]
    items_to_translate = [
        {"id": p["id"], "title": p["title"], "description": p["description"]}
        for p in all_products if p["id"] in cart_ids
    ]
    
    system_instruction = (
        "You are a strict e-commerce localization utility. Translate the provided list of items "
        "including their descriptions and titles into the targeted configuration. "
        "Return ONLY valid minified JSON that mirrors the input structure with rewritten texts. "
        "Do not wrap your output in markdown blocks or include pre/post conversational text strings."
    )
    
    prompt = f"""
    Target Language: {data.target_language}
    Tone Preference: {data.tone}
    
    Items Array to Transform:
    {json.dumps(items_to_translate)}
    """
    
    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        return {"translated_cart_items": json.loads(completion.text.strip())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Localization engine failure: {str(e)}")


# FEATURE 4: Multimodal Visual Shopping Search
@app.post("/products/search/visual")
async def visual_search(file: UploadFile = File(...)):
    """Accepts an uploaded image file and returns matching variants from our catalog list."""
    client = get_gemini_client()
    catalog = fetch_catalog()
    
    light_catalog = [{"id": p["id"], "title": p["title"], "category": p["category"]["name"]} for p in catalog][:30]
    image_bytes = await file.read()
    
    prompt = f"""
    Analyze the uploaded item in this visual photo. Look over our catalog data:
    {light_catalog}
    
    Find and return up to 3 Product IDs from the catalog list that look visually similar in color, shape, pattern, or utility.
    Return your answer strictly as a comma-separated list of IDs, nothing else. Example: 4, 12, 18
    """
    
    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash', # Updated to 2.5-flash as it is multimodal
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                prompt
            ]
        )
        
        id_strings = completion.text.strip().split(",")
        matched_ids = [int(i.strip()) for i in id_strings if i.strip().isdigit()]
        
        results = [p for p in catalog if p["id"] in matched_ids]
        return {"matched_products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FEATURE 5: AI Review Summary & Sentiment Analytics
@app.post("/products/reviews/analyze")
def analyze_reviews(data: ReviewsInput):
    """Compiles long list blocks of community feedback text into dynamic summaries."""
    client = get_gemini_client()
    prompt = f"""
    Read through these customer reviews left on our site product page:
    {data.reviews}
    Summarize the overall shared customer consensus. Extract core pros and cons.
    """ 
    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewSummarySchema
            )
        )
        return json.loads(completion.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))