import os
#import io
import json
import requests
import numpy as np
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings
#import firebase_admin
#from firebase_admin import credentials, firestore

#Environment Configurations
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
app = FastAPI(title="Fake Store AI Platform Wrapper")

#if not firebase_admin._apps:
    #firebase_admin.initialize_app()
#db = firestore.client()

STORE_URL = "https://escuelajs.co"
shopping_cart = []

#Standard Core Pydantic Models
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

# New Pydantic Models for AI Capabilities
class AutoCategoryInput(BaseModel):
    title: str
    description: str

class RewriteDescriptionInput(BaseModel):
    description: str
    target_language: str = "Hindi"
    tone: str = "Semi-Professional, friendly, and concise"

class ReviewsInput(BaseModel):
    reviews: list[str]

# Structural Schemas for AI Responses
class CategorizationSchema(BaseModel):
    suggested_category_name: str
    confidence_score: float
    reasoning: str

class ReviewSummarySchema(BaseModel):
    overall_sentiment: str  # e.g., Positive, Mixed, Negative
    pros: list[str]
    cons: list[str]
    verdict: str

#Helper Functions
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
    #Fetches the external inventory list directly with failure handling
    response = requests.get(STORE_URL)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch store catalog.")
    return response.json()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Store AI API Wrapper! Go to /docs to test endpoints."}

@app.get("/hello")
def say_hello():
    return {"message": "Hello World"}

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

# UPDATED FEATURE: AI Recommendations Endpoint
@app.get("/cart/ai-recommendation")
def get_cart_recommendations():
    """
    Reads items currently inside the shopping cart, looks at available inventory, 
    and asks Gemini to write tailored recommendations.
    """
    if not shopping_cart:
        return {"recommendation": "Your cart is empty! Add products first to get AI recommendations."}

    client = get_gemini_client()
    all_products = fetch_catalog()
    # Isolate product IDs in cart
    cart_ids = [item["product_id"] for item in shopping_cart]
    # Map details cleanly to keep prompts light and save token overhead
    current_cart_details = [
        {"title": p["title"], "category": p["category"]["name"], "price": p["price"]}
        for p in all_products if p["id"] in cart_ids
    ]
    
    available_catalog = [
        {"id": p["id"], "title": p["title"], "category": p["category"]["name"], "price": p["price"]}
        for p in all_products if p["id"] not in cart_ids
    ][:15] 

    prompt = f"""
    You are an e-commerce assistant. Review the customer's current shopping cart:
    {current_cart_details}

    Select the 2 best complementary items from this store catalog snippet:
    {available_catalog}

    Provide a short, welcoming, 2-sentence recommendation to the user explaining why they should add those items.
    """

    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return {"recommendation": completion.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini processing error: {str(e)}")


# FEATURE 1: Smart Search using Vector Embeddings (Concept Match)
@app.get("/products/search/smart")
def smart_search(query: str):
    """
    Finds items conceptually matching a user query, even if keywords mismatch.
    """
    client = get_gemini_client()
    catalog = fetch_catalog()
    
    # 1. Generate text embedding for user's query
    query_embed_resp = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )
    query_vector = np.array(query_embed_resp.embeddings.values)
    
    # 2. Build product context strings
    product_texts = [f"{p['title']}: {p['description']}" for p in catalog]
    
    # 3. Batch generate text embeddings for entire store catalog
    catalog_embed_resp = client.models.embed_content(
        model="text-embedding-004",
        contents=product_texts
    )
    
    # 4. Rank results via Cosine Similarity score
    scored_products = []
    for idx, emb in enumerate(catalog_embed_resp.embeddings):
        prod_vector = np.array(emb.values)
        similarity = np.dot(query_vector, prod_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(prod_vector))
        scored_products.append((similarity, catalog[idx]))
        
    # 5. Sort by highest score first and return top 5
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return {"results": [item[1] for item in scored_products[:5]]}


# FEATURE 2: Automated Product Categorization (Structured Outputs)
@app.post("/products/ai-classify")
def classify_product(item: AutoCategoryInput):
    """
    Analyzes item titles and descriptions to correctly place them into segments.
    """
    client = get_gemini_client()
    
    prompt = f"""
    Analyze this item submission for an e-commerce marketplace:
    Title: {item.title}
    Description: {item.description}
    
    Determine the single most accurate category name (e.g., Electronics, Clothes, Shoes, Furniture, Toys).
    """
    
    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CategorizationSchema
            )
        )
        return json.loads(completion.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FEATURE 3: Multimodal Visual Shopping Search (Image Upload)
@app.post("/products/search/visual")
async def visual_search(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns matching variants from our catalog list.
    """
    client = get_gemini_client()
    catalog = fetch_catalog()
    
    # Prune catalog data size down to stay safely inside token boundaries
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
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type),
                prompt
            ]
        )
        
        # Parse the raw comma list text out to recover full item matches
        id_strings = completion.text.strip().split(",")
        matched_ids = [int(i.strip()) for i in id_strings if i.strip().isdigit()]
        
        results = [p for p in catalog if p["id"] in matched_ids]
        return {"matched_products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# FEATURE 4: Localization & Persona Rewrites
@app.post("/products/ai-translate")
def translate_description(data: RewriteDescriptionInput):
    """
    Translates or re-tones descriptions for target regional audiences.
    """
    client = get_gemini_client()
    prompt = f"""
    Rewrite this product description.
    Target Language: {data.target_language}
    Tone Preference: {data.tone}
    Original Description: {data.description}
    """
    try:
        completion = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return {"rewritten_text": completion.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FEATURE 5: AI Review Summary & Sentiment Analytics
@app.post("/products/reviews/analyze")
def analyze_reviews(data: ReviewsInput):
    """
    Compiles long list blocks of community feedback text into dynamic summaries.
    """
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
