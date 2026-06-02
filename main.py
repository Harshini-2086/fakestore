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
import firebase_admin
from firebase_admin import credentials, firestore

#Environment Configurations
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
app = FastAPI(title="Fake Store AI Platform Wrapper")

if not firebase_admin._apps:
    firebase_admin.initialize_app()
db = firestore.client()

STORE_URL = "https://escuelajs.co"

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
    apikey = os.environ.get("GEMINI_API_KEY")
    if not apikey:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is missing."
        )
        
    client = genai.Client(
        api_key=apikey,
        http_options={"max_retries": 3}
    )
    return client

def fetch_catalog():
    # Fetches the external inventory list directly with failure handling
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


# ==========================================
# NEW INFRASTRUCTURE: DATABASE CACHE SYNC
# ==========================================

@app.post("/products/sync")
def sync_products_to_cache():
    """
    OPTIMIZATION: Pulls the external catalog, generates text embeddings ONCE via Gemini,
    and saves everything to Firestore. Run this once at startup or via a cron job.
    """
    client = get_gemini_client()
    catalog = fetch_catalog()
    
    # Context strings to turn into embeddings
    product_texts = [f"{p['title']}: {p['description']}" for p in catalog]
    
    # Batch generate text embeddings for entire store catalog
    catalog_embed_resp = client.models.embed_content(
        model="text-embedding-004",
        contents=product_texts
    )
    
    # Save to Firestore using a high-performance batch write
    batch = db.batch()
    for idx, p in enumerate(catalog):
        prod_id = str(p["id"])
        doc_ref = db.collection("products").document(prod_id)
        
        # Grab vector array values cleanly
        embedding_values = catalog_embed_resp.embeddings[idx].values
        
        product_data = {
            "product_details": p,
            "embedding": embedding_values
        }
        batch.set(doc_ref, product_data)
        
    batch.commit()
    return {"message": f"Successfully cached {len(catalog)} products and vector embeddings in Firestore!"}


@app.get("/products")
def list_products():
    """Reads inventory directly out of our Firestore cache."""
    docs = db.collection("products").stream()
    products = [doc.to_dict()["product_details"] for doc in docs]
    
    # Fallback to direct fetch if the cache hasn't been seeded yet
    if not products:
        return fetch_catalog()
    return products


# ==========================================
# UPGRADED FEATURES: MULTI-USER STORAGE
# ==========================================

@app.post("/cart/{user_id}/add")
def add_to_cart(user_id: str, item: CartInput):
    """Adds or updates an item in a specific user's persistent database cart."""
    # Validate product exists in our system
    prod_ref = db.collection("products").document(str(item.product_id)).get()
    if not prod_ref.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Product with ID {item.product_id} does not exist in the database cache. Run /products/sync first."
        )
    
    cart_ref = db.collection("carts").document(user_id)
    cart_doc = cart_ref.get()
    
    if cart_doc.exists:
        items = cart_doc.to_dict().get("items", [])
        # Increment quantity if item is already in cart
        for existing_item in items:
            if existing_item["product_id"] == item.product_id:
                existing_item["quantity"] += item.quantity
                break
        else:
            items.append({"product_id": item.product_id, "quantity": item.quantity})
    else:
        items = [{"product_id": item.product_id, "quantity": item.quantity}]
        
    cart_ref.set({"items": items})
    return {"message": f"Successfully updated cart for user: {user_id}", "cart_items": items}


@app.get("/cart/{user_id}")
def view_cart(user_id: str):
    """Retrieves the persistent cart details for a single isolated user."""
    cart_ref = db.collection("carts").document(user_id).get()
    if not cart_ref.exists:
        return {"total_items_in_cart": 0, "cart_items": []}
        
    items = cart_ref.to_dict().get("items", [])
    return {
        "total_items_in_cart": sum(item["quantity"] for item in items), 
        "cart_items": items
    }


@app.get("/cart/{user_id}/ai-recommendation")
def get_cart_recommendations(user_id: str):
    """Reads isolated user cart from Firestore and builds smart tailored recommendations."""
    cart_ref = db.collection("carts").document(user_id).get()
    if not cart_ref.exists:
        return {"recommendation": "Your cart is empty! Add products first to get AI recommendations."}

    cart_items = cart_ref.to_dict().get("items", [])
    if not cart_items:
        return {"recommendation": "Your cart is empty! Add products first to get AI recommendations."}

    client = get_gemini_client()
    
    # Retrieve product directory from database cache
    docs = db.collection("products").stream()
    all_products = [doc.to_dict()["product_details"] for doc in docs]
    
    cart_ids = [item["product_id"] for item in cart_items]
    
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


# ==========================================
# OPTIMIZED AI SMART SEARCH
# ==========================================

@app.get("/products/search/smart")
def smart_search(query: str):
    """
    HIGH SPEED VECTORS: Compares incoming user query only against 
    pre-calculated embeddings pulled directly from Firestore.
    """
    client = get_gemini_client()
    
    # 1. Generate text embedding ONLY for the single incoming search query
    query_embed_resp = client.models.embed_content(
        model="text-embedding-004",
        contents=query
    )
    # Note: query_embed_resp.embeddings is a list; we index [0] to extract values safely
    query_vector = np.array(query_embed_resp.embeddings[0].values)
    
    # 2. Pull pre-cached vectors directly out of Firestore stream
    products_ref = db.collection("products")
    docs = products_ref.stream()
    
    scored_products = []
    for doc in docs:
        data = doc.to_dict()
        prod_vector = np.array(data["embedding"])
        product_details = data["product_details"]
        
        # Calculate Cosine Similarity
        similarity = np.dot(query_vector, prod_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(prod_vector))
        scored_products.append((similarity, product_details))
        
    # 3. Sort by highest matching score first and return top 5
    scored_products.sort(key=lambda x: x[0], reverse=True)
    return {"results": [item[1] for item in scored_products[:5]]}


# ==========================================
# MULTIMODAL, TRANSLATION, AND ANALYTICS
# ==========================================

@app.post("/products/ai-classify")
def classify_product(item: AutoCategoryInput):
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

@app.post("/products/search/visual")
async def visual_search(file: UploadFile = File(...)):
    client = get_gemini_client()
    
    # Use cached products instead of external catalog to reduce latency
    docs = db.collection("products").stream()
    catalog = [doc.to_dict()["product_details"] for doc in docs]
    if not catalog:
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
            model='gemini-2.5-flash',
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

@app.post("/products/ai-translate")
def translate_description(data: RewriteDescriptionInput):
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

@app.post("/products/reviews/analyze")
def analyze_reviews(data: ReviewsInput):
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