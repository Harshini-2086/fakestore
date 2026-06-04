import os
import json
import requests
import numpy as np
from contextlib import asynccontextmanager
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Depends
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings

# --- DATABASE ENGINE INTEGRATIONS ---
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, select, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector


# Environment Configurations
class Settings(BaseSettings):
    # Fallback default value connects locally if no environmental variable overrides it
    DATABASE_URL: str = "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Setup relational connection pools and core base declarations
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Request dependency helper to allocate distinct transaction workspaces safely
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Database Tabular Formats ---
class DBProduct(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String)
    category_id = Column(Integer)
    category_name = Column(String)
    category_image = Column(String)
    category_slug = Column(String)
    images = Column(JSON)  
    embedding = Column(Vector(768), nullable=True) # gemini-embedding-001 output shape

class DBCartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, default=1, index=True) 
    product_id = Column(Integer, nullable=False)
    quantity = Column(Integer, default=1)

# --- Lifespan Event Initializer ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes atomic structural setups immediately upon app boot sequence."""
    # 1. Reach out to database engine and unlock vector extensions if not built
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # 2. Automatically generate matching table structures if absent
    Base.metadata.create_all(bind=engine)
    
    # 3. Synchronize initial item catalog from external source safely
    db = SessionLocal()
    try:
        print("Checking/Syncing catalog inventory with PostgreSQL database...")
        response = requests.get("https://api.escuelajs.co/api/v1/products")
        if response.status_code == 200:
            catalog_items = response.json()
            client = get_gemini_client()
            
            for p in catalog_items:
                existing = db.query(DBProduct).filter(DBProduct.id == p["id"]).first()
                if not existing:
                    # Request embedding values upfront during structural caching tasks
                    text_chunk = f"{p['title']}: {p['description']}"
                    try:
                        embed_resp = client.models.embed_content(
                            model="gemini-embedding-001",
                            contents=text_chunk
                        )
                        vector_data = embed_resp.embeddings[0].values
                    except Exception:
                        vector_data = None

                    new_product = DBProduct(
                        id=p["id"],
                        title=p["title"],
                        price=float(p["price"]),
                        description=p["description"],
                        category_id=p["category"]["id"],
                        category_name=p["category"]["name"],
                        category_image=str(p["category"]["image"]),
                        category_slug=p["category"].get("slug", ""),
                        images=p["images"],
                        embedding=vector_data
                    )
                    db.add(new_product)
            db.commit()
            print("Database inventory up to date.")
    except Exception as e:
        print(f"Non-blocking catalog sync warning at boot: {str(e)}")
    finally:
        db.close()
        
    yield

app = FastAPI(title="Fake Store AI Platform Wrapper", lifespan=lifespan)

# --- Standard Core Pydantic Models ---
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

class DynamicTranslateInput(BaseModel):
    target_language: str = "Hindi"
    tone: str = "Semi-Professional, friendly, and concise"

class ReviewsInput(BaseModel):
    reviews: list[str]

# --- Structural Schemas for AI Responses ---
class ReviewSummarySchema(BaseModel):
    overall_sentiment: str  
    pros: list[str]
    cons: list[str]
    verdict: str

class ItemRecommendation(BaseModel):
    cart_product_id: int
    cart_product_title: str
    recommended_product_ids: list[int]
    pitch_to_user: str

class MultiCartAnalysisSchema(BaseModel):
    recommendations_by_item: list[ItemRecommendation]
    global_reasoning: str

# --- Helper Functions ---
def get_gemini_client():
    apikey = os.environ.get("GEMINI_API_KEY", settings.GEMINI_API_KEY).strip()
    if not apikey:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY environment variable is missing."
        )
    return genai.Client(api_key=apikey, vertexai=False)

def format_db_product(p: DBProduct):
    """Maps custom flat SQL tables back into standardized nested objects."""
    return {
        "id": p.id,
        "title": p.title,
        "price": p.price,
        "description": p.description,
        "category": {
            "id": p.category_id,
            "name": p.category_name,
            "image": p.category_image,
            "slug": p.category_slug
        },
        "images": p.images
    }

# --- Standard Route Handlers ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Store AI API Wrapper! Go to /docs to test endpoints."}

@app.get("/products")
def list_products(db: Session = Depends(get_db)):
    products = db.query(DBProduct).all()
    return [format_db_product(p) for p in products]

@app.post("/cart/add")
def add_to_cart(item: CartInput, db: Session = Depends(get_db)):
    product = db.query(DBProduct).filter(DBProduct.id == item.product_id).first()
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Product with ID {item.product_id} does not exist in local inventory tables."
        )
    
    existing_cart_item = db.query(DBCartItem).filter(DBCartItem.product_id == item.product_id).first()
    if existing_cart_item:
        existing_cart_item.quantity += item.quantity
    else:
        db.add(DBCartItem(product_id=item.product_id, quantity=item.quantity))
        
    db.commit()
    return {"message": "Successfully added item to database cart"}

@app.get("/cart")
def view_cart(db: Session = Depends(get_db)):
    cart_items = db.query(DBCartItem).all()
    output = [{"product_id": c.product_id, "quantity": c.quantity} for c in cart_items]
    return {
        "total_items_in_cart": len(output), 
        "cart_items": output
    }

# --- FEATURE 1: Combined Cart Analysis, Categorization & Inventory Matching ---
@app.get("/cart/ai-analyze-and-match")
def analyze_cart_and_match_inventory(db: Session = Depends(get_db)):
    cart_items = db.query(DBCartItem).all()
    if not cart_items:
        raise HTTPException(status_code=400, detail="Your database cart is empty!")
        
    client = get_gemini_client()
    cart_ids = [item.product_id for item in cart_items]
    db_cart_products = db.query(DBProduct).filter(DBProduct.id.in_(cart_ids)).all()
    
    items_in_cart = [
        {"id": p.id, "title": p.title, "description": p.description} 
        for p in db_cart_products
    ]
    
    db_catalog = db.query(DBProduct).filter(DBProduct.id.notin_(cart_ids)).limit(30).all()
    available_catalog = [
        {"id": p.id, "title": p.title, "category": p.category_name, "description": p.description}
        for p in db_catalog
    ]

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
                response_schema=MultiCartAnalysisSchema 
            )
        )
        
        ai_response = json.loads(completion.text)
        item_recommendations = ai_response.get("recommendations_by_item", [])
        
        final_output = []
        for rec in item_recommendations:
            rec_ids = rec.get("recommended_product_ids", [])
            matched_db_products = db.query(DBProduct).filter(DBProduct.id.in_(rec_ids)).all()
            
            final_output.append({
                "cart_product_id": rec.get("cart_product_id"),
                "cart_product_title": rec.get("cart_product_title"),
                "pitch_to_user": rec.get("pitch_to_user"),
                "similar_matched_products": [format_db_product(p) for p in matched_db_products]
            })
            
        return {
            "global_reasoning": ai_response.get("global_reasoning"),
            "recommendations": final_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Matching pipeline failed: {str(e)}")
    
# --- FEATURE 2: Smart Search via Native pgvector Database Operators ---
@app.get("/products/search/smart")
def smart_search(query: str, db: Session = Depends(get_db)):
    """Executes a semantic vector search query natively inside the SQL database layer."""
    client = get_gemini_client()
    
    query_embed_resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )
    query_vector = query_embed_resp.embeddings[0].values
    
    # Executes clean cosine calculations directly inside PostgreSQL using the <=> operator
    stmt = (
        select(DBProduct)
        .order_by(DBProduct.embedding.cosine_distance(query_vector))
        .limit(15)
    )
    
    results = db.scalars(stmt).all()
    return {"results": [format_db_product(p) for p in results]}

# --- FEATURE 3: Cart Context Localization & Translation ---
@app.post("/cart/translate-descriptions")
def translate_cart_descriptions(data: DynamicTranslateInput, db: Session = Depends(get_db)):
    cart_items = db.query(DBCartItem).all()
    if not cart_items:
        raise HTTPException(status_code=400, detail="Shopping cart database table is empty.")
        
    client = get_gemini_client()
    cart_ids = [item.product_id for item in cart_items]
    db_products = db.query(DBProduct).filter(DBProduct.id.in_(cart_ids)).all()
    
    items_to_translate = [
        {"id": p.id, "title": p.title, "description": p.description}
        for p in db_products
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

# --- FEATURE 4: Multimodal Visual Shopping Search ---
@app.post("/products/search/visual")
async def visual_search(file: UploadFile = File(...), db: Session = Depends(get_db)):
    client = get_gemini_client()
    db_catalog = db.query(DBProduct).limit(30).all()
    light_catalog = [{"id": p.id, "title": p.title, "category": p.category_name} for p in db_catalog]
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
        
        results = db.query(DBProduct).filter(DBProduct.id.in_(matched_ids)).all()
        return {"matched_products": [format_db_product(p) for p in results]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- FEATURE 5: AI Review Summary & Sentiment Analytics ---
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