import os
import sys
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, select, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector

# ==========================================
# 1. DATABASE & CLOUD CONFIGURATION
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("CRITICAL ERROR: DATABASE_URL environment variable is missing!")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable is missing!")
    return genai.Client(api_key=api_key)

# ==========================================
# 2. DATA MODELS (PostgreSQL / SQLAlchemy)
# ==========================================
class DBProduct(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    description = Column(String)
    category_id = Column(Integer)
    category_name = Column(String)
    category_image = Column(String)
    category_slug = Column(String)
    images = Column(JSON)  # Stores lists of image URLs safely
    embedding = Column(Vector(3072))  # Gemini dimensions

class DBCartItem(Base):
    __tablename__ = "cart_items"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    quantity = Column(Integer, default=1)

# ==========================================
# 3. HELPER FUNCTIONS & FALLBACKS
# ==========================================
def fetch_catalog():
    """Fallback function to fetch raw API items if DB is empty or unreachable."""
    response = requests.get("https://fakestoreapi.com/products", timeout=10)
    if response.status_code == 200:
        raw_items = response.json()
        # Format strings cleanly to match expected frontend structure
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "price": float(item["price"]),
                "description": item["description"],
                "category": {
                    "id": item.get("id"),
                    "name": item.get("category", "General"),
                    "image": item.get("image", "https://placehold.co/600x400"),
                    "slug": item.get("category", "general").lower().replace(" ", "-")
                },
                "images": [item.get("image")]
            }
            for item in raw_items
        ]
    return []

def format_db_product(p: DBProduct):
    """Maps custom flat SQL tables safely back into standardized nested objects."""
    img_list = p.images if isinstance(p.images, list) else [p.images] if p.images else []
    return {
        "id": p.id,
        "title": p.title,
        "price": p.price,
        "description": p.description,
        "category": {
            "id": p.category_id if p.category_id is not None else p.id,
            "name": p.category_name if p.category_name else "General",
            "image": p.category_image if p.category_image else "https://placehold.co/600x400",
            "slug": p.category_slug if p.category_slug else "general"
        },
        "images": img_list
    }

# ==========================================
# 4. LIFESPAN / BOOTUP SYNC MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes atomic structural setups immediately upon app boot sequence."""
    import time
    
    # 1. Ensure the Vector extension is enabled in Postgres
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # 2. Automatically generate matching table structures if absent
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        print("--> LOGSTAMP: Checking catalog inventory with PostgreSQL database...", flush=True)
        existing_count = db.query(DBProduct).count()
        print(f"--> LOGSTAMP: Current products in database: {existing_count}", flush=True)
        
        if existing_count == 0:
            print("--> LOGSTAMP: Database is empty. Attempting live catalog pull...", flush=True)
            response = requests.get("https://fakestoreapi.com/products", timeout=10)
            
            if response.status_code == 200:
                catalog_items = response.json()
                client = get_gemini_client()
                print(f"--> LOGSTAMP: Found {len(catalog_items)} items. Calculating embeddings with rate-limit pacing...", flush=True)
                
                for idx, p in enumerate(catalog_items):
                    text_chunk = f"{p.get('title', '')}: {p.get('description', '')}"
                    vector_data = None
                    
                    # Retry logic for Gemini API to avoid free-tier rate limits
                    for attempt in range(3):
                        try:
                            embed_resp = client.models.embed_content(
                                model="gemini-embedding-001",
                                contents=text_chunk
                            )
                            vector_data = embed_resp.embeddings[0].values
                            break  # Success! Break the retry loop
                        except Exception as embed_err:
                            print(f"--> LOGSTAMP: Gemini hit a hiccup on item {p.get('id')} (Attempt {attempt+1}/3): {embed_err}", flush=True)
                            time.sleep(2)  # Cool down before retrying
                    
                    # Map the product data cleanly
                    cat_name = p.get('category', 'General')
                    new_product = DBProduct(
                        id=p["id"],
                        title=p["title"],
                        price=float(p["price"]),
                        description=p["description"],
                        category_id=p.get("id"),
                        category_name=cat_name,
                        category_image=str(p.get("image", "https://placehold.co/600x400")),
                        category_slug=cat_name.lower().replace(" ", "-"),
                        images=[p.get("image")] if "image" in p else p.get("images", []),
                        embedding=vector_data  # Will be saved as None if embeddings completely fail, preventing a crash!
                    )
                    db.add(new_product)
                    
                    # Small 300ms pause between items to stay well under free tier rate limits
                    time.sleep(0.3)
                
                db.commit()
                print("--> LOGSTAMP: Database inventory initialization successful!", flush=True)
        else:
            print("--> LOGSTAMP: Local cache verification completed. Sync skipped.", flush=True)
    except Exception as e:
        print(f"--> LOGSTAMP: Non-blocking catalog sync error at boot: {str(e)}", file=sys.stderr, flush=True)
    finally:
        db.close()
    yield

app = FastAPI(title="AI-Powered E-Commerce Store Platform", lifespan=lifespan)

# ==========================================
# 5. API ROUTES
# ==========================================

@app.get("/")
def home():
    return {"status": "online", "message": "Welcome to the Production AI E-Commerce API"}

@app.get("/products")
def list_products(db: Session = Depends(get_db)):
    """Fetches store products cleanly with zero 500 crashes if DB state fails."""
    try:
        products = db.query(DBProduct).all()
        if not products:
            print("--> LOGSTAMP: Database empty. Running raw fallback...", flush=True)
            return fetch_catalog()
        return [format_db_product(p) for p in products]
    except Exception as db_err:
        print(f"--> LOGSTAMP: Database read failed ({db_err}). Falling back to live API...", flush=True)
        return fetch_catalog()

@app.get("/products/search/smart")
def smart_search(query: str, db: Session = Depends(get_db)):
    """AI Vector semantic search using native pgvector cosine distances."""
    if not query:
        raise HTTPException(status_code=400, detail="Search query string missing")
    
    try:
        client = get_gemini_client()
        embed_resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=query
        )
        query_vector = embed_resp.embeddings[0].values
        
        # Calculate cosine similarity using native SQL extensions (<=> distance metric)
        stmt = select(DBProduct).order_by(DBProduct.embedding.cosine_distance(query_vector)).limit(5)
        results = db.scalars(stmt).all()
        
        if not results:
            return {"message": "No matching concepts found", "results": []}
            
        return [format_db_product(p) for p in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic matching pipeline failed: {str(e)}")

@app.get("/cart")
def view_cart(db: Session = Depends(get_db)):
    return db.query(DBCartItem).all()

@app.post("/cart/add")
def add_to_cart(product_id: int, quantity: int = 1, db: Session = Depends(get_db)):
    item = db.query(DBCartItem).filter(DBCartItem.product_id == product_id).first()
    if item:
        item.quantity += quantity
    else:
        item = DBCartItem(product_id=product_id, quantity=quantity)
        db.add(item)
    db.commit()
    return {"message": "Cart persistence storage synced", "product_id": product_id}