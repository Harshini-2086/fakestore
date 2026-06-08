import os
import sys
import asyncio
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
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
    images = Column(JSON)
    embedding = Column(Vector(3072), nullable=True)  # Gemini text-embedding-004 dimensions

class DBCartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    quantity = Column(Integer, default=1)

# ==========================================
# 3. HELPER FUNCTIONS & FALLBACKS
# ==========================================
def fetch_catalog_sync():
    """
    Synchronous fallback using requests — only called from non-async contexts
    or wrapped in asyncio.to_thread(). Returns [] on failure.
    """
    import requests
    try:
        response = requests.get("https://fakestoreapi.com/products", timeout=10)
        if response.status_code == 200:
            raw_items = response.json()
            return [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "price": float(item["price"]),
                    "description": item["description"],
                    "category": {
                        # FIX: derive a stable integer ID from the category string, not the product ID
                        "id": abs(hash(item.get("category", "general"))) % 10000,
                        "name": item.get("category", "General"),
                        "image": item.get("image", "https://placehold.co/600x400"),
                        "slug": item.get("category", "general").lower().replace(" ", "-")
                    },
                    "images": [item.get("image")]
                }
                for item in raw_items
            ]
    except Exception as e:
        print(f"--> LOGSTAMP: fetch_catalog_sync failed: {e}", flush=True)
    return []

async def fetch_catalog_async() -> list:
    """
    Async version of the catalog fetch — safe to call inside lifespan or async routes.
    Uses httpx so it never blocks the event loop.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://fakestoreapi.com/products")
            if response.status_code == 200:
                raw_items = response.json()
                return [
                    {
                        "id": item["id"],
                        "title": item["title"],
                        "price": float(item["price"]),
                        "description": item["description"],
                        "category": {
                            # FIX: stable category ID derived from name, not product ID
                            "id": abs(hash(item.get("category", "general"))) % 10000,
                            "name": item.get("category", "General"),
                            "image": item.get("image", "https://placehold.co/600x400"),
                            "slug": item.get("category", "general").lower().replace(" ", "-")
                        },
                        "images": [item.get("image")]
                    }
                    for item in raw_items
                ]
    except Exception as e:
        print(f"--> LOGSTAMP: fetch_catalog_async failed: {e}", flush=True)
    return []

def format_db_product(p: DBProduct) -> dict:
    """Maps flat SQL rows back into the standardized nested product shape."""
    img_list = p.images if isinstance(p.images, list) else ([p.images] if p.images else [])
    return {
        "id": p.id,
        "title": p.title,
        "price": p.price,
        "description": p.description,
        "category": {
            "id": p.category_id,
            "name": p.category_name or "General",
            "image": p.category_image or "https://placehold.co/600x400",
            "slug": p.category_slug or "general"
        },
        "images": img_list
    }

# ==========================================
# 4. LIFESPAN / BOOTUP SYNC MANAGEMENT
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs at startup. All blocking I/O is either replaced with async equivalents
    (httpx, asyncio.sleep) or wrapped in asyncio.to_thread() so the event loop
    is never frozen.
    """
    # 1. Enable pgvector extension and create tables — run in a thread because
    #    SQLAlchemy's synchronous engine would block the event loop otherwise.
    def setup_db():
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        Base.metadata.create_all(bind=engine)

    await asyncio.to_thread(setup_db)

    # 2. Seed catalog if empty
    db = SessionLocal()
    try:
        print("--> LOGSTAMP: Checking catalog inventory...", flush=True)

        # DB count is a blocking call — wrap it
        existing_count = await asyncio.to_thread(db.query(DBProduct).count)
        print(f"--> LOGSTAMP: Products currently in DB: {existing_count}", flush=True)

        if existing_count == 0:
            print("--> LOGSTAMP: DB empty — fetching catalog from API...", flush=True)

            # FIX: use async HTTP — never blocks the event loop
            catalog_items = await fetch_catalog_async()

            if catalog_items:
                # Instantiate Gemini client once, outside the loop
                gemini_client = get_gemini_client()
                print(f"--> LOGSTAMP: Embedding {len(catalog_items)} products...", flush=True)

                for idx, p in enumerate(catalog_items):
                    text_chunk = f"{p['title']}: {p['description']}"
                    vector_data = None

                    # FIX: Gemini SDK is synchronous — wrap each call in to_thread()
                    #      so it doesn't freeze the event loop between items
                    for attempt in range(3):
                        try:
                            embed_resp = await asyncio.to_thread(
                                gemini_client.models.embed_content,
                                model="gemini-embedding-001",
                                contents=text_chunk
                            )
                            vector_data = embed_resp.embeddings[0].values
                            break
                        except Exception as embed_err:
                            print(
                                f"--> LOGSTAMP: Gemini error on item {p['id']} "
                                f"(attempt {attempt + 1}/3): {embed_err}",
                                flush=True
                            )
                            # FIX: asyncio.sleep instead of time.sleep — never blocks event loop
                            await asyncio.sleep(2)

                    cat = p["category"]
                    new_product = DBProduct(
                        id=p["id"],
                        title=p["title"],
                        price=p["price"],
                        description=p["description"],
                        # FIX: use the correctly derived category_id from the helper, not product id
                        category_id=cat["id"],
                        category_name=cat["name"],
                        category_image=cat["image"],
                        category_slug=cat["slug"],
                        images=p["images"],
                        embedding=vector_data  # None if all retries failed — safe, column is nullable
                    )

                    # FIX: DB add is synchronous — wrap it
                    await asyncio.to_thread(db.add, new_product)

                    # Pace requests to stay under Gemini free-tier limits (non-blocking)
                    await asyncio.sleep(0.3)

                # FIX: commit is synchronous — wrap it
                await asyncio.to_thread(db.commit)
                print("--> LOGSTAMP: Catalog seeding complete!", flush=True)
            else:
                print("--> LOGSTAMP: API returned no items. DB remains empty.", flush=True)
        else:
            print("--> LOGSTAMP: Catalog already populated. Skipping seed.", flush=True)

    except Exception as e:
        print(f"--> LOGSTAMP: Lifespan seed error: {e}", file=sys.stderr, flush=True)
        await asyncio.to_thread(db.rollback)
    finally:
        await asyncio.to_thread(db.close)

    yield  # App runs here

    # Shutdown — nothing extra needed, connection pool handles cleanup
    print("--> LOGSTAMP: Shutting down gracefully.", flush=True)


app = FastAPI(title="AI-Powered E-Commerce Store Platform", lifespan=lifespan)

# ==========================================
# 5. API ROUTES
# ==========================================

@app.get("/")
def home():
    return {"status": "online", "message": "Welcome to the Production AI E-Commerce API"}


@app.get("/products")
async def list_products(db: Session = Depends(get_db)):
    """
    Returns all products. Falls back to live API if DB is empty or unreachable.
    DB query is wrapped in to_thread() to stay non-blocking.
    """
    try:
        products = await asyncio.to_thread(db.query(DBProduct).all)
        if not products:
            print("--> LOGSTAMP: DB empty — using live API fallback.", flush=True)
            # FIX: use async fetch in this async route
            return await fetch_catalog_async()
        return [format_db_product(p) for p in products]
    except Exception as db_err:
        print(f"--> LOGSTAMP: DB read failed ({db_err}) — falling back to live API.", flush=True)
        return await fetch_catalog_async()


@app.get("/products/search/smart")
async def smart_search(query: str, db: Session = Depends(get_db)):
    """
    Semantic vector search via pgvector cosine distance.
    Gemini embedding call is wrapped in to_thread() — non-blocking.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Search query string is required.")

    try:
        gemini_client = get_gemini_client()

        # FIX: wrap synchronous Gemini call
        embed_resp = await asyncio.to_thread(
            gemini_client.models.embed_content,
            model="gemini-embedding-001",
            contents=query
        )
        query_vector = embed_resp.embeddings[0].values

        # FIX: wrap synchronous DB query
        def run_vector_query():
            stmt = (
                select(DBProduct)
                .order_by(DBProduct.embedding.cosine_distance(query_vector))
                .limit(5)
            )
            return db.scalars(stmt).all()

        results = await asyncio.to_thread(run_vector_query)

        if not results:
            return {"message": "No matching products found.", "results": []}

        return [format_db_product(p) for p in results]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@app.get("/cart")
async def view_cart(db: Session = Depends(get_db)):
    """Returns all items currently in the cart."""
    return await asyncio.to_thread(db.query(DBCartItem).all)


@app.post("/cart/add")
async def add_to_cart(product_id: int, quantity: int = 1, db: Session = Depends(get_db)):
    """Adds a product to the cart or increments quantity if it already exists."""
    def _add():
        item = db.query(DBCartItem).filter(DBCartItem.product_id == product_id).first()
        if item:
            item.quantity += quantity
        else:
            item = DBCartItem(product_id=product_id, quantity=quantity)
            db.add(item)
        db.commit()
        return item

    await asyncio.to_thread(_add)
    return {"message": "Cart updated successfully.", "product_id": product_id, "quantity": quantity}