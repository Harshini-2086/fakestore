import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings

# This automatically lods your local .env file when running on your machine,
# but ignores it safely without crashing when deploying on Cloud Run!
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

app = FastAPI(title="Fake Store API Wrapper")

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

STORE_URL = "https://api.escuelajs.co/api/v1/products"
shopping_cart = []

@app.get("/hello")
def say_hello():
    return {"message": "Hello World"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fake Store API Wrapper! Go to /docs to test endpoints."}

@app.get("/products")
def list_products():
    response = requests.get(STORE_URL)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=400, detail="Failed to fetch products from Fake Store")

@app.post("/cart/add")
def add_to_cart(item: CartInput):
    cart_item = {"product_id": item.product_id, "quantity": item.quantity}
    shopping_cart.append(cart_item)
    return {"message": "Successfully added item to cart", "added_item": cart_item}

@app.get("/cart")
def view_cart():
    return {
        "total_items_in_cart": len(shopping_cart), 
        "cart_items": shopping_cart
    }



@app.get("/cart/ai-recommendation")
def get_cart_recommendations():
    """
    Reads items currently inside the shopping cart, looks at available inventory, 
    and asks Gemini to write tailored recommendations.
    """
    if not shopping_cart:
        return {"recommendation": "Your cart is empty! Add products first to get AI recommendations."}

    #  INITIALIZE THE CLIENT HERE INSIDE THE ROUTE
    try:
        # This will now successfully grab the GEMINI_API_KEY set in your Cloud Run Console
        client = genai.Client() 
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail="GenAI Client failed to initialize. Check Cloud Run Environment Variables."
        )

    # Fetch store catalog
    response = requests.get(STORE_URL)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to load items for AI parsing.")
    all_products = response.json()

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
