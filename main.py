import os
from urllib import response
from google import genai
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

from fastapi import HTTPException, status

@app.post("/cart/add")
def add_to_cart(item: CartInput):
    #getting the product catalog to validate the entered product_id
    response = requests.get(STORE_URL)
    if response.status_code != 200:
        raise HTTPException(
            status_code=500, 
            detail="Failed to fetch products from store catalog for validation."
        )
    all_products = response.json()
    # 1. Extract all valid product IDs from your catalog
    valid_ids = [p["id"] for p in all_products]
    
    # 2. Check if the user's product_id exists in that list
    if item.product_id not in valid_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product with ID {item.product_id} does not exist."
        )
        
    # 3. If valid, add the single item sent by the HTTP request
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

    try:
        # using os.environ.get to get the gemini API key from the enviornment
        apikey = os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=apikey) #passing it directly to the constructor of the genai client
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

    #Isolate product IDs in cart
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
