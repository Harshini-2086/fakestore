from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests

app = FastAPI(title="Fake Store API Wrapper")

# --- Pydantic Data Models ---
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

# --- Configuration & In-Memory Storage ---
STORE_URL = "https://api.escuelajs.co/api/v1/products"
# Note: This list clears out whenever Cloud Run scales to zero or restarts.
shopping_cart = []

# --- API Endpoints ---

# Part 1 Requirement: Simple Hello World
@app.get("/hello")
def say_hello():
    return {"message": "Hello World"}

# Wrapper API: List all products from the external Fake Store
@app.get("/products")
def list_products():
    response = requests.get(STORE_URL)
    if response.status_code == 200:
        return response.json()
    else:
        raise HTTPException(status_code=400, detail="Failed to fetch products from Fake Store")

# Wrapper API: Add an item to the shopping cart
@app.post("/cart/add")
def add_to_cart(item: CartInput):
    cart_item = {"product_id": item.product_id, "quantity": item.quantity}
    shopping_cart.append(cart_item)
    return {"message": "Successfully added item to cart", "added_item": cart_item}

# Wrapper API: List all items currently inside the shopping cart
@app.get("/cart")
def view_cart():
    return {
        "total_items_in_cart": len(shopping_cart), 
        "cart_items": shopping_cart
    }