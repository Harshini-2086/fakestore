from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests

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