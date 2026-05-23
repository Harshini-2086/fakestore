import requests

print("--- INTERACTIVE TESTING SCRIPT INITIALIZED ---")

# LOCAL TESTING: Use this while running 'uvicorn main:app --reload' on your machine
BASE_URL = "http://127.0.0.1:8000"

# LIVE TESTING: Uncomment the line below and paste your live Cloud Run URL once deployed!
# BASE_URL = "https://fake-store-api-xxxxxx.a.run.app" 

def handle_user_selection(action: str, data: dict = None):
    try:
        if action == "view_products":
            response = requests.get(f"{BASE_URL}/products")
            return response.json()
        elif action == "add_to_cart":
            response = requests.post(f"{BASE_URL}/cart/add", json=data)
            return response.json()
        elif action == "view_cart":
            response = requests.get(f"{BASE_URL}/cart")
            return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": f"Could not connect to server at {BASE_URL}. Is it running?"}

def main_menu():
    print("\n--- Fake Store Testing Menu ---")
    print("1. View Available Products")
    print("2. Add a Product to Cart")
    print("3. View Shopping Cart")
    print("4. Exit")
    
    choice = input("Select an option (1-4): ")
    
    if choice == "1":
        print("\nFetching products from server...")
        print(handle_user_selection("view_products"))
    elif choice == "2":
        try:
            prod_id = int(input("Enter Product ID: "))
            qty = int(input("Enter Quantity: "))
            payload = {"product_id": prod_id, "quantity": qty}
            print(handle_user_selection("add_to_cart", data=payload))
        except ValueError:
            print("Please enter valid numeric values.")
    elif choice == "3":
        print(handle_user_selection("view_cart"))
    elif choice == "4":
        print("Goodbye!")
        return False
    return True

if __name__ == "__main__":
    running = True
    while running:
        running = main_menu()