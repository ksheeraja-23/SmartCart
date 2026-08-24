import pandas as pd

from src.eda import run_eda
from src.recommendation import create_basket, train_recommendation_model, recommend
from src.evaluation import evaluate_model
from src.rfid import simulate_rfid_cart

products = pd.read_csv("data/products.csv")
transactions = pd.read_csv("data/transactions.csv")

run_eda(products, transactions)

basket = create_basket(transactions)

frequent_itemsets, rules = train_recommendation_model(basket)

print("\nNumber of useful rules:", len(rules))

cart = [
    "Amul Milk 1L",
    "Britannia Bread"
]

print("\nCurrent Cart:")
print(cart)

recommendations = recommend(
    cart,
    rules,
    top_n=5
)

print("\nRecommended Products:")

if recommendations.empty:
    print("No recommendations available")
else:
    print(recommendations.to_string(index=False))

train_rules, precision, recall = evaluate_model(
    transactions
)

print("\nPrecision@5:", round(precision * 100, 2), "%")
print("Recall@5:", round(recall * 100, 2), "%")

rfid_scans = [
    products["RFID_UID"].iloc[0],
    products["RFID_UID"].iloc[1],
    products["RFID_UID"].iloc[4]
]

simulate_rfid_cart(
    rfid_scans,
    products,
    train_rules,
    recommend
)