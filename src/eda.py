import pandas as pd
import matplotlib.pyplot as plt


def run_eda(products, transactions):

    print("\nProducts dataset shape:", products.shape)
    print("Transactions dataset shape:", transactions.shape)

    print("\nProduct columns:")
    print(products.columns.tolist())

    print("\nTransaction columns:")
    print(transactions.columns.tolist())

    print("\nProduct dataset information:")
    products.info()

    print("\nTransaction dataset information:")
    transactions.info()

    print("\nUnique Product IDs:",
          products["Product_ID"].nunique())

    duplicate_rfid = products["RFID_UID"].duplicated().sum()

    print("\nDuplicate RFID UIDs:", duplicate_rfid)

    print("Unique RFID UIDs:",
          products["RFID_UID"].nunique())

    unique_transactions = transactions["Transaction_ID"].nunique()
    unique_customers = transactions["Customer_ID"].nunique()
    unique_products = transactions["Product_ID"].nunique()

    print("\nUnique transactions:", unique_transactions)
    print("Unique customers:", unique_customers)
    print("Unique products purchased:", unique_products)

    print("\nProduct category distribution:")
    print(products["Category"].value_counts())

    print("\nPrice statistics:")
    print(products["Price"].describe())

    valid_rfids = set(products["RFID_UID"])

    invalid_rfids = transactions[
        ~transactions["RFID_UID"].isin(valid_rfids)
    ]

    print("\nInvalid RFID records:",
          len(invalid_rfids))

    valid_product_ids = set(products["Product_ID"])

    invalid_products = transactions[
        ~transactions["Product_ID"].isin(valid_product_ids)
    ]

    print("Invalid Product ID records:",
          len(invalid_products))

    rfid_product_check = (
        transactions
        .groupby("RFID_UID")["Product_ID"]
        .nunique()
    )

    print(
        "\nMaximum Product IDs associated with one RFID:",
        rfid_product_check.max()
    )

    product_rfid_check = (
        transactions
        .groupby("Product_ID")["RFID_UID"]
        .nunique()
    )

    print(
        "Maximum RFID UIDs associated with one Product:",
        product_rfid_check.max()
    )

    timestamp = pd.to_datetime(
        transactions["Timestamp"],
        errors="coerce"
    )

    quantity = pd.to_numeric(
        transactions["Quantity"],
        errors="coerce"
    )

    price = pd.to_numeric(
        transactions["Price"],
        errors="coerce"
    )

    total_value = pd.to_numeric(
        transactions["Total_Value"],
        errors="coerce"
    )

    print("\nTimestamp datatype:", timestamp.dtype)
    print("Quantity datatype:", quantity.dtype)
    print("Price datatype:", price.dtype)
    print("Total Value datatype:", total_value.dtype)

    invalid_quantity = transactions[
        quantity <= 0
    ]

    invalid_price = transactions[
        price <= 0
    ]

    print("\nInvalid quantity records:",
          len(invalid_quantity))

    print("Invalid price records:",
          len(invalid_price))

    most_purchased = (
        transactions
        .groupby("Product_Name")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nMost purchased products:")
    print(most_purchased)

    category_sales = (
        transactions
        .groupby("Category")["Quantity"]
        .sum()
        .sort_values(ascending=False)
    )

    print("\nProduct category sales:")
    print(category_sales)

    products_per_category = (
        products["Category"]
        .value_counts()
    )

    print("\nNumber of products in each category:")
    print(products_per_category)

    transactions_per_customer = (
        transactions
        .groupby("Customer_ID")["Transaction_ID"]
        .nunique()
    )

    print("\nTransactions per customer:")
    print(transactions_per_customer.describe())

    products_per_transaction = (
        transactions
        .groupby("Transaction_ID")["Product_ID"]
        .nunique()
    )

    print("\nProducts per transaction:")
    print(products_per_transaction.describe())

    top_transaction_products = (
        transactions
        .groupby("Product_Name")["Transaction_ID"]
        .nunique()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTop 10 products by number of transactions:")
    print(top_transaction_products)

    total_sales = total_value.sum()

    average_transaction_value = (
        transactions
        .groupby("Transaction_ID")["Total_Value"]
        .sum()
        .mean()
    )

    print("\nTotal sales value:")
    print(total_sales)

    print("\nAverage transaction value:")
    print(average_transaction_value)

    print("\nFinal dataset summary:")

    print("Total products:", len(products))
    print("Unique RFID tags:", products["RFID_UID"].nunique())
    print("Unique product IDs:", products["Product_ID"].nunique())
    print("Unique transactions:", transactions["Transaction_ID"].nunique())
    print("Unique customers:", transactions["Customer_ID"].nunique())
    print("Transaction-product records:", len(transactions))

    plt.figure(figsize=(10, 6))
    most_purchased.sort_values().plot(kind="barh")
    plt.title("Top 10 Most Purchased Products")
    plt.xlabel("Quantity Sold")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    category_sales.sort_values().plot(kind="barh")
    plt.title("Sales Quantity by Product Category")
    plt.xlabel("Quantity Sold")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    products_per_category.sort_values().plot(kind="barh")
    plt.title("Number of Products per Category")
    plt.xlabel("Number of Products")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    transactions_per_customer.value_counts().sort_index().plot(
        kind="bar"
    )
    plt.title("Distribution of Transactions per Customer")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    products["Price"].plot(
        kind="hist",
        bins=10
    )
    plt.title("Product Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.show()