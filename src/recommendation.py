import pandas as pd

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)


def create_basket(transactions):

    basket = (
        transactions
        .groupby(
            ["Transaction_ID", "Product_Name"]
        )["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = basket > 0

    return basket


def train_recommendation_model(basket):

    frequent_itemsets = apriori(
        basket,
        min_support=0.05,
        use_colnames=True
    )

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.4
    )

    rules = rules[
        [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift"
        ]
    ]

    rules = rules[
        (rules["support"] >= 0.05) &
        (rules["confidence"] >= 0.4) &
        (rules["lift"] >= 1.2)
    ]

    return frequent_itemsets, rules


def recommend(cart_products, rules, top_n=5):

    recommendations = []

    cart_products = set(cart_products)

    for _, rule in rules.iterrows():

        antecedents = set(rule["antecedents"])
        consequents = set(rule["consequents"])

        if antecedents.issubset(cart_products):

            for product in consequents:

                if product not in cart_products:

                    recommendations.append({
                        "Product": product,
                        "Confidence": rule["confidence"],
                        "Lift": rule["lift"],
                        "Support": rule["support"]
                    })

    if not recommendations:
        return pd.DataFrame()

    recommendations = pd.DataFrame(
        recommendations
    )

    recommendations = (
        recommendations
        .sort_values(
            ["Confidence", "Lift"],
            ascending=[False, False]
        )
        .drop_duplicates("Product")
        .head(top_n)
    )

    return recommendations