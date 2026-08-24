import pandas as pd

from sklearn.model_selection import train_test_split

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)


def create_training_rules(train_transactions):

    basket = (
        train_transactions
        .groupby(
            ["Transaction_ID", "Product_Name"]
        )["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = basket > 0

    print("\nTraining basket shape:",
          basket.shape)

    frequent_itemsets = apriori(
        basket,
        min_support=0.05,
        use_colnames=True
    )

    print("\nTraining frequent itemsets:")
    print(frequent_itemsets.head())

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

    print("\nNumber of training rules:",
          len(rules))

    return rules


def evaluate_model(transactions):

    transaction_ids = (
        transactions["Transaction_ID"]
        .astype(str)
        .unique()
        .tolist()
    )

    train_ids, test_ids = train_test_split(
        transaction_ids,
        test_size=0.2,
        random_state=42
    )

    train_transactions = transactions[
        transactions["Transaction_ID"]
        .astype(str)
        .isin(train_ids)
    ].copy()

    test_transactions = transactions[
        transactions["Transaction_ID"]
        .astype(str)
        .isin(test_ids)
    ].copy()

    print("\nTraining transactions:",
          train_transactions["Transaction_ID"].nunique())

    print("Testing transactions:",
          test_transactions["Transaction_ID"].nunique())

    print("Total transactions:",
          transactions["Transaction_ID"].nunique())

    train_rules = create_training_rules(
        train_transactions
    )

    precision, recall = calculate_metrics(
        test_transactions,
        train_rules
    )

    print("\nRecommendation Evaluation")

    return train_rules, precision, recall


def calculate_metrics(test_transactions, rules):

    from src.recommendation import recommend

    precision_scores = []
    recall_scores = []

    for transaction_id in test_transactions[
        "Transaction_ID"
    ].unique():

        products = (
            test_transactions[
                test_transactions["Transaction_ID"]
                == transaction_id
            ]["Product_Name"]
            .unique()
            .tolist()
        )

        if len(products) < 2:
            continue

        cart = products[:-1]
        actual_product = products[-1]

        recommendations = recommend(
            cart,
            rules,
            top_n=5
        )

        recommended_products = (
            recommendations["Product"]
            .tolist()
            if not recommendations.empty
            else []
        )

        relevant = (
            1
            if actual_product in recommended_products
            else 0
        )

        precision_scores.append(
            relevant / 5
        )

        recall_scores.append(
            relevant
        )

    precision = (
        sum(precision_scores) /
        len(precision_scores)
        if precision_scores
        else 0
    )

    recall = (
        sum(recall_scores) /
        len(recall_scores)
        if recall_scores
        else 0
    )

    return precision, recall