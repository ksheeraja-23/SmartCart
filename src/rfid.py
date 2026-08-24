def get_product_from_rfid(rfid_uid, products):

    product = products[
        products["RFID_UID"].astype(str)
        == str(rfid_uid)
    ]

    if product.empty:
        return None

    return product.iloc[0]


def simulate_rfid_cart(
    rfid_scans,
    products,
    rules,
    recommend
):

    cart = []

    for rfid_uid in rfid_scans:

        product = get_product_from_rfid(
            rfid_uid,
            products
        )

        print("\nRFID Scanned:", rfid_uid)

        if product is None:

            print("RFID not found")

            continue

        print(
            "Product ID:",
            product["Product_ID"]
        )

        print(
            "Product Name:",
            product["Product_Name"]
        )

        print(
            "Category:",
            product["Category"]
        )

        print(
            "Price:",
            product["Price"]
        )

        product_name = product["Product_Name"]

        if product_name not in cart:

            cart.append(product_name)

        print("\nCurrent Cart:")
        print(cart)

        recommendations = recommend(
            cart,
            rules,
            top_n=5
        )

        if recommendations.empty:

            print(
                "\nNo recommendations available"
            )

        else:

            print(
                "\nRecommended Products:"
            )

            print(
                recommendations[
                    [
                        "Product",
                        "Confidence",
                        "Lift",
                        "Support"
                    ]
                ].to_string(index=False)
            )