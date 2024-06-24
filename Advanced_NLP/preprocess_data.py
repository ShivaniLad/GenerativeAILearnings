from datasets import load_dataset

review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
product_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full",
                               trust_remote_code=True)

product_lookup = {row['parent_asin']: row for row in product_dataset}


def add_product_details(data):
    asin = data['parent_asin']
    if asin in product_lookup.keys():
        product_info = {'product_title': product_lookup[asin]['title']}
        data.update(product_info)

    return data


updated_review_dataset = review_dataset.map(add_product_details)

features = ['rating', 'title', 'text', 'verified_purchase', 'product_title']
dataset = updated_review_dataset.remove_columns([x for x in updated_review_dataset['full'].features if x not in features])

dataset = dataset.filter(lambda x: x['verified_purchase'] and len(x['text']) > 100)

dataset['full'].to_pandas().to_csv("product_reviews.csv")
