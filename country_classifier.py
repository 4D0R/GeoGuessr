from google.cloud import storage
from tqdm import tqdm

storage_client = storage.Client()
bucket = storage_client.get_bucket("geoguessr-imgs")

buckets = storage_client.list_blobs("geoguessr-imgs", prefix=f"small_scrape/", delimiter="/")
for _ in buckets:
    pass

print("Hello and welcome to the country classifier.")
print("To rename a folder, type the proper country name.")
print("To delete a folder, type DELETE.")
print("To skip a folder, press enter.")
print("To exit, press CTRL+C.")
print("Have fun!")

for prefix in buckets.prefixes:
    user_country = input(f"{prefix}: ")
    if user_country:
        blobs = tqdm(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            blobs.set_description(f"Copying {blob.name.split('/')[-1]}")
            bucket.rename_blob(blob, new_name=f"small_scrape/{user_country} /{blob.name.split('/')[-1]}")
    elif user_country == "DELETE":
        blobs = tqdm(bucket.list_blobs(prefix=prefix))
        for blob in blobs:
            blobs.set_description(f"Deleting {blob.name.split('/')[-1]}")
            blob.delete()
    else:
        print("Skipping")