from selenium import webdriver
from selenium.webdriver.common.by import By
from constants import countries
from pathlib import Path
from time import sleep
from multiprocessing import Pool
import os
from google.cloud import storage

IMGS_PER_COUNTRY = 1000
FOLDER = "small_scrape"

storage_client = storage.Client()
bucket = storage_client.get_bucket("geoguessr-imgs")

def main():
    # Multiprocessing
    with Pool() as pool:
        pool.map(scrape_country, countries.keys())


def scrape_country(country_code):
    # Create a directory for the results
    results = Path("./data/scraped_images")
    os.makedirs(results, exist_ok=True)

    # Create a new webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    # Go to country
    sleep(1)
    driver.get(f"https://randomstreetview.com/{country_code}")
    driver.set_window_size(800, 450)
    sleep(1)

    # Create a directory for results
    country_path = Path(countries[country_code])
    os.makedirs(results / country_path, exist_ok=True)

    # Remove watermarks from page
    for id in [
        "controls",
        "map_canvas",
        "adnotice",
        "ad",
        "address",
        "minimaximize",
        "share",
    ]:
        driver.execute_script(f"document.getElementById('{id}').style.opacity = 0;")
    for className in ["gm-fullscreen-control", "intro_splash", "gmnoprint"]:
        driver.execute_script(
            f"Array.prototype.map.call(document.getElementsByClassName('{className}'), x => x.style.opacity = 0);"
        )

    # Save screenshot of current image and then click to get next one!
    # num_imgs = len([name for name in os.listdir(results / country_path) if os.path.isfile(name)])
    num_imgs = len(list(bucket.list_blobs(prefix=f"{FOLDER}/{countries[country_code]}")))
    index = 0
    if num_imgs < IMGS_PER_COUNTRY:
        print(f"Scraping {IMGS_PER_COUNTRY-num_imgs} images from {countries[country_code]}...")
    else:
        driver.close()
        return
    while num_imgs < IMGS_PER_COUNTRY:
        location_data = driver.execute_script(f"return randomLocations.{country_code}")
        if (index >= len(location_data)):
            driver.refresh()
            sleep(1)
            index = 0
            continue
        file_name = f"{location_data[index]['lat']}x{location_data[index]['lng']}"
        index += 1

        if not os.path.isfile(results / country_path / f"{file_name}.jpg"):
            sleep(3)
            for className in ["gm-style"]:
                driver.execute_script(
                    f"Array.prototype.map.call(document.getElementsByClassName('{className}'), x => x.style.display = 'flex');"
                )
            driver.save_screenshot(results / country_path / f"{file_name}.jpg")
            
            blob = bucket.blob(f"{FOLDER}/{country_path} /{file_name}.jpg")
            blob.upload_from_filename(results / country_path / f"{file_name}.jpg")
            
            driver.find_element(By.ID, "next").click()
        num_imgs += 1

    # Close driver
    driver.close()
    print(f"    Done Scraping {countries[country_code]}.")


if __name__ == "__main__":
    main()