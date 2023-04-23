from selenium import webdriver
from selenium.webdriver.common.by import By
from constants import countries
from pathlib import Path
from time import sleep
from multiprocessing import Pool
import os


def main():
    # Multiprocessing
    with Pool() as pool:
        pool.map(scrape_country, countries.keys())


def scrape_country(country_code):
    print(f"Scraping {countries[country_code]}...")

    # Create a directory for the results
    results = Path("./images")
    os.makedirs(results, exist_ok=True)

    # Create a new webdriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    # Go to country
    sleep(1)
    driver.get(f"https://randomstreetview.com/{country_code}")
    driver.set_window_size(800, 450)
    sleep(10)

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
    curr_img = 0
    while curr_img < 1:
        if not os.path.isfile(results / country_path / f"{curr_img}.jpg"):
            sleep(3)
            for className in ["gm-style"]:
                driver.execute_script(
                    f"Array.prototype.map.call(document.getElementsByClassName('{className}'), x => x.style.display = 'flex');"
                )
            driver.save_screenshot(results / country_path / f"{curr_img}.jpg")
            
            
            driver.find_element(By.ID, "next").click()
        curr_img += 1

    # Close driver
    driver.close()


if __name__ == "__main__":
    main()