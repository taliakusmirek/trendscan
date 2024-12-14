import os
import time
import random
import json
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Create dataset

# Scrape e-commerce sites for product images using Selenium
class scrapeSites:
    def __init__(self, websites):
        self.websites = websites
        self.count = 0
        self.output_dir = "json"
        self.image_dir = "images"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        self.init_driver()

    def init_driver(self):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        custom_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        options.add_argument(f"user-agent={custom_user_agent}")
        self.driver = webdriver.Chrome(options=options)





    # Extract features such as color (using OpenCV) or texture descriptors.
    def extract_image_features(self, image):
        return
    
    # Resize images for uniformity
    def resize_image(self, img_name):
        return






    def download_image(self, base_url, img_src, img_name):
        try:
            full_url = requests.compat.urljoin(base_url, img_src)
            response = requests.get(full_url, stream=True)
            if response.status_code == 200:
                img_path = os.path.join(self.image_dir, img_name)
                with open(img_path, "wb") as file:
                    file.write(response.content)
                print(f"Image saved: {img_path}")
                return img_path
            else:
                print(f"Failed to download image: {full_url}")
                return None
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def fetch_img_desc(self, website):
        try:
            self.driver.get(website)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "img")))
            time.sleep(random.uniform(2, 5))  # Mimic human browsing

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            images = soup.find_all("img")
            descriptions = []

            for img in images:
                img_src = img.get("src")
                img_alt = img.get("alt", "").strip() or "No description available"
                img_name = f"product_{self.count}.jpg"

                if img_src:
                    img_path = self.download_image(website, img_src, img_name)
                    if img_path:
                        descriptions.append({"image_file": img_path, "description": img_alt})
                    self.count += 1
            return descriptions
        except Exception as e:
            print(f"Error scraping {website}: {e}")
            return []
    
    def save_to_json(self, data, index):
        file_path = os.path.join(self.output_dir, f"{index}-imagerecognition.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        print(f"Saved: {file_path}")
    
    def scrape_all(self):
        for index, website in enumerate(self.websites):
            print(f"Scraping website: {website}")
            descriptions = self.fetch_img_desc(website)
            if descriptions:
                self.save_to_json(descriptions, index)
    
    def close(self):
        self.driver.quit()
    
if __name__ == "__main__":
    websites = [
            "https://www.zara.com/us/",
            "https://www.asos.com/us/women/fashion-feed/?ctaref=ww|fashionandbeauty",
            "https://www.aritzia.com/us/en/favourites-1",
    	    "https://www.aritzia.com/us/en/new",
            "https://www.aritzia.com/en/clothing",
            "https://www.glamour.com/fashion",
            "https://www.cosmopolitan.com/style-beauty/fashion/",
            "https://www.elle.com/fashion/",
            "https://www2.hm.com/en_us/women/seasonal-trending/trending-now.html",
    	    "https://www2.hm.com/en_us/women/deals/bestsellers.html",
            "https://www.modaoperandi.com/editorial/what-we-are-wearing",
            "https://www.abercrombie.com/shop/us/womens-new-arrivals",
            "https://shop.mango.com/us/women/featured/whats-new_d55927954?utm_source=c-producto-destacados&utm_medium=email&utm_content=woman&utm_campaign=E_WSWEOP24&sfmc_id=339434986&cjext=768854443022715810",
            "https://www.whowhatwear.com/section/fashion",
            "https://www.whowhatwear.com/fashion/trends",
            "https://www.shopcider.com/collection/new?listSource=homepage%3Bcollection_new%3B1",
            "https://www.shopcider.com/product/list?collection_id=94&link_url=https%3A%2F%2Fwww.shopcider.com%2Fproduct%2Flist%3Fcollection_id%3D94&operationpage_title=homepage&operation_position=2&operation_type=category&operation_content=Bestsellers&operation_image=&operation_update_time=1712742203550&listSource=homepage%3Bcollection_94%3B2",
            "https://www.prettylittlething.us/new-in-us.html",
            "https://www.prettylittlething.us/shop-by/trends.html",
            "https://us.princesspolly.com/collections/new",
            "https://us.princesspolly.com/collections/best-sellers",
            "https://www.aloyoga.com/collections/new-arrivals",
            "https://www.pullandbear.com/us/woman/new-arrivals-n6491"
    ]
    scraper = scrapeSites(websites)
    scraper.scrape_all()
    scraper.close()

# Preprocess data
# Resize images for uniformity, then extract features or texture descriptors
# Create labeled datasets with classes for different features


# Model training
# Use TensorFlow and a pretrained CNN
# Finetune model based on dataset outputs (desired features)
# Train separate models for specific tasks such as color extraction, occasion, and weather.

# Deployment
# Saved the trained me and deploy on a cloud service for real-time API calls that users can make on the web app.