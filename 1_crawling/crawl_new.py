import pandas as pd
import re
import time
import random
import argparse
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# pip install webdriver-manager selenium-stealth
from webdriver_manager.chrome import ChromeDriverManager
from selenium_stealth import stealth
from tqdm import tqdm
import platform

def create_driver(proxy: str = None):
    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("window-size=1920,1080")

    # 운영체제에 따라 Chrome 바이너리 경로 설정
    system_platform = platform.system()
    if system_platform == "Darwin":  # macOS
        chrome_options.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    elif system_platform == "Windows":
        chrome_options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        if not os.path.exists(chrome_options.binary_location):  # 혹시 다른 경로일 경우
            chrome_options.binary_location = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"

    # User-Agent
    ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
          "AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/137.0.0.0 Safari/537.36")
    chrome_options.add_argument(f"user-agent={ua}")

    if proxy:
        chrome_options.add_argument(f"--proxy-server={proxy}")

    # webdriver-manager로 자동 설치된 드라이버 사용
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # CDP 레벨 지문 은폐
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
      "source": """
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});
      """
    })
    driver.execute_cdp_cmd("Network.enable", {})
    driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
        "headers": {"accept-language": "en-US,en;q=0.9"}
    })

    # selenium-stealth 추가 은폐
    stealth(driver,
            languages=["en-US", "en"],
            vendor="Google Inc.",
            platform="Win32",
            webgl_vendor="Intel Inc.",
            renderer="Intel Iris OpenGL Engine",
            fix_hairline=True)

    return driver

def crawl_chunk(chunk_id: str, proxy: str = None):
    asin_df = pd.read_csv(f"asin/crawl_again_ver4_{chunk_id}.csv", header=None)
    asin_list = asin_df.iloc[:, 0].tolist()

    result_file = f"output/output-ver4/output_crawl_ver4_{chunk_id}.csv"
    if os.path.exists(result_file):
        existing = pd.read_csv(result_file)
        processed_asins = set(existing['ASIN'])
        data = existing.to_dict("records")
        print(f"기존 저장: {len(processed_asins)}개 이미 처리")
    else:
        processed_asins = set()
        data = []
        print("저장된 결과 없음")

    print(f"전체 ASIN 개수: {len(asin_list)}")
    print(f"남은 개수: {len(asin_list) - len(processed_asins)}")

    driver = create_driver(proxy)
    wait = WebDriverWait(driver, 0.5, poll_frequency=0.1)
    processed = len(processed_asins)

    for asin in tqdm(asin_list, desc="진행률", ncols=100):
        if asin in processed_asins:
            continue

        driver.get(f"https://www.amazon.com/dp/{asin}?language=en_US")
        time.sleep(random.uniform(0.5, 1.0))

        # 1) Title
        try:
            title = wait.until(EC.presence_of_element_located((By.ID, "productTitle"))).text.strip()
        except:
            title = "N/A"

        # 2) Rating
        try:
            rating = driver.find_element(By.ID, "acrPopover").get_attribute("title").split()[0]
        except:
            try:
                rating = driver.find_element(By.CLASS_NAME, "a-icon-alt").text.strip().split()[0]
            except:
                rating = "N/A"

        # 3) Rating Count
        try:
            rc = driver.find_element(By.ID, "acrCustomerReviewText").text.strip()
            rating_count = re.sub(r"[^\d]", "", rc)
        except:
            rating_count = "N/A"

        # 4) Bought Last Month
        try:
            bought = driver.find_element(By.ID, "social-proofing-faceout-title-tk_bought").text.strip()
            bought_last_month = re.sub(r"\+", "", bought.split()[0])
        except:
            bought_last_month = "N/A"

        # 5) Categories
        try:
            elems = driver.find_elements(By.CSS_SELECTOR, "a.a-link-normal.a-color-tertiary")
            cats = [e.text.strip() for e in elems[:8]]
            while len(cats) < 8:
                cats.append("N/A")
        except:
            cats = ["N/A"] * 8

        data.append({
            'ASIN': asin,
            'Title': title,
            'Rating': rating,
            'Rating Count': rating_count,
            'Bought Last Month': bought_last_month,
            **{f"Category {i+1}": cats[i] for i in range(8)}
        })

        processed += 1
        if processed % 25 == 0:
            pd.DataFrame(data).to_csv(result_file, index=False)

    pd.DataFrame(data).to_csv(result_file, index=False)
    driver.quit()
    print("크롤링 완료, 결과 저장됨:", result_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=str, required=True, help="크롤링할 ASIN 청크 ID")
    parser.add_argument("--proxy", type=str, default=None, help="사용할 프록시 (예: http://ip:port)")
    args = parser.parse_args()
    crawl_chunk(args.chunk_id, args.proxy)
