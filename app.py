"""
scrape_propertypro_full.py

Selenium-based scraper for PropertyPro.ng to scrape listings across all Nigerian states.
Outputs: real_estate_nigeria_full.csv

Notes:
- Requires Chrome + chromedriver (matching versions) OR Chromium + chromedriver.
- If you hit any selector mismatch (site layout updates), adjust the CSS selectors
  around the 'parse_listing_card' function (marked with comments).
"""

import time
import re
import csv
import os
import math
from datetime import datetime
from typing import Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as dateparser

# ========== CONFIG ==========
OUTPUT_CSV = "real_estate_nigeria_full.csv"
HEADLESS = True
START_PAGE = 1
MAX_PAGES_PER_STATE = 200        # safety limit per state (adjust)
DELAY_BETWEEN_REQUESTS = 1.5    # seconds
STATES = [
    "Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue","Borno","Cross River","Delta",
    "Ebonyi","Edo","Ekiti","Enugu","Gombe","Imo","Jigawa","Kaduna","Kano","Katsina","Kebbi",
    "Kogi","Kwara","Lagos","Nasarawa","Niger","Ogun","Ondo","Osun","Oyo","Plateau","Rivers",
    "Sokoto","Taraba","Yobe","Zamfara","Federal Capital Territory"
]
BASE_SEARCH_URL = "https://www.propertypro.ng/property-for-sale"  # we append query params or navigate and filter
# ============================

def make_driver():
    options = Options()
    if HEADLESS:
        # new headless flag sometimes better, but keep flexible
        try:
            options.add_argument("--headless=new")
        except Exception:
            options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    # Use unique user-data-dir to avoid SessionNotCreatedException in devcontainers
    options.add_argument(f"--user-data-dir=/tmp/chrome_dev_{time.time()}")
    options.add_argument("--disable-extensions")
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(30)
    return driver

def normalize_price(raw_price: Optional[str]):
    """
    Extract numeric NGN price and period (per year, per sqm, per month, etc.)
    Returns (int_price_or_None, period_or_None)
    Example raw_price: "₦ 2,500,000" or "₦4,500,000 / year" or "From ₦10,000 / month"
    """
    if not raw_price:
        return None, None
    rp = raw_price.strip()
    # unify common separators
    rp = rp.replace("\u00A0", " ").replace(",", "").replace(" ", " ")
    # extract period words like "per year", "/ year", "/month", "per sqm"
    period = None
    m_period = re.search(r"(per|/)\s*(year|month|sqm|m2|annum|yr|monthly|yearly|sq ?m|sqm)", rp, re.I)
    if m_period:
        period = m_period.group(2).lower()
    # extract first big number
    digits = re.findall(r"(\d{4,})", rp.replace(",", ""))
    if digits:
        # take the first large number as price (NGN)
        val = int(digits[0])
        # Heuristics: if extracted number seems small (<1000) maybe it's monthly in hundreds; keep as-is.
        return val, period
    # fallback: try any number
    digits_any = re.findall(r"(\d+)", rp)
    if digits_any:
        return int(digits_any[0]), period
    return None, period

def extract_number_from_text(text: Optional[str]):
    if not text:
        return None
    t = re.sub(r"[^\d\.]", "", text)
    if t == "":
        return None
    if "." in t:
        try:
            return float(t)
        except:
            return None
    else:
        try:
            return int(t)
        except:
            return None

def parse_listing_card(card_html: str, base_url="https://www.propertypro.ng"):
    """
    card_html: HTML of the listing card (string).
    Returns a dict of extracted fields.
    NOTE: If site structure changes, adjust the selectors below.
    """
    soup = BeautifulSoup(card_html, "html.parser")
    # Adaptable selectors — check card structure and tweak if needed:
    # Many listing cards have a header/title in <h4> or <h3>, price in <h3> or span.price, and address in <address>.
    title = None
    if soup.find("h4"):
        title = soup.find("h4").get_text(strip=True)
    elif soup.find("h3"):
        title = soup.find("h3").get_text(strip=True)
    elif soup.find("a", {"class": "property-title"}):
        title = soup.find("a", {"class": "property-title"}).get_text(strip=True)

    # price raw
    price_raw = None
    # try few places
    price_tag = soup.find("h3")
    if price_tag:
        price_raw = price_tag.get_text(strip=True)
    else:
        p_tag = soup.find("span", class_=re.compile(r"price", re.I))
        if p_tag:
            price_raw = p_tag.get_text(strip=True)

    price_ngn, price_period = normalize_price(price_raw)

    # location
    location_raw = None
    if soup.find("address"):
        location_raw = soup.find("address").get_text(strip=True)
    else:
        # sometimes in small tags
        ltag = soup.find("div", class_=re.compile(r"location", re.I))
        if ltag:
            location_raw = ltag.get_text(strip=True)

    city = None
    state = None
    if location_raw and "," in location_raw:
        parts = [p.strip() for p in location_raw.split(",")]
        # heuristics: first part city, last part state
        if len(parts) >= 2:
            city = parts[0]
            state = parts[-1]

    # features (beds/baths/toilets/size)
    beds = baths = toilets = None
    size_sqm = None
    # Typical features are in <li> or spans — search for keywords
    for li in soup.find_all(["li","span","p","div"]):
        txt = li.get_text(" ", strip=True).lower()
        if "bed" in txt and beds is None:
            beds = extract_number_from_text(txt)
        if "bath" in txt and baths is None:
            baths = extract_number_from_text(txt)
        if "toilet" in txt and toilets is None:
            toilets = extract_number_from_text(txt)
        if ("sqm" in txt or "sq m" in txt or "sq.m" in txt or "size" in txt) and size_sqm is None:
            size_sqm = extract_number_from_text(txt)

    # link to listing
    link = None
    a = soup.find("a", href=True)
    if a:
        href = a["href"]
        if href.startswith("http"):
            link = href
        else:
            link = base_url.rstrip("/") + href

    # attempt to get listing date/time if present
    listing_date = None
    # some cards show "Added X days ago" or a date label - attempt to find it
    for dt in soup.find_all(["time","span","small","p","div"]):
        t = dt.get_text(" ", strip=True)
        if re.search(r"\b(ago|added|posted|days|day|month|year)\b", t, re.I) or re.search(r"\d{4}-\d{2}-\d{2}", t):
            # try parse
            try:
                parsed = dateparser.parse(t, fuzzy=True)
                if parsed:
                    listing_date = parsed.date().isoformat()
                    break
            except:
                continue

    listing_year = None
    if listing_date:
        try:
            listing_year = int(listing_date.split("-")[0])
        except:
            listing_year = None

    # attempt to guess property type from title or tags
    property_type = None
    if title:
        tt = title.lower()
        for p in ["duplex","semi-duplex","semi duplex","detached","terraced","flat","apartment","studio","bungalow","mansion","duplex"]:
            if p in tt:
                property_type = p.replace(" ", "-")
                break

    # fallback: find an explicit tag
    tag = soup.find("span", class_=re.compile(r"(type|property-type)", re.I))
    if tag and not property_type:
        property_type = tag.get_text(strip=True)

    return {
        "title": title,
        "property_type": property_type,
        "price_raw": price_raw,
        "price_ngn": price_ngn,
        "price_period": price_period,
        "location_raw": location_raw,
        "city": city,
        "state": state,
        "bedrooms": beds,
        "bathrooms": baths,
        "toilets": toilets,
        "size_sqm": size_sqm,
        "listing_date": listing_date,
        "listing_year": listing_year,
        "link": link
    }

def write_header_if_needed(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "title","property_type","price_raw","price_ngn","price_period",
                "location_raw","city","state","bedrooms","bathrooms","toilets",
                "size_sqm","listing_date","listing_year","link","scrape_state","scrape_date"
            ])

def already_scraped(link, path):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, usecols=["link"])
        return link in df["link"].astype(str).tolist()
    except Exception:
        return False

def run_scrape():
    driver = make_driver()
    write_header_if_needed(OUTPUT_CSV)
    total = 0

    try:
        for state in STATES:
            print(f"\n=== Scraping state: {state} ===")
            # We will attempt to call the search URL with the state slug (very often site supports "in-StateName")
            # Construct a state-friendly slug
            state_slug = state.lower().replace(" ", "-").replace("federal-capital-territory","federal-capital-territory")
            # common patterns:
            search_url = f"{BASE_SEARCH_URL}-in-{state_slug}"
            # fallback: basic listing page with query param
            alt_url = f"{BASE_SEARCH_URL}?state={state}"
            # try direct navigation to either URL
            urls_to_try = [search_url, alt_url, BASE_SEARCH_URL]

            page_num = START_PAGE
            pages_without_results = 0
            while page_num <= MAX_PAGES_PER_STATE:
                url_page = f"{urls_to_try[0]}?page={page_num}" if "?" in urls_to_try[0] else f"{urls_to_try[0]}?page={page_num}"
                try:
                    print(f"Loading: {url_page}")
                    driver.get(url_page)
                except Exception as e:
                    # fallback: try alt
                    try:
                        alt_page = f"{alt_url}?page={page_num}"
                        print(f"Primary failed, trying alt: {alt_page}")
                        driver.get(alt_page)
                        url_page = alt_page
                    except Exception as e2:
                        print("Both page loads failed for this page; breaking out of this state's pagination.")
                        break

                # wait for results container (adjust selector if site differs)
                try:
                    WebDriverWait(driver, 12).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div"))
                    )
                except TimeoutException:
                    print("Timed out waiting for page content; continuing.")
                time.sleep(1.0)

                source = driver.page_source
                soup = BeautifulSoup(source, "html.parser")
                # Update this selector if the site uses another container class (e.g., 'single-room-sale' or 'property-list-card')
                cards = soup.find_all("div", class_=re.compile(r"(single-room-sale|property-list-card|listing|card)", re.I))
                if not cards:
                    # try more aggressive selection
                    cards = soup.select("article, li.listing, div.listing-card, div[class*='property']")

                if not cards:
                    print(f"No listing cards found on page {page_num} for state {state}.")
                    pages_without_results += 1
                    if pages_without_results >= 2:
                        # likely no more pages for this state
                        break
                    page_num += 1
                    continue

                pages_without_results = 0
                scraped_this_page = 0
                for c in cards:
                    item = parse_listing_card(str(c))
                    if not item["link"]:
                        continue
                    # Avoid duplicates
                    if already_scraped(item["link"], OUTPUT_CSV):
                        continue
                    # Basic year filter: if listing_year exists and we only want 2024/2025, we could skip others.
                    # The user asked to focus on 2024/2025 if available — we will keep everything but mark listing_year.
                    item["scrape_state"] = state
                    item["scrape_date"] = datetime.utcnow().isoformat()
                    # Write to CSV
                    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            item.get("title"),
                            item.get("property_type"),
                            item.get("price_raw"),
                            item.get("price_ngn"),
                            item.get("price_period"),
                            item.get("location_raw"),
                            item.get("city"),
                            item.get("state"),
                            item.get("bedrooms"),
                            item.get("bathrooms"),
                            item.get("toilets"),
                            item.get("size_sqm"),
                            item.get("listing_date"),
                            item.get("listing_year"),
                            item.get("link"),
                            item.get("scrape_state"),
                            item.get("scrape_date"),
                        ])
                    scraped_this_page += 1
                    total += 1

                print(f"Page {page_num}: scraped {scraped_this_page} listings (total so far {total}).")
                page_num += 1
                time.sleep(DELAY_BETWEEN_REQUESTS)

            # small delay between states
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("User interrupted. Exiting gracefully.")
    except WebDriverException as e:
        print("WebDriver error:", e)
    finally:
        driver.quit()
        print(f"Scraping finished. Total listings scraped: {total}")
        if os.path.exists(OUTPUT_CSV):
            df = pd.read_csv(OUTPUT_CSV)
            print(f"Saved CSV rows: {len(df)}")

if __name__ == "__main__":
    run_scrape()
