import requests
import cloudscraper

url = "https://www.pro-football-reference.com/players/A/AlleJo02.htm"

print("Testing requests library...")
try:
    r = requests.get(url, timeout=10)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting cloudscraper...")
try:
    cs = cloudscraper.create_scraper()
    r = cs.get(url, timeout=10)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
