import requests
import os

# Try to detect proxy from environment or system
proxies = {
    'http': os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'),
    'https': os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy'),
}

print("Environment proxy settings:")
print(f"  HTTP_PROXY: {proxies['http']}")
print(f"  HTTPS_PROXY: {proxies['https']}")

# Try common Opera VPN proxy ports
opera_proxies = [
    'http://127.0.0.1:1080',
    'http://127.0.0.1:8080', 
    'http://127.0.0.1:3128',
    'socks5://127.0.0.1:1080',
]

url = "https://www.pro-football-reference.com/players/A/AlleJo02.htm"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

print("\n\nTrying with environment proxy settings...")
session = requests.Session()
try:
    r = session.get(url, headers=headers, timeout=10, proxies=proxies if proxies['https'] else None)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {str(e)[:80]}")

print("\n\nTrying Opera VPN common proxy ports...")
for proxy in opera_proxies:
    print(f"\nTrying {proxy}...")
    try:
        session = requests.Session()
        proxy_dict = {'http': proxy, 'https': proxy}
        r = session.get(url, headers=headers, timeout=5, proxies=proxy_dict)
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print("  SUCCESS!")
            break
    except Exception as e:
        print(f"  Error: {str(e)[:60]}")
