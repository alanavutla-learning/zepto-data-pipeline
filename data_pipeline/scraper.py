import requests

url = "https://www.zepto.com/"

response = requests.get(url)

print("Status code:", response.status_code)
print("Page length:", len(response.text))