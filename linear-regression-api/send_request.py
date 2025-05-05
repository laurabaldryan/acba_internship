import requests

print("Reading file...")
with open("diamonds.csv", "r", encoding="utf-8") as data:
    csv_content = data.read()

print("Sending POST request to http://127.0.0.1:5000/predict...")
try:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"csv": csv_content},
        timeout=10
    )
    print("Response received.")
    print(response.json())
except requests.exceptions.Timeout:
    print("Request timed out.")
except requests.exceptions.ConnectionError:
    print("Connection error. Is Flask running?")
except requests.exceptions.JSONDecodeError:
    print("Response not in JSON format.")
    print("Raw response:", response.text)
