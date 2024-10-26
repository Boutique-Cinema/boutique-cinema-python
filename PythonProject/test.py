import requests

url = "http://localhost:8000/detect"
message = "Test message"
file_path = "sample.jpg"

with open(file_path, "rb") as file:
    response = requests.post(url, data={"message": message}, files={"file": file})

print(response.json())
