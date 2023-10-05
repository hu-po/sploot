import os
import googlemaps
import requests

# Get the API key from https://console.developers.google.com/
API_KEY = os.environ["GOOGLE_SDK_API_KEY"]

# Define the parameters for the request
params = {
  "center": "37.775,-122.418",
  "zoom": 15,
  "size": "600x400",
  "sensor": "false"
}

# Make the request
request_str = f"https://maps.googleapis.com/maps/api/staticmap?{'&'.join([f'{k}={v}' for k, v in params.items()])}&key={API_KEY}"
print(request_str)
response = requests.get(request_str)


# Check the response status code
if response.status_code == 200:
  # Get the image data
  image_data = response.content

  # Save the image data to a file
  with open("image.png", "wb") as f:
    f.write(image_data)

else:
  print("Error: " + str(response.status_code))