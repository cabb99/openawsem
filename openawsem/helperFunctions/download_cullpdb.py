import os
import requests
from bs4 import BeautifulSoup

# URL of the directory containing the files
base_url = "http://dunbrack.fccc.edu/pisces/download/"

# File name pattern to match
file_pattern = "cullpdb_pc80.0_res0.0-3.0_len40-10000_R0.3_Xray_d2023_10_05_chains41105"

# Directory where the downloaded files will be saved
output_dir = "downloaded_files"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Send a GET request to the base URL and parse the HTML content
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find and download files that match the pattern
for link in soup.find_all('a'):
    file_name = link.get('href')
    if file_name and file_name.startswith(file_pattern):
        file_url = base_url + file_name
        output_file_path = os.path.join(output_dir, file_name)

        # Download the file
        response = requests.get(file_url, stream=True)
        with open(output_file_path, 'wb') as output_file:
            for chunk in response.iter_content(chunk_size=8192):
                output_file.write(chunk)

        print(f"Downloaded: {file_name}")

print("Download complete.")