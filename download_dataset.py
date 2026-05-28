import os
import requests
import zipfile
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

dataset_dir = "./dataset"
os.makedirs(dataset_dir, exist_ok=True)


def download_dataset(dataset_name, file_url, dataset_dir):
    print(f"\nDownloading {dataset_name}...")
    print(f"URL: {file_url}")

    zip_path = os.path.join(dataset_dir, f"{dataset_name.lower()}.zip")

    r = requests.get(file_url, stream=True, timeout=600, verify=False)

    
    if r.status_code != 200:
        print(f"Failed: HTTP {r.status_code}")
        return

    with open(zip_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded: {os.path.getsize(zip_path) / 1024 / 1024:.1f} MB")

    # Extract to dataset_dir/dataset_name/
    extract_dir = os.path.join(dataset_dir, dataset_name.lower())
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    os.remove(zip_path)

    print(f"Extracted to: {extract_dir}/")
    for f in sorted(os.listdir(extract_dir))[:20]:
        print(f"  {f}")
    print("  ...")


download_dataset(
    dataset_name="HAR",
    file_url="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    dataset_dir=dataset_dir
)

download_dataset(
    dataset_name="WISDM",
    file_url="https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip",
    dataset_dir=dataset_dir
)

print("\nAll done!")
