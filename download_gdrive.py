import os
import sys

try:
    import requests
except ImportError:
    requests = None


def download_from_google_drive(file_id: str, dest_path: str):
    """Download a file from Google Drive by file ID."""
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    if requests is None:
        # fallback to urllib
        import urllib.request
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        urllib.request.urlretrieve(url, dest_path)
        return

    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}

    response = session.get(url, params=params, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
            break

    if token:
        params["confirm"] = token
        response = session.get(url, params=params, stream=True)

    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


if __name__ == "__main__":
    files = [
        ("1IAfLZwu6rJzyWKgBToqwSmmVYU6VbjVs", "data/hyperliquid_trades.csv"),
        ("1PgQC0tO8XN-wqkNyghWc_-mnrYv_nhSf", "data/fear_greed_index.csv"),
    ]

    for file_id, dest in files:
        print(f"Downloading {file_id} -> {dest}")
        try:
            download_from_google_drive(file_id, dest)
            print("  done")
        except Exception as e:
            print("  failed:", e)
            sys.exit(1)
