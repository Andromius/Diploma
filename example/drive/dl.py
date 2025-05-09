import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Set up authentication
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'path/to/credentials.json'  # Update this path

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build('drive', 'v3', credentials=creds)

def list_files_in_folder(folder_id):
    """List all files inside a specific folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

def download_file(file_id, file_name, save_path):
    """Download a file from Google Drive."""
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(save_path, file_name)

    with open(file_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    print(f'Downloaded: {file_name}')

def download_folder(folder_id, save_path):
    """Download an entire folder from Google Drive."""
    os.makedirs(save_path, exist_ok=True)
    files = list_files_in_folder(folder_id)
    
    for file in files:
        download_file(file['id'], file['name'], save_path)

# Replace with your folder ID and local download path
FOLDER_ID = "1EZ4hltMrqjt_eZwwlyZ6xlNIEHYhqt_p"
SAVE_PATH = "./downloaded_folder"

download_folder(FOLDER_ID, SAVE_PATH)
