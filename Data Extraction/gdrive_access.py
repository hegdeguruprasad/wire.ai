from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload
import pickle
import os

SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_drive():
    """Authenticates and returns the Google Drive service instance."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('gdrive_client_credentials.json', SCOPES)
        creds = flow.run_local_server(port=8080)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)

drive_service = authenticate_drive()

def list_drive_files():
    """Lists the first 10 files in Google Drive."""
    results = drive_service.files().list(pageSize=10, fields="files(id, name)").execute()
    files = results.get('files', [])
    if not files:
        print("No files found.")
    else:
        print("Files:")
        for file in files:
            print(f"{file['name']} ({file['id']})")

def upload_file(file_path, file_name, folder_id=None):
    """Uploads a file to Google Drive."""
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_name} with ID: {file.get('id')}")
    return file.get('id')

def download_file(file_id, output_path):
    """Downloads a file from Google Drive."""
    request = drive_service.files().get_media(fileId=file_id)
    with open(output_path, 'wb') as file:
        file.write(request.execute())
    print(f"Downloaded file to {output_path}")

def create_folder(folder_name):
    """Creates a new folder in Google Drive."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = drive_service.files().create(body=file_metadata, fields='id').execute()
    print(f"Folder '{folder_name}' created with ID: {folder.get('id')}")
    return folder.get('id')

def move_file(file_id, folder_id):
    """Moves a file to a different folder in Google Drive."""
    file = drive_service.files().get(fileId=file_id, fields='parents').execute()
    previous_parents = ",".join(file.get('parents'))
    
    drive_service.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=previous_parents,
        fields='id, parents'
    ).execute()
    
    print(f"File {file_id} moved to folder {folder_id}")

def delete_file(file_id):
    """Deletes a file from Google Drive."""
    drive_service.files().delete(fileId=file_id).execute()
    print(f"File {file_id} deleted.")

def search_files(query, max_results=10):
    """Searches for files in Google Drive based on a query."""
    results = drive_service.files().list(
        q=query, pageSize=max_results, fields="files(id, name)"
    ).execute()
    
    files = results.get('files', [])
    if not files:
        print("No matching files found.")
    else:
        for file in files:
            print(f"{file['name']} ({file['id']})")
    return files

def share_file(file_id, email=None, role='reader'):
    """Shares a file publicly or with a specific user."""
    permission = {'type': 'anyone' if not email else 'user', 'role': role}
    
    if email:
        permission['emailAddress'] = email
    
    drive_service.permissions().create(fileId=file_id, body=permission).execute()
    print(f"Shared file {file_id} with {'anyone' if not email else email}")

if __name__ == "__main__":
    # Example Usage (Uncomment to test each function)
    list_drive_files()  # List files
    # upload_file("D:/example.pdf", "Uploaded File.pdf")  # Upload file
    # download_file("your-file-id", "D:/downloaded.pdf")  # Download file
    # folder_id = create_folder("My New Folder")  # Create a folder
    # move_file("your-file-id", "your-folder-id")  # Move file to folder
    # delete_file("your-file-id")  # Delete a file
    # search_files("mimeType='application/pdf'")  # Search for PDFs
    # share_file("your-file-id", "user@example.com", "writer")  # Share file with user
