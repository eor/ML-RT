'''
clients_secrets is a file specific to a project. If you are part of the project 'ML_RT' and want to use this file
for uploading data, then you can get this project-specific clients_secrets.json from @aayushsingla or you can just go ahead
and create your own clients_secrets.json file as described below.

If you are not a part of project 'ML_RT' and want to use this script for own purposes, then follow the instructions below:
1. Obtain client_secrets-xxxxxxxxxxxxxxxx.json from google console following all the instructions in this doc
https://d35mpxyw7m7k7g.cloudfront.net/bigdata_1/Get+Authentication+for+Google+Service+API+.pdf . Just download 
the .json file and don't do anything else.
2. Rename the client_secrets-xxxxxxxxxxxxxxxx.json to client_secrets.json and place it in the same folder as this file.
(Remember, this file is important and can be used by anyone to access and modify your drive files. 
Don't commit it or post it publically in any  condition.)

*********************************************************************************************************
usage: 
    
    python3 gdrive_upload.py -d dest_folder -s src_path1 src_path2 ..... src_pathN
    
    where,
    dest_folder: folder name inside the root directory in the google drive where you want to save your data.
                 (Note: max. depth of path1 allowed from root directory is 1 and you need to specify
                 just the folder name. for ex: 'ML-RT' will work where as, './ML-RT' or 'ML-RT/results' won't)
    
    src_pathX:   relative paths to source directory which you want to upload on drive. Folders will
                 be created with the same name.
**********************************************************************************************************

Note: if you face a RedirectMissingLocation Error while upoading some files, this is a issue with
httplib2 version that breaks pydrive. try using `pip install httplib2==0.15.0` to switch it to a stable version.
'''

# Import Google libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
import googleapiclient.errors

# Import general libraries
from argparse import ArgumentParser
from os import chdir, listdir, stat
from sys import exit
import ast
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import pathlib


def parse_args():
    """ 
        Parse arguments
    """

    parser = ArgumentParser(
        description="Upload local folder to Google Drive")
    parser.add_argument('-s', '--source', nargs="+", default=[],
                        help='list of folders to upload')
    parser.add_argument('-d', '--destination', type=str, 
                        help='destination folder in Google Drive')

    return parser.parse_args()

def authenticate():
    """ 
        Authenticate to Google API
    """
    
    print('\n1. Please open the below link and sign in to your account.')
    print('2. Once you sign in, you will be displayed a verification code. Copy that code into the command line and press enter. \n')

    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)
    
    if drive is None:
        print('Authentication Failure')
        os._exit(0)
    
    return drive

def get_folder_id(drive, parent_folder_id, folder_name):
    """ 
    Check if destination folder exists and return it's ID
    :param drive: An instance of GoogleAuth
    :param parent_folder_id: the id of the parent of the folder we are uploading the files to
    :param folder_name: the name of the folder in the drive 
    """

    # Auto-iterate through all files in the parent folder.
    file_list = GoogleDriveFileList()

    try:
        file_list = drive.ListFile(
            {'q': "'{0}' in parents and trashed=false".format(parent_folder_id)}
                                  ).GetList()
    # Exit if the parent folder doesn't exist
    except googleapiclient.errors.HttpError as err:
        # Parse error message
        message = ast.literal_eval(err.content)['error']['message']
        if message.contains('File not found'):
            print(message + folder_name)
            os._exit(1)
        # Exit with stacktrace in case of other error
        else:
            raise

    # Find the the destination folder in the parent folder's files
    for file in file_list:
        if file['title'] == folder_name:
            print('title: %s, id: %s' % (file['title'], file['id']))
            return file['id']

        
def create_folder(drive, folder_name, parent_folder_id):
    """ 
    Create folder on Google Drive
    :param drive: An instance of GoogleAuth
    :param folder_id: the id of the folder we are uploading the files to
    :param parent_folder_id: the id of the parent of the folder we are uploading the files to
    """
    
    folder_metadata = {
        'title': folder_name,
        # Define the file type as folder
        'mimeType': 'application/vnd.google-apps.folder',
        # ID of the parent folder        
        'parents': [{"kind": "drive#fileLink", "id": parent_folder_id}]
    }

    folder = drive.CreateFile(folder_metadata)
    folder.Upload()

    # Return folder informations
    print('title: %s, id: %s' % (folder['title'], folder['id']))
    return folder['id']
          


def upload_files_in_folder(drive, folder_id, src_folder_name):
    """ 
    Recursively upload files from a folder and its subfolders to Google Drive 
    :param drive: An instance of GoogleAuth
    :param folder_id: the id of the folder we are uploading the files to
    :param src_folder_name: the path to the source folder to upload
    """
    print("\n Folder:", src_folder_name)
    
    # Iterate through all files in the folder.
    for object_name in os.listdir(src_folder_name):
        filepath = os.path.join(src_folder_name, object_name)
        
        # Check the file's size and skip the file if it's empty
        statinfo = os.stat(filepath)
        if statinfo.st_size > 0:
            # Upload file to folder.
            if os.path.isdir(filepath):

                child_folder_id = get_folder_id(drive, folder_id, object_name)
                
                # Create the folder if it doesn't exists
                if not child_folder_id:
                    child_folder_id = create_folder(drive, object_name, folder_id)
                else:
                    print('folder {0} already exists'.format(object_name))
                    
                upload_files_in_folder(drive, child_folder_id, filepath)
            else: 
                print('Uploading file: ', object_name)
                f = drive.CreateFile({"title":object_name, "parents": [{"kind": "drive#fileLink", "id": folder_id}]})
                f.SetContentFile(filepath)
                f.Upload()
                print('Uploaded: title: %s,\n          id: %s\n' % (f['title'], f['id']))
                f = None
        else:
            print('file {0} is empty'.format(file))

def main():
    """ 
        Main
    """
    args = parse_args()    
    
    src_folder_names = args.source
    parent_folder_name = args.destination
    
    # you can also hard code folders here for ease ;p
    # must be commented to use command line for specifying paths    
    src_folder_names = [
#     '../test/paper/run_MLP1_DTW_17H',
#     '../test/paper/run_MLP1_MSE_12H',
#     '../test/paper/run_CVAE1_DTW_32H',
#     '../test/paper/run_CVAE1_MSE_42H',
#     '../test/paper/run_LSTM1_MSE_29H',
#     '../test/paper/run_LSTM1_DTW_40H',
#     '../test/paper/run_MLP2_DTW_23T',
#     '../test/paper/run_MLP2_MSE_25T',
#     '../test/paper/run_CVAE1_MSE_36T',
#     '../test/paper/run_CVAE1_DTW_39T',
#     '../test/paper/run_LSTM1_DTW_3T',
#     '../test/paper/run_LSTM1_MSE_4T',
    '../test/paper/run_CLSTM1_MSE_13C',
    '../test/paper/run_CLSTM1_DTW_45C'        
    ]
    parent_folder_name = 'ML_RT_results'
    
    if len(src_folder_names) <= 0:
      print('\nNo files to upload.\n')
      os._exit(0)

    if parent_folder_name is None:
      print('\nPlease specify the destination in google drive.\n')
      os._exit(0)
    
    print('\nAll these folder will be uploaded to google drive:\n', src_folder_names)
    
    
    # Authenticate to Google API
    drive = authenticate()
    
    # Get parent folder ID
    parent_folder_id = get_folder_id(drive, 'root', parent_folder_name)

    for src_path in src_folder_names:
        if os.path.isdir(src_path):
            dst_folder_name = os.path.basename(src_path)
        else:
            dst_folder_name = os.path.basename(os.path.dirname(src_path))

        # Get destination folder ID
        folder_id = get_folder_id(drive, parent_folder_id, dst_folder_name)
        
        # Create the folder if it doesn't exists
        if not folder_id:
            print('creating folder ' + dst_folder_name)
            folder_id = create_folder(drive, dst_folder_name, parent_folder_id)
        else:
            print('folder {0} already exists'.format(dst_folder_name))
            
        # Upload the files    
        upload_files_in_folder(drive, folder_id, src_path)


if __name__ == "__main__":
    main()

