import os
import configparser
import gdown

# Append the root directory to Python path
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def main():
    
    """
    Main function to download a file specified in a configuration file.
    
    Reads a configuration file to get the URL and file name for the file to be downloaded. If the file
    does not already exist locally, it downloads the file and saves it in a specified data directory.
    
    Configuration file should have the following format:
    [DOWNLOAD]
    url = <file_url>
    filename = <file_name>
    
    [DEFAULT]
    data_directory = <data_directory_path>
    """
    
    # Read configuration file
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    config.read(config_path)

    ## Retrieve values from the configuration file
    
    # For download and loading data
    url_file = config.get('DOWNLOAD', 'url')
    filename = config.get('DOWNLOAD', 'filename')
    data_directory = config.get('DEFAULT', 'data_directory')
    
    # Define the file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    file_path = os.path.join(parent_dir, data_directory, filename)
    
    # Check if the file already exists, if not, download it
    if not os.path.exists(file_path):
        download(url_file, filename)
    else:
        print(f"File {filename} already exists.")

def download(url, filename):
    """
    Download a file from a given URL and save it with the given filename.
    If the data directory does not exist, create it.
    
    :param url: The URL of the file to be downloaded.
    :param filename: The name under which to save the file locally.
    :return: None
    """
    # Check if data directory exists, if not create it
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    data_directory = os.path.join(parent_dir, 'data')
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    # Full path to the file
    file_path = os.path.join(data_directory, filename)

    # Download the file
    gdown.download(url, file_path, quiet=False, fuzzy=True)
    print(f'File successfully downloaded as {file_path}')

        
if __name__ == "__main__":
    main()