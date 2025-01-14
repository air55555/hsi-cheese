import os
from git import Repo


def fetch_file_from_git(repo_url, file_path, local_dir, save_as=None):
    """
    Fetches a file from a remote Git repository and saves it locally.

    Parameters:
        repo_url (str): The URL of the remote Git repository.
        file_path (str): The relative path of the file in the repository.
        local_dir (str): The local directory to clone/pull the repository.
        save_as (str): Optional. The path to save the file locally. If not provided, uses the same name.

    Returns:
        str: The path of the saved file.
    """
    # Clone the repo if it doesn't exist locally
    if not os.path.exists(local_dir):
        print(f"Cloning repository: {repo_url}")
        repo = Repo.clone_from(repo_url, local_dir, branch = 'main')
    else:
        print(f"Pulling latest changes for repository: {repo_url}")
        repo = Repo(local_dir)
        repo.remotes.origin.pull()

    # Ensure the file exists in the repo
    full_file_path = os.path.join(local_dir, file_path)
    if not os.path.exists(full_file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist in the repository.")

    # Define the save path
    save_path = save_as or os.path.basename(file_path)

    # Copy the file to the desired location
    with open(full_file_path, 'rb') as src_file, open(save_path, 'wb') as dest_file:
        dest_file.write(src_file.read())

    print(f"File saved to: {save_path}")
    return save_path


# Example usage
repo_url = "https://github.com/air55555/DPHSIR.git"  # Replace with the actual Git repo URL
file_path_in_repo = "utils.py"  # Replace with the relative path to the file in the repo
local_clone_dir = "./local_repo"  # Directory to clone the repository
output_file_path = "./libs/utils.py"  # Optional: path to save the file locally

fetch_file_from_git(repo_url, file_path_in_repo, local_clone_dir, output_file_path)




