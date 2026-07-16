import os
import subprocess



class GcloudSync:

    def sync_folder_to_gcloud(self, gcp_bucket_url, filepath, filename):
        command = ["gsutil", "cp", os.path.join(filepath, filename), f"gs://{gcp_bucket_url}"]
        subprocess.run(command, check=True)

    def sync_folder_from_gcloud(self, gcp_bucket_url, filename, destination):
        command = ["gsutil", "cp", f"gs://{gcp_bucket_url}/{filename}", os.path.join(destination, filename)]
        subprocess.run(command, check=True)