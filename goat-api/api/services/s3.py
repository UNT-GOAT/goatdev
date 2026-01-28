"""
S3 Service

Handles all S3 operations:
- Checking if images exist
- Downloading raw images for processing
- Uploading annotated images
- Generating pre-signed URLs
"""

import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional, Dict, List
import os
import tempfile


class S3Service:
    def __init__(self, captures_bucket: str, processed_bucket: str, region: str = "us-east-2"):
        self.captures_bucket = captures_bucket
        self.processed_bucket = processed_bucket
        self.region = region
        self.s3 = boto3.client("s3", region_name=region)
    
    def check_connection(self) -> bool:
        """Check if S3 is accessible"""
        try:
            self.s3.head_bucket(Bucket=self.captures_bucket)
            return True
        except ClientError:
            return False
    
    def images_exist(self, timestamp: str) -> bool:
        """
        Check if images exist for a given timestamp.
        
        Expected structure:
            s3://goat-captures/captures/{timestamp}/
                {timestamp}_side.jpg
                {timestamp}_top.jpg
                {timestamp}_front.jpg
        """
        prefix = f"captures/{timestamp}/"
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.captures_bucket,
                Prefix=prefix,
                MaxKeys=3
            )
            
            # Check if we have at least 3 images
            if "Contents" not in response:
                return False
            
            return len(response["Contents"]) >= 3
            
        except ClientError:
            return False
    
    def download_images(self, timestamp: str, local_dir: str) -> Dict[str, str]:
        """
        Download raw images from S3 to local directory.
        
        Returns dict mapping view -> local filepath
        e.g., {"side": "/tmp/abc/side.jpg", "top": "/tmp/abc/top.jpg", ...}
        """
        prefix = f"captures/{timestamp}/"
        downloaded = {}
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.captures_bucket,
                Prefix=prefix
            )
            
            if "Contents" not in response:
                return downloaded
            
            for obj in response["Contents"]:
                key = obj["Key"]
                filename = Path(key).name
                
                # Determine view from filename
                view = None
                for v in ["side", "top", "front"]:
                    if v in filename.lower():
                        view = v
                        break
                
                if not view:
                    continue
                
                local_path = os.path.join(local_dir, f"{view}.jpg")
                self.s3.download_file(self.captures_bucket, key, local_path)
                downloaded[view] = local_path
            
            return downloaded
            
        except ClientError as e:
            print(f"Error downloading images: {e}")
            return downloaded
    
    def upload_annotated_images(self, timestamp: str, local_paths: Dict[str, str]) -> Dict[str, str]:
        """
        Upload annotated/debug images to processed bucket.
        
        Args:
            timestamp: Processing timestamp
            local_paths: Dict mapping view -> local filepath
        
        Returns:
            Dict mapping view -> S3 key
        """
        uploaded = {}
        
        for view, local_path in local_paths.items():
            if not os.path.exists(local_path):
                continue
            
            s3_key = f"results/{timestamp}/{view}_annotated.jpg"
            
            try:
                self.s3.upload_file(
                    local_path,
                    self.processed_bucket,
                    s3_key,
                    ExtraArgs={"ContentType": "image/jpeg"}
                )
                uploaded[view] = s3_key
            except ClientError as e:
                print(f"Error uploading {view}: {e}")
        
        return uploaded
    
    def upload_results_json(self, timestamp: str, results: dict) -> Optional[str]:
        """Upload results JSON to processed bucket"""
        import json
        
        s3_key = f"results/{timestamp}/measurements.json"
        
        try:
            self.s3.put_object(
                Bucket=self.processed_bucket,
                Key=s3_key,
                Body=json.dumps(results, indent=2),
                ContentType="application/json"
            )
            return s3_key
        except ClientError as e:
            print(f"Error uploading results JSON: {e}")
            return None
    
    def get_annotated_image_url(self, timestamp: str, view: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate a pre-signed URL for an annotated image.
        
        Args:
            timestamp: Processing timestamp
            view: "side", "top", or "front"
            expires_in: URL expiration time in seconds (default 1 hour)
        
        Returns:
            Pre-signed URL or None if image doesn't exist
        """
        s3_key = f"results/{timestamp}/{view}_annotated.jpg"
        
        try:
            # Check if object exists
            self.s3.head_object(Bucket=self.processed_bucket, Key=s3_key)
            
            # Generate pre-signed URL
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.processed_bucket,
                    "Key": s3_key
                },
                ExpiresIn=expires_in
            )
            return url
            
        except ClientError:
            return None
    
    def get_annotated_image_urls(self, timestamp: str, expires_in: int = 3600) -> Dict[str, str]:
        """Get pre-signed URLs for all annotated images"""
        urls = {}
        for view in ["side", "top", "front"]:
            url = self.get_annotated_image_url(timestamp, view, expires_in)
            if url:
                urls[view] = url
        return urls
