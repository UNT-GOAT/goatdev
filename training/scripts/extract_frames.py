#!/usr/bin/env python3
"""
Extract frames from training videos.
Downloads videos from S3, extracts frames, organizes by view.

Usage:
    python extract_frames.py                    # Process all videos
    python extract_frames.py --prefix 20260130  # Process specific date
    python extract_frames.py --fps 10           # Extract 10 frames per second
"""

import argparse
import boto3
import os
import subprocess
from pathlib import Path


S3_BUCKET = 'goat-training-ACCOUNTID'  # TODO: Update this
OUTPUT_DIR = Path('./extracted_frames')


def download_videos(s3, prefix='videos/'):
    """Download all videos from S3 to local temp dir"""
    temp_dir = Path('./temp_videos')
    temp_dir.mkdir(exist_ok=True)
    
    paginator = s3.get_paginator('list_objects_v2')
    
    downloaded = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.mp4'):
                local_path = temp_dir / key.replace('/', '_')
                print(f"Downloading {key}...")
                s3.download_file(S3_BUCKET, key, str(local_path))
                downloaded.append(local_path)
    
    return downloaded


def extract_frames(video_path, output_dir, fps=5):
    """Extract frames from a video using ffmpeg"""
    # Parse video filename: timestamp_goatid_view.mp4
    filename = video_path.stem  # e.g., "20260130_120000_goat42_side"
    parts = filename.split('_')
    
    if len(parts) >= 4:
        view = parts[-1]  # side, top, or front
        goat_id = '_'.join(parts[2:-1])  # Handle goat IDs with underscores
        timestamp = '_'.join(parts[:2])
    else:
        view = 'unknown'
        goat_id = 'unknown'
        timestamp = filename
    
    # Create output directory
    view_dir = output_dir / view
    view_dir.mkdir(parents=True, exist_ok=True)
    
    # Output pattern
    output_pattern = view_dir / f"{timestamp}_{goat_id}_%04d.jpg"
    
    # Extract frames
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality JPEG
        str(output_pattern)
    ]
    
    print(f"Extracting frames from {video_path.name} at {fps} fps...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return 0
    
    # Count extracted frames
    frame_count = len(list(view_dir.glob(f"{timestamp}_{goat_id}_*.jpg")))
    print(f"  Extracted {frame_count} frames")
    return frame_count


def main():
    parser = argparse.ArgumentParser(description='Extract frames from training videos')
    parser.add_argument('--prefix', default='videos/', help='S3 prefix filter')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second to extract')
    parser.add_argument('--output', default='./extracted_frames', help='Output directory')
    parser.add_argument('--local', help='Process local video file instead of S3')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if args.local:
        # Process single local file
        video_path = Path(args.local)
        if video_path.exists():
            extract_frames(video_path, output_dir, args.fps)
        else:
            print(f"File not found: {args.local}")
        return
    
    # Download from S3
    s3 = boto3.client('s3')
    
    print(f"Downloading videos from s3://{S3_BUCKET}/{args.prefix}...")
    videos = download_videos(s3, args.prefix)
    
    if not videos:
        print("No videos found!")
        return
    
    print(f"\nFound {len(videos)} videos. Extracting frames...\n")
    
    total_frames = 0
    for video in videos:
        frames = extract_frames(video, output_dir, args.fps)
        total_frames += frames
        # Clean up video after processing
        video.unlink()
    
    # Clean up temp dir
    temp_dir = Path('./temp_videos')
    if temp_dir.exists():
        temp_dir.rmdir()
    
    print(f"\n{'='*50}")
    print(f"Total frames extracted: {total_frames}")
    print(f"Output directory: {output_dir}")
    print(f"\nFrames organized by view:")
    for view in ['side', 'top', 'front']:
        view_dir = output_dir / view
        if view_dir.exists():
            count = len(list(view_dir.glob('*.jpg')))
            print(f"  {view}: {count} frames")


if __name__ == '__main__':
    main()
