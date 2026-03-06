#!/usr/bin/env python3
"""
Model downloader for Hermit Crab.
Downloads a quantized Phi-2 model from HuggingFace.
This model was chosen because:
1. 1.6B parameters fits in <2GB RAM with Q4_K_M quantization
2. Apache 2.0 license (commercial friendly)
3. Good performance for its size
4. Modern architecture with strong reasoning capabilities
"""

import os
import sys
import hashlib
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_INFO = {
    "name": "phi-2.Q4_K_M.gguf",
    "url": "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf",
    "expected_md5": "a5a8c0f5b5c5b5c5b5c5b5c5b5c5b5c5",  # Placeholder - update after download
    "size_mb": 1024  # Approximately 1GB
}

def calculate_file_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except FileNotFoundError:
        return ""

def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress tracking."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 10%
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if int(percent) % 10 == 0:
                            logger.info(f"Downloaded {downloaded}/{total_size} bytes ({percent:.1f}%)")
        
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False

def main():
    """Main download function."""
    models_dir = Path("models")
    models_dir.m