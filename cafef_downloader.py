#!/usr/bin/env python3
"""
CafeF Data Downloader - Python Version
Downloads and extracts CafeF trading data files with interactive interface
"""

import os
import sys
import zipfile
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Configuration
BASE_URL = "https://cafef1.mediacdn.vn/data/ami_data"
FILE_PREFIX = "CafeF.SolieuGD.Upto"
DOWNLOAD_DIR = "./cafef_data"
EXTRACT_DIR = "./cafef_data/extracted"
AUTO_UNZIP = True

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_status(message: str) -> None:
    """Print status message in green"""
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {message}")

def print_warning(message: str) -> None:
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}[WARN]{Colors.NC} {message}")

def print_error(message: str) -> None:
    """Print error message in red"""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")

def print_header(message: str) -> None:
    """Print header message in cyan"""
    print(f"{Colors.CYAN}{message}{Colors.NC}")

class CafeFDownloader:
    def __init__(self, download_dir: str = DOWNLOAD_DIR, extract_dir: str = EXTRACT_DIR):
        self.download_dir = Path(download_dir)
        self.extract_dir = Path(extract_dir)
        self.auto_unzip = AUTO_UNZIP
        self.session = self._create_session()
        
        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.extract_dir.mkdir(parents=True, exist_ok=True)

    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        return session

    @staticmethod
    def validate_date_format(date_str: str, date_format: str) -> bool:
        """Validate date format"""
        try:
            if date_format == "YYYYMMDD":
                datetime.strptime(date_str, "%Y%m%d")
                return len(date_str) == 8 and date_str.isdigit()
            elif date_format == "YYYY-MM-DD":
                datetime.strptime(date_str, "%Y-%m-%d")
                return True
        except ValueError:
            return False
        return False

    @staticmethod
    def convert_date(date_str: str) -> str:
        """Convert YYYYMMDD to DDMMYYYY format"""
        if len(date_str) == 8:
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            return f"{day}{month}{year}"
        return date_str

    def get_user_input(self, prompt: str, date_format: str = None, 
                      input_type: str = "string") -> str:
        """Get and validate user input"""
        while True:
            try:
                user_input = input(f"{prompt}: ").strip()
                
                if input_type == "days":
                    if not user_input:  # Default value
                        return "7"
                    days = int(user_input)
                    if days > 0:
                        return str(days)
                    else:
                        print_error("Please enter a positive number")
                        continue
                
                elif date_format:
                    if self.validate_date_format(user_input, date_format):
                        return user_input
                    else:
                        print_error(f"Invalid date format. Please use {date_format}")
                        continue
                
                elif input_type == "yes_no":
                    if user_input.lower() in ['y', 'yes', '']:
                        return "true"
                    elif user_input.lower() in ['n', 'no']:
                        return "false"
                    else:
                        print_error("Please enter 'y' for yes or 'n' for no")
                        continue
                
                return user_input
                
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                sys.exit(0)
            except ValueError:
                print_error("Please enter a valid number")

    def download_file(self, date_folder: str, date_converted: str) -> bool:
        """Download a single file"""
        url = f"{BASE_URL}/{date_folder}/{FILE_PREFIX}{date_converted}.zip"
        filename = f"{FILE_PREFIX}{date_converted}.zip"
        file_path = self.download_dir / filename
        
        print_status(f"Downloading: {filename}")
        
        try:
            # Check if file already exists
            if file_path.exists():
                print_warning(f"File already exists: {filename}")
                choice = input("Overwrite? (y/n): ").lower()
                if choice not in ['y', 'yes']:
                    print_status("Skipping download")
                    if self.auto_unzip:
                        return self.extract_file(file_path, date_converted)
                    return True
            
            # Download with progress bar
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
            
            print_status(f"Successfully downloaded: {filename}")
            
            # Extract if enabled
            if self.auto_unzip:
                return self.extract_file(file_path, date_converted)
            
            return True
            
        except requests.exceptions.RequestException as e:
            print_error(f"Failed to download {filename}: {str(e)}")
            return False
        except Exception as e:
            print_error(f"Unexpected error downloading {filename}: {str(e)}")
            return False

    def extract_file(self, zip_path: Path, date_converted: str) -> bool:
        """Extract a ZIP file"""
        extract_path = self.extract_dir / date_converted
        extract_path.mkdir(parents=True, exist_ok=True)
        
        print_status(f"Extracting: {zip_path.name}")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            print_status(f"Successfully extracted: {zip_path.name}")
            
            # List extracted files
            extracted_files = list(extract_path.iterdir())
            if extracted_files:
                print_status("Extracted files:")
                for file in extracted_files:
                    if file.is_file():
                        file_size = file.stat().st_size
                        size_mb = file_size / (1024 * 1024)
                        print(f"  {file.name} ({size_mb:.2f} MB)")
            
            return True
            
        except zipfile.BadZipFile:
            print_error(f"Invalid ZIP file: {zip_path.name}")
            return False
        except Exception as e:
            print_error(f"Failed to extract {zip_path.name}: {str(e)}")
            return False

    def generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """Generate list of dates between start and end dates"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start > end:
                print_error("Start date must be before end date")
                return []
            
            dates = []
            current = start
            while current <= end:
                dates.append(current.strftime("%Y%m%d"))
                current += timedelta(days=1)
            
            return dates
            
        except ValueError as e:
            print_error(f"Invalid date format: {str(e)}")
            return []

    def download_single(self, date_input: Optional[str] = None) -> None:
        """Download single file"""
        if not date_input:
            date_input = self.get_user_input(
                "Enter date (YYYYMMDD, e.g., 20250529)", 
                "YYYYMMDD"
            )
        
        date_folder = date_input
        date_converted = self.convert_date(date_input)
        self.download_file(date_folder, date_converted)

    def download_range(self, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> None:
        """Download files for date range"""
        if not start_date:
            start_date = self.get_user_input(
                "Enter start date (YYYY-MM-DD, e.g., 2025-05-25)", 
                "YYYY-MM-DD"
            )
        
        if not end_date:
            end_date = self.get_user_input(
                "Enter end date (YYYY-MM-DD, e.g., 2025-05-30)", 
                "YYYY-MM-DD"
            )
        
        print_status(f"Generating date range from {start_date} to {end_date}")
        dates = self.generate_date_range(start_date, end_date)
        
        if not dates:
            return
        
        success_count = 0
        total_count = len(dates)
        
        for date in dates:
            date_converted = self.convert_date(date)
            if self.download_file(date, date_converted):
                success_count += 1
        
        print()
        print_status(f"Download summary: {success_count}/{total_count} files downloaded successfully")

    def download_recent(self, days: Optional[int] = None) -> None:
        """Download recent files"""
        if days is None:
            days_input = self.get_user_input(
                "Enter number of days (default: 7)", 
                input_type="days"
            )
            days = int(days_input)
        
        print_status(f"Downloading last {days} days of data")
        
        success_count = 0
        total_count = days
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            date_converted = self.convert_date(date_str)
            
            if self.download_file(date_str, date_converted):
                success_count += 1
        
        print()
        print_status(f"Download summary: {success_count}/{total_count} files downloaded successfully")

    def show_interactive_menu(self) -> None:
        """Show interactive menu"""
        print_header("=== CafeF Data Downloader - Python Version ===")
        print()
        
        # Ask for unzip preference
        if self.auto_unzip is None:
            unzip_pref = self.get_user_input(
                "Do you want to automatically unzip downloaded files? (y/n, default: y)",
                input_type="yes_no"
            )
            self.auto_unzip = unzip_pref == "true"
            print()
        
        while True:
            print("Please choose an option:")
            print("1) Download single file")
            print("2) Download date range")
            print("3) Download recent files")
            print(f"4) Toggle auto-unzip (currently: {'ON' if self.auto_unzip else 'OFF'})")
            print("5) Exit")
            print()
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    print()
                    print_status("Single file download selected")
                    self.download_single()
                    break
                    
                elif choice == "2":
                    print()
                    print_status("Date range download selected")
                    self.download_range()
                    break
                    
                elif choice == "3":
                    print()
                    print_status("Recent files download selected")
                    self.download_recent()
                    break
                    
                elif choice == "4":
                    self.auto_unzip = not self.auto_unzip
                    status = "enabled" if self.auto_unzip else "disabled"
                    print_status(f"Auto-unzip {status}")
                    print()
                    
                elif choice == "5":
                    print_status("Exiting...")
                    sys.exit(0)
                    
                else:
                    print_error("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="CafeF Data Downloader - Python Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 cafef_downloader.py                    # Interactive mode
  python3 cafef_downloader.py single             # Prompts for date
  python3 cafef_downloader.py single 20250529    # Direct download
  python3 cafef_downloader.py range              # Prompts for dates
  python3 cafef_downloader.py range 2025-05-25 2025-05-30  # Direct download
  python3 cafef_downloader.py recent             # Prompts for days
  python3 cafef_downloader.py recent 10          # Last 10 days
  python3 cafef_downloader.py --no-unzip single 20250529   # No extraction
        """
    )
    
    parser.add_argument(
        'command', 
        nargs='?', 
        choices=['single', 'range', 'recent', 'interactive', 'menu'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args', 
        nargs='*', 
        help='Arguments for the command'
    )
    
    parser.add_argument(
        '--no-unzip', 
        action='store_true',
        help='Disable automatic unzipping'
    )
    
    parser.add_argument(
        '--download-dir',
        default=DOWNLOAD_DIR,
        help=f'Download directory (default: {DOWNLOAD_DIR})'
    )
    
    parser.add_argument(
        '--extract-dir',
        default=EXTRACT_DIR,
        help=f'Extract directory (default: {EXTRACT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = CafeFDownloader(args.download_dir, args.extract_dir)
    
    # Set auto-unzip preference
    if args.no_unzip:
        downloader.auto_unzip = False
    
    # Execute command
    if not args.command or args.command in ['interactive', 'menu']:
        downloader.show_interactive_menu()
        
    elif args.command == 'single':
        date_arg = args.args[0] if args.args else None
        downloader.download_single(date_arg)
        
    elif args.command == 'range':
        start_date = args.args[0] if len(args.args) > 0 else None
        end_date = args.args[1] if len(args.args) > 1 else None
        downloader.download_range(start_date, end_date)
        
    elif args.command == 'recent':
        days = int(args.args[0]) if args.args else None
        downloader.download_recent(days)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)