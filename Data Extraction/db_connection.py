import os
import socket
import logging
import subprocess
import platform
import ipaddress
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient
import ssl
import urllib.request

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# MongoDB Connection Parameters
USERNAME = os.getenv('MONGODB_USERNAME')
PASSWORD = os.getenv('MONGODB_PASSWORD')
CLUSTER = os.getenv('MONGODB_CLUSTER')
DATABASE = os.getenv('MONGODB_DATABASE', 'default_database')
APP_NAME = os.getenv('APP_NAME', 'Component-Cluster')

def run_system_command(command):
    """
    Run system command and capture output
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def advanced_hostname_resolution(hostname: str) -> str:
    """
    Advanced hostname resolution with multiple strategies
    """
    # Possible hostname variations
    hostname_variations = [
        hostname,
        f"cluster0.{hostname}",
        f"cluster1.{hostname}"
    ]

    # DNS resolution strategies
    resolution_strategies = [
        # Strategy 1: socket.gethostbyname
        lambda h: socket.gethostbyname(h),
        
        # Strategy 2: urllib.request
        lambda h: socket.gethostbyname(urllib.request.urlopen(f'http://{h}').geturl().split('//')[1].split('/')[0]),
        
        # Strategy 3: Multiple DNS servers
        lambda h: next(
            socket.gethostbyname(h) 
            for dns in ['8.8.8.8', '1.1.1.1', '9.9.9.9']
            if socket.socket(socket.AF_INET, socket.SOCK_DGRAM).bind(('', 0))
        )
    ]

    # Try resolution strategies
    for variant in hostname_variations:
        for strategy in resolution_strategies:
            try:
                ip = strategy(variant)
                
                # Validate IP address
                ipaddress.ip_address(ip)
                
                logger.info(f"Successfully resolved {variant} to {ip}")
                return ip
            except Exception as e:
                logger.warning(f"Resolution failed for {variant}: {e}")
    
    raise socket.gaierror(f"Could not resolve hostname: {hostname}")

def diagnose_network_connectivity():
    """
    Comprehensive network connectivity diagnosis
    """
    print("\n--- Network Connectivity Diagnostics ---")
    
    # Operating System Information
    print(f"Operating System: {platform.platform()}")
    
    # Network Configuration
    print("\nNetwork Configuration:")
    if platform.system() == "Windows":
        stdout, stderr = run_system_command("ipconfig /all")
        print(stdout)
    else:
        stdout, stderr = run_system_command("ifconfig")
        print(stdout)
    
    # Advanced Hostname Resolution Test
    print("\nAdvanced Hostname Resolution:")
    try:
        ip = advanced_hostname_resolution(CLUSTER)
        print(f"Resolved IP: {ip}")
    except Exception as e:
        print(f"Hostname resolution failed: {e}")
    
    # DNS Resolution Tests
    print("\nDNS Resolution Tests:")
    dns_servers = ['8.8.8.8', '1.1.1.1', '9.9.9.9']
    
    for dns in dns_servers:
        print(f"\nTesting DNS resolution with {dns}:")
        try:
            cmd = f"nslookup {CLUSTER} {dns}"
            stdout, stderr = run_system_command(cmd)
            print(stdout)
        except Exception as e:
            print(f"DNS resolution failed: {e}")
    
    # Traceroute to MongoDB cluster
    print("\nTraceroute to MongoDB Cluster:")
    if platform.system() == "Windows":
        stdout, stderr = run_system_command(f"tracert {CLUSTER}")
    else:
        stdout, stderr = run_system_command(f"traceroute {CLUSTER}")
    print(stdout)

def get_mongodb_client(timeout_ms: int = 5000) -> Optional[MongoClient]:
    """
    Establish a secure MongoDB connection with comprehensive diagnostics
    """
    try:
        # Validate connection parameters
        if not all([USERNAME, PASSWORD, CLUSTER]):
            raise ValueError("Missing MongoDB connection parameters")
        
        # Resolve hostname
        try:
            advanced_hostname_resolution(CLUSTER)
        except Exception as resolution_error:
            logger.critical(f"Hostname resolution failed: {resolution_error}")
            raise
        
        # Construct connection string
        connection_string = (
            f"mongodb+srv://{USERNAME}:{PASSWORD}@{CLUSTER}/"
            "?retryWrites=true&w=majority"
        )
        
        # Enhanced connection options with SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        client = MongoClient(
            connection_string,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
            tls=True,
            tlsContext=ssl_context
        )
        
        # Verify connection with a ping
        client.admin.command('ping')
        
        logger.info("Successfully established MongoDB connection")
        return client
    
    except Exception as e:
        logger.critical(f"Detailed MongoDB Connection Error: {e}")
        raise

# Run comprehensive network diagnostics
if __name__ == '__main__':
    diagnose_network_connectivity()