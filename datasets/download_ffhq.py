import os
import subprocess
import zipfile
import shutil

def download_ffhq_from_kaggle():
    """Download FFHQ dataset from Kaggle"""
    
    print("🎭 FFHQ Kaggle Downloader")
    print("="*40)
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API found")
    except ImportError:
        print("❌ Kaggle not found. Installing...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        print("✅ Kaggle installed")
    
    # Setup directories
    data_dir = "data"
    ffhq_dir = os.path.join(data_dir, "ffhq")
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if already downloaded
    if os.path.exists(ffhq_dir):
        image_count = len([f for f in os.listdir(ffhq_dir) if f.endswith(('.png', '.jpg'))])
        if image_count > 10000:
            print(f"✅ FFHQ already downloaded ({image_count} images)")
            return
    
    print("📥 Downloading FFHQ from Kaggle...")
    print("💡 This may take 10-30 minutes depending on your connection")
    
    try:
        # Download dataset
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", "lamsimon/ffhq",
            "-p", data_dir
        ], check=True)
        
        # Find the downloaded zip file
        zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip') and 'ffhq' in f.lower()]
        
        if not zip_files:
            print("❌ Download failed - no zip file found")
            return
        
        zip_path = os.path.join(data_dir, zip_files[0])
        print(f"📦 Extracting {zip_files[0]}...")
        
        # Extract to ffhq directory
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffhq_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        
        # Count final images
        image_count = 0
        for root, dirs, files in os.walk(ffhq_dir):
            image_count += len([f for f in files if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"✅ FFHQ download complete!")
        print(f"📊 Total images: {image_count}")
        print(f"📁 Location: {os.path.abspath(ffhq_dir)}")
        print("\n🚀 You can now run: python main.py --datasets ffhq")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\n💡 Manual alternative:")
        print("1. Go to: https://www.kaggle.com/datasets/lamsimon/ffhq")
        print("2. Download manually")
        print("3. Extract to: data/ffhq/")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    download_ffhq_from_kaggle()