import os
import subprocess
import argparse
import sys

def main():
    print("🚀 Starting Automated Setup for CR-JEPA Foundation Model...")
    
    # Ensure dependencies are installed
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    
    # 1. Download Foundation Weights
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_weights", "copernicus_fm")
    os.makedirs(weights_dir, exist_ok=True)
    
    expected_path = os.path.join(weights_dir, "CopernicusFM_ViT_base_varlang_e100.pth")
    if not os.path.exists(expected_path):
        print(f"Downloading Copernicus-FM ViT-Base Foundation Weights to {weights_dir}...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="wangyi111/Copernicus-FM", 
                filename="CopernicusFM_ViT_base_varlang_e100.pth",
                local_dir=weights_dir,
                local_dir_use_symlinks=False
            )
            print("✅ Weights successfully downloaded!")
        except Exception as e:
            print(f"❌ Failed to download weights via API: {e}")
            print("Attempting to use CLI...")
            subprocess.check_call(["hf", "download", "wangyi111/Copernicus-FM", "CopernicusFM_ViT_base_varlang_e100.pth", "--local-dir", weights_dir])
    else:
        print("✅ Foundation weights already exist!")

    print("✅ Setup Complete! You are ready to train.")

if __name__ == "__main__":
    main()
