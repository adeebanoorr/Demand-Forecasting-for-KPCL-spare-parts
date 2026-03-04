import os
import shutil
from pathlib import Path

def sync_assets():
    project_root = Path(__file__).resolve().parent.parent
    webapp_root = project_root / "src" / "webapp" / "public" / "assets"
    
    # Paths to sync
    data_src = project_root / "data" / "processed"
    fig_src = project_root / "reports" / "figures"
    
    # Create target dirs
    webapp_root.mkdir(parents=True, exist_ok=True)
    
    # Sync Data (CSVs)
    data_dest = webapp_root / "data"
    if data_dest.exists():
        shutil.rmtree(data_dest)
    shutil.copytree(data_src, data_dest, ignore=shutil.ignore_patterns('*.gitkeep'))
    print(f"Synced data to {data_dest}")
    
    # Sync Figures (PNGs)
    fig_dest = webapp_root / "figures"
    if fig_dest.exists():
        shutil.rmtree(fig_dest)
    shutil.copytree(fig_src, fig_dest, ignore=shutil.ignore_patterns('*.gitkeep'))
    print(f"Synced figures to {fig_dest}")

if __name__ == "__main__":
    sync_assets()
