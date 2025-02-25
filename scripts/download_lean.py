import subprocess
import os
from pathlib import Path
import sys
import argparse
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import RAW_DATA_DIR, LEAN4_GITHUB_REPO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_lean_repo(repo_url=LEAN4_GITHUB_REPO, target_dir=None, depth=1):
    """Download the Lean 4 GitHub repository."""
    if target_dir is None:
        target_dir = RAW_DATA_DIR / "lean4"
    
    os.makedirs(target_dir, exist_ok=True)
    
    logger.info(f"Cloning Lean 4 repository from {repo_url} to {target_dir}...")
    
    try:
        # Use depth parameter to limit clone size for large repositories
        subprocess.run(
            ["git", "clone", "--depth", str(depth), repo_url, str(target_dir)],
            check=True
        )
        logger.info("Successfully cloned Lean 4 repository.")
        
        # Also clone mathlib4 for more examples
        mathlib_dir = Path(target_dir).parent / "mathlib4"
        if not mathlib_dir.exists():
            logger.info("Cloning mathlib4 repository...")
            subprocess.run(
                ["git", "clone", "--depth", str(depth), "https://github.com/leanprover-community/mathlib4.git", str(mathlib_dir)],
                check=True
            )
            logger.info("Successfully cloned mathlib4 repository.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download Lean 4 repositories")
    parser.add_argument("--repo", default=LEAN4_GITHUB_REPO, help="Lean 4 repository URL")
    parser.add_argument("--target", default=None, help="Target directory")
    parser.add_argument("--depth", type=int, default=1, help="Git clone depth")
    
    args = parser.parse_args()
    download_lean_repo(args.repo, args.target, args.depth)

if __name__ == "__main__":
    main()