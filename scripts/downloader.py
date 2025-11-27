import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from git import Repo, RemoteProgress

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProgressPrinter(RemoteProgress):
    """
    Progress callback for git clone operations.
    """
    def update(self, op_code: int, cur: int, max_count: int = None, msg: str = ""):
        percent = (cur / max_count * 100) if max_count else cur
        logger.info(f"Cloning progress: {percent:.1f}% {msg}")


def load_config(path: Path) -> Dict:
    """
    Load configuration settings from a JSON file.

    Args:
        path (Path): Path to the config JSON file.

    Returns:
        Dict: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_directories(paths: List[Path]) -> None:
    """Ensure that each path in the list exists, creating it if necessary."""
    for directory in paths:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


class DataLoader:
    """
    Handles downloading (cloning) of repositories specified in the config.
    """
    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir.expanduser()
        ensure_directories([self.raw_dir])

    def git_clone(self, repo_url: str, target_dir: Path) -> bool:
        """
        Clone a git repository to the specified local directory.

        Args:
            repo_url (str): URL of the repository to clone.
            target_dir (Path): Local directory path for cloning.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            logger.info(f"Cloning {repo_url} into {target_dir}")
            Repo.clone_from(repo_url, str(target_dir), progress=ProgressPrinter())
            return True
        except Exception as e:
            logger.error(f"Error cloning {repo_url}: {e}")
            return False

    def load_data(self, training_urls: List[str]) -> List[Path]:
        """
        Clone all repositories listed in training_urls into raw_dir.

        Args:
            training_urls (List[str]): List of repository URLs.

        Returns:
            List[Path]: List of successfully cloned repository paths.
        """
        cloned = []
        for url in training_urls:
            # Use full repository name (including version) to avoid name collisions
            repo_name = Path(url).name
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]
            target = self.raw_dir / repo_name
            if self.git_clone(url, target):
                cloned.append(target)
        return cloned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ML training data by cloning specified Git repositories."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "config.json",
        help="Path to the JSON configuration file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_config(args.config)
        raw = Path(config["data_pipeline"]["raw_dir"])
        extractor = DataLoader(raw)

        urls = config["data_pipeline"].get("training_urls", [])
        cloned_paths = extractor.load_data(urls)

        logger.info(f"Successfully cloned {len(cloned_paths)} repositories into {raw}")
    except Exception as e:
        logger.error(f"Fatal error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main()
