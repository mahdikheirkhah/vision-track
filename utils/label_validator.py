from pathlib import Path
from typing import List
from loguru import logger

class YOLOLabelValidator:
    """
    A utility class to validate YOLO format label files.
    Checks for out-of-bounds coordinates, negative dimensions, and formatting errors.
    """

    def __init__(self, label_dir: str) -> None:
        """
        Initializes the validator with the target directory.

        Args:
            label_dir (str): The path to the directory containing YOLO .txt labels.

        Returns:
            None
        """
        try:
            self.label_dir: Path = Path(label_dir)
            if not self.label_dir.exists():
                raise FileNotFoundError(f"Label directory does not exist: {self.label_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOLabelValidator: {e}")
            raise

    def validate_line(self, line: str, line_num: int, file_path: Path) -> bool:
        """
        Validates a single line of a YOLO label file to ensure it meets constraints.

        Args:
            line (str): The string content of the line.
            line_num (int): The current line number (for logging).
            file_path (Path): The path to the file (for logging).

        Returns:
            bool: True if the line is valid, False otherwise.
        """
        try:
            parts: List[str] = line.strip().split()
            
            # Check for correct number of arguments (class, x_center, y_center, width, height)
            if len(parts) != 5:
                logger.warning(f"{file_path.name} (Line {line_num}): Expected 5 values, found {len(parts)}.")
                return False

            # YOLO format requires floats for coordinates
            x_center: float = float(parts[1])
            y_center: float = float(parts[2])
            width: float = float(parts[3])
            height: float = float(parts[4])

            is_valid: bool = True
            
            # Center coordinates must be exactly between 0.0 and 1.0
            if not (0.0 <= x_center <= 1.0) or not (0.0 <= y_center <= 1.0):
                logger.warning(f"{file_path.name} (Line {line_num}): Center coordinates out of bounds: {x_center}, {y_center}")
                is_valid = False
            
            # Width and height must be > 0 and <= 1.0
            if not (0.0 < width <= 1.0) or not (0.0 < height <= 1.0):
                logger.warning(f"{file_path.name} (Line {line_num}): Dimensions invalid: w={width}, h={height}")
                is_valid = False

            return is_valid

        except ValueError as e:
            logger.warning(f"{file_path.name} (Line {line_num}): Could not parse numerical values: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating line {line_num} in {file_path.name}: {e}")
            raise

    def validate_file(self, file_path: Path) -> bool:
        """
        Reads a single file and validates all its bounding box lines.

        Args:
            file_path (Path): The path to the text file.

        Returns:
            bool: True if the entire file is perfectly formatted, False otherwise.
        """
        try:
            with open(file_path, 'r') as f:
                lines: List[str] = f.readlines()
            
            file_is_valid: bool = True
            for index, line in enumerate(lines, 1):
                if not line.strip():
                    continue  # Ignore empty lines
                
                if not self.validate_line(line, index, file_path):
                    file_is_valid = False
                    
            return file_is_valid
            
        except Exception as e:
            logger.error(f"Failed to read/process file {file_path}: {e}")
            raise

    def run_validation(self) -> None:
        """
        Executes the validation pipeline across all .txt files in the target directory.

        Returns:
            None
        """
        try:
            logger.info(f"Starting bounding box validation in: {self.label_dir}")
            txt_files: List[Path] = list(self.label_dir.glob("*.txt"))
            
            if not txt_files:
                logger.warning("No .txt files found to validate.")
                return

            corrupted_count: int = 0
            for file_path in txt_files:
                if not self.validate_file(file_path):
                    corrupted_count += 1

            if corrupted_count == 0:
                logger.success(f"All {len(txt_files)} label files are perfectly formatted!")
            else:
                logger.error(f"Found {corrupted_count} corrupted files out of {len(txt_files)}. Please delete or fix these files before training.")
                
        except Exception as e:
            logger.error(f"Validation pipeline failed critically: {e}")
            raise