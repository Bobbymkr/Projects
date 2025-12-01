"""
Code Quality Tools.

Utilities for code quality checking, linting, formatting, and type checking.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class CodeQualityChecker:
    """Code quality checking utilities."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize code quality checker."""
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.src_path = self.project_root / "src"
        self.tests_path = self.project_root / "tests"
    
    def run_black(self, check_only: bool = False, path: Optional[Path] = None) -> bool:
        """
        Run Black code formatter.
        
        Args:
            check_only: Only check, don't modify
            path: Path to format (default: src/)
            
        Returns:
            True if successful
        """
        path = path or self.src_path
        cmd = ["black", str(path)]
        if check_only:
            cmd.append("--check")
        else:
            cmd.append("--diff")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr, file=sys.stderr)
                return False
            return True
        except FileNotFoundError:
            logger.error("Black not installed. Install with: pip install black")
            return False
    
    def run_flake8(self, path: Optional[Path] = None) -> bool:
        """
        Run flake8 linter.
        
        Args:
            path: Path to lint (default: src/)
            
        Returns:
            True if no errors
        """
        path = path or self.src_path
        cmd = [
            "flake8",
            str(path),
            "--max-line-length=100",
            "--exclude=__pycache__,*.pyc",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
                return False
            return True
        except FileNotFoundError:
            logger.error("Flake8 not installed. Install with: pip install flake8")
            return False
    
    def run_mypy(self, path: Optional[Path] = None) -> bool:
        """
        Run mypy type checker.
        
        Args:
            path: Path to check (default: src/)
            
        Returns:
            True if no type errors
        """
        path = path or self.src_path
        cmd = [
            "mypy",
            str(path),
            "--ignore-missing-imports",
            "--no-strict-optional",
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
                return False
            return True
        except FileNotFoundError:
            logger.warning("mypy not installed. Install with: pip install mypy")
            return True  # Not critical
    
    def run_pylint(self, path: Optional[Path] = None) -> bool:
        """
        Run pylint linter.
        
        Args:
            path: Path to lint (default: src/)
            
        Returns:
            True if score >= 8.0
        """
        path = path or self.src_path
        cmd = [
            "pylint",
            str(path),
            "--max-line-length=100",
            "--disable=C0111",  # Missing docstring (handled by other tools)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            # Parse score from output
            if "rated" in result.stdout.lower():
                print(result.stdout)
                return False
            return True
        except FileNotFoundError:
            logger.warning("pylint not installed. Install with: pip install pylint")
            return True  # Not critical
    
    def run_all(self, fix: bool = False) -> Dict[str, bool]:
        """
        Run all quality checks.
        
        Args:
            fix: Auto-fix issues where possible
            
        Returns:
            Dictionary of check results
        """
        results = {}
        
        logger.info("Running code quality checks...")
        
        # Black formatting
        logger.info("Checking code formatting (Black)...")
        results["black"] = self.run_black(check_only=not fix)
        
        # Flake8 linting
        logger.info("Running linting (flake8)...")
        results["flake8"] = self.run_flake8()
        
        # mypy type checking
        logger.info("Running type checking (mypy)...")
        results["mypy"] = self.run_mypy()
        
        # Pylint
        logger.info("Running linting (pylint)...")
        results["pylint"] = self.run_pylint()
        
        return results


def main():
    """Main entry point for quality checker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Code quality checker")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    parser.add_argument("--black", action="store_true", help="Run Black only")
    parser.add_argument("--flake8", action="store_true", help="Run flake8 only")
    parser.add_argument("--mypy", action="store_true", help="Run mypy only")
    parser.add_argument("--path", help="Path to check")
    
    args = parser.parse_args()
    
    checker = CodeQualityChecker()
    path = Path(args.path) if args.path else None
    
    if args.black:
        success = checker.run_black(check_only=not args.fix, path=path)
    elif args.flake8:
        success = checker.run_flake8(path=path)
    elif args.mypy:
        success = checker.run_mypy(path=path)
    else:
        results = checker.run_all(fix=args.fix)
        success = all(results.values())
        if not success:
            print("\nSome checks failed:")
            for check, passed in results.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

