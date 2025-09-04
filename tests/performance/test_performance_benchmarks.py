import unittest
import subprocess
import sys
import os

class TestPerformanceBenchmarks(unittest.TestCase):
    def setUp(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.python_path = os.path.abspath(os.path.join(self.project_root, 'src'))
        self.env = os.environ.copy()
        if 'PYTHONPATH' in self.env:
            self.env['PYTHONPATH'] = f"{self.python_path};{self.env['PYTHONPATH']}"
        else:
            self.env['PYTHONPATH'] = self.python_path

    def test_dqn_benchmark_script(self):
        """Test that benchmark_dqn.py runs and produces output."""
        script_path = os.path.join(self.project_root, 'src', 'rl', 'benchmark_dqn.py')
        command = [sys.executable, script_path]
        
        # Run the script as a subprocess
        process = subprocess.run(command, capture_output=True, text=True, env=self.env, cwd=self.project_root)
        
        # Assert that the command ran successfully
        self.assertEqual(process.returncode, 0, f"Script failed with error: {process.stderr}")
        self.assertIn("SB3: Time=", process.stdout)
        self.assertIn("Custom: Time=", process.stdout)
        print(f"\nbenchmark_dqn.py output:\n{process.stdout}")

    def test_methods_benchmark_script(self):
        """Test that benchmark_methods.py runs and produces output."""
        script_path = os.path.join(self.project_root, 'src', 'rl', 'benchmark_methods.py')
        command = [sys.executable, script_path]
        
        # Run the script as a subprocess
        process = subprocess.run(command, capture_output=True, text=True, env=self.env, cwd=self.project_root)
        
        # Assert that the command ran successfully
        self.assertEqual(process.returncode, 0, f"Script failed with error: {process.stderr}")
        self.assertIn("Average wait_time Comparison", process.stdout) # This will be in the matplotlib output
        self.assertIn("Average queue_length Comparison", process.stdout)
        self.assertIn("Average efficiency Comparison", process.stdout)
        print(f"\nbenchmark_methods.py output:\n{process.stdout}")

if __name__ == '__main__':
    unittest.main()