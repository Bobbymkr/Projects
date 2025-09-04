import unittest
import subprocess
import os

class TestLiveSimulation(unittest.TestCase):
    def test_live_simulation_runs(self):
        # Define the path to the live_simulation.py script
        script_path = os.path.join('src', 'rl', 'live_simulation.py')
        
        # Define a dummy config file for the simulation
        # In a real scenario, you might want to create a temporary config file
        # or use an existing minimal one.
        config_path = os.path.join('configs', 'intersection.json')

        # Command to run the live simulation script with a limited number of steps
        # We use --no-display to prevent the OpenCV window from opening during testing
        command = [
            'python',
            script_path,
            '--config', config_path,
            '--steps', '10',
            '--delay', '0.01',
            '--no-display' # Add a flag to disable display for automated testing
        ]

        # Add the project root to PYTHONPATH
        env = os.environ.copy()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        env['PYTHONPATH'] = project_root + os.pathsep + env.get('PYTHONPATH', '')

        try:
            # Run the command as a subprocess
            # capture_output=True captures stdout and stderr
            # text=True decodes stdout/stderr as text
            result = subprocess.run(command, capture_output=True, text=True, check=True, env=env)
            
            # Assert that the command ran successfully (exit code 0)
            self.assertEqual(result.returncode, 0, f"Live simulation failed with error: {result.stderr}")
            
            # Assert that the simulation started and completed
            self.assertIn("Live simulation started in no-display mode.", result.stdout)
            self.assertIn("Simulation completed!", result.stdout)

        except subprocess.CalledProcessError as e:
            self.fail(f"Live simulation subprocess failed: {e.stderr}")
        except FileNotFoundError:
            self.fail(f"Python or script not found. Ensure python is in PATH and script_path is correct: {script_path}")

if __name__ == '__main__':
    unittest.main()