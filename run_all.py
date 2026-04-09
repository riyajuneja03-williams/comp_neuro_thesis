import os
import sys

dir = os.path.dirname(os.path.abspath(__file__))
py = sys.executable

# helper function to run terminal commands
def run_command(cmd):
    os.system(f'"{py}" "{dir}/{cmd}"')

methods = ["ps", "mi", "logisi", "cma", "unified"]

run_command("synthetic_saver.py")
run_command("synthetic_df.py")

for method in methods:
    run_command(f'apply_{method}.py')

for method in methods:
    run_command(f'save_{method}.py')

run_command("analyze_methods.py")
run_command("all_figs.py")
