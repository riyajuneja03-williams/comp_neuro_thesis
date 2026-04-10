import os
import sys

dir = os.path.dirname(os.path.abspath(__file__))
py = sys.executable

# helper function to run terminal commands
def run_command(cmd):
    os.system(f'"{py}" "{dir}/{cmd}"')

methods = ["ps", "mi", "logisi", "cma", "unified"]

# generate synthetic data
run_command("synthetic_saver.py")
run_command("synthetic_df.py")

# apply BD methods to synthetic data
for method in methods:
    run_command(f'apply_{method}.py')

for method in methods:
    run_command(f'save_{method}.py')

# analyze results
run_command("analyze_methods.py")
run_command("all_figs.py")

# import PD data
run_command("pd_data.py")
