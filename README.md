DESCRIPTION:
This repo contains all the code for my computationsl neuroscience thesis in Python. Broadly, we are working on developing and evaluating a pipeline in Python for burst detection in neural spike trains, starting with synthetic data and then applying the methods to rodent data (healthy and Parkinsonian).

HOW TO RUN: 
Steps to run:
1. Create thesis directory.
2. Activate virtual environment (see below)
3. Run scripts: $ python *script_name.py*

--> synthetic_saver.py: iterate through parameters to create directories & files
--> synthetic_df.py: create dataframe from files
--> apply_ps.py: apply poisson surprise method to spike trains & save to files
--> apply_mi.py: apply MaxInterval method to spike trains & save to files
--> save_ps.py: append poisson surprise data to dataframe
--> save_mi.py: append MaxInterval data to dataframe
--> all_figs.py: create & save all figures

--> fig_create.py: helper function to create figures (don't need to run)
--> logisi.py: helper function to detect bursts using LogISI method (don't need to run)
--> maxinterval.py: helper function to detect bursts using MaxInterval method (don't need to run)
--> poissonsurprise.py: helper function to detect bursts using poisson surprise method (don't need to run)
--> synspiketrain.py: helper function to generate synthetic spike trains (don't need to run)
--> stats.py: helper function to generate spike & burst statistics for spike train (don't need to run)

4. All data & figures will be saved in the thesis directory

VIRTUAL ENVIRONMENT
Python version 3.10.7 was used for this repository. Follow the instructions at https://docs.python.org/3/library/venv.html to create the virtual environment on your machine. Then, install all the required dependencies/Python modules for this repository by running $ pip install -r requirements.txt
