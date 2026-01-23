**DESCRIPTION**

This repo contains all the code for my computationsl neuroscience thesis in Python. Broadly, we are working on developing and evaluating a pipeline in Python for burst detection in neural spike trains, starting with synthetic data and then applying the methods to rodent data (healthy and Parkinsonian).


**HOW TO RUN**

Steps to run:
1. Create thesis directory.
2. Activate virtual environment (see below)
3. Run scripts: $ python *script_name.py*

    a. synthetic_saver.py: iterate through parameters to create directories & files

    b. synthetic_df.py: create dataframe from files
   
    c. apply_ps.py: apply poisson surprise method to spike trains & save to files
   
    d. apply_mi.py: apply MaxInterval method to spike trains & save to files

    e. apply_logisi.py: apply LogISI method to spike trains & save to files

    f. apply_cma.py: apply CMA method to spike trains & save to files
   
    g. save_ps.py: append poisson surprise data to dataframe
   
    h. save_mi.py: append MaxInterval data to dataframe

    i. save_logisi.py: append LogISI data to dataframe

    j. save_cma.py: append CMA data to dataframe
   
    k. all_figs.py: create & save all figures

   
5. All data & figures will be saved in the thesis directory


**Scripts that don't need to be run**

a. fig_create.py: helper function to create figures 
    
b. logisi.py: helper function to detect bursts using LogISI method 
    
c. maxinterval.py: helper function to detect bursts using MaxInterval method
    
d. poissonsurprise.py: helper function to detect bursts using poisson surprise method 

e. cma.py: helper function to detect bursts using cumulative moving average method 

f. synspiketrain.py: helper function to generate synthetic spike trains 
    
g. stats.py: helper function to generate spike & burst statistics for spike train 


**VIRTUAL ENVIRONMENT**

Python version 3.10.7 was used for this repository. Follow the instructions at https://docs.python.org/3/library/venv.html to create the virtual environment on your machine. Then, install all the required dependencies/Python modules for this repository by running $ pip install -r requirements.txt
