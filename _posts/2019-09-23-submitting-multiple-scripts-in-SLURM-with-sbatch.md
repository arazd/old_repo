---
layout: post
title: "Submitting multiple scripts in SLURM with sbatch"
date: 2019-09-23
---

In this short tutorial I will show how to *submit multiple programs as a single SLURM job*. Often we don't want to submit several similar jobs separately, so we want to join them and submit as one job using **sbatch**. We can make programs inside the job run:
* in parallel
OR
* sequentially

Let's take a look how we can do it. Assume we have a script **my_script.py** inside **my_folder** that we want to run with different batch sizes. We want to run the script on GPU using virtual environment **conda_env**.


<br/>

<span id="highlight">**Running several scripts in parallel**</span>

Create the following sbatch scipt and name it **4_scripts_parallel**:

```shell
#!/bin/bash

#SBATCH --mem=20G # 20 Gb of memory in total
#SBATCH --job-name=4_parallel_scripts
#SBATCH --output=4_parallel_scripts_%j.log
#SBATCH --ntasks=4
#SBATCH -p gpu
#SBATCH --gres=gpu:4
#SBATCH -c 8 # 2 cores per task
#SBATCH -t 05:00:00 # time for the job to run

date;hostname;pwd

# example of activating virtual environment
export PATH=/home/my_username/anaconda3/bin:$PATH
source activate conda_env

# usually we should go to SLURM submit directory
cd $SLURM_SUBMIT_DIR
export XDG_RUNTIME_DIR=""


# Go to the folder with my_script.py
cd /home/my_username/my_folder

# Execute jobs in parallel
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=100 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=50 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=20 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=10
wait
echo 'Finished running 4 parallel scripts!'
```

Submit the job to SLURM queue using this command:

```shell
sbatch 4_scripts_parallel
```

<br/>

<span class='underline_magical'>Now let's break down the sbatch script that we used to run parallel programs</span>.

It starts with defining *overall parameters of the job*, like total amount of memory, total number of tasks, total number of GPUs etc. So if you're running 4 parallel programs and each of them requires a GPU, make sure that you request 4 GPUs at the beginning of the script:

```shell
#!/bin/bash

#SBATCH --mem=20G # 20 Gb of memory in total
#SBATCH --job-name=4_parallel_scripts
#SBATCH --output=4_parallel_scripts_%j.log   # output will be saved in .log file
#SBATCH --ntasks=4   # number of tasks
#SBATCH -p gpu   # partition is gpu
#SBATCH --gres=gpu:4   # total number of GPUs
#SBATCH -c 8 # 2 cores per task
#SBATCH -t 05:00:00 # time for the job to run
```

After that comes a step that may vary depending on your system settings. In our example, we activate virtual environment and go to SLURM submit directory:

```shell  
date;hostname;pwd

# example of activating virtual environment
export PATH=/home/my_username/anaconda3/bin:$PATH
source activate conda_env

# usually we should go to SLURM submit directory
cd $SLURM_SUBMIT_DIR
export XDG_RUNTIME_DIR=""
```

Finally, we go to directory that contains our script and run 4 sequential scripts with different batch sizes:

```shell  
cd /home/my_username/my_folder

# Execute jobs in parallel
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=100 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=50 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=20 &
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=10
wait
echo 'Finished running 4 parallel scripts!'
```

<br/>

<span id="highlight">**Running several scripts one after another**</span>

This is very similar to the above example (the main difference from the previous job is that sequential scripts has no **&** sign after srun).

Create the following sbatch scipt and name it **4_scripts_sequential** :

```shell
#!/bin/bash

#SBATCH --mem=5G # 5 Gb of memory in total
#SBATCH --job-name=4_sequential_scripts
#SBATCH --output=4_sequential_scripts_%j.log
#SBATCH --ntasks=4
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 2 # 2 cores per task
#SBATCH -t 20:00:00 # time for the job to run

date;hostname;pwd

# example of activating virtual environment
export PATH=/home/my_username/anaconda3/bin:$PATH
source activate conda_env

# usually we should go to SLURM submit directory
cd $SLURM_SUBMIT_DIR
export XDG_RUNTIME_DIR=""


# Go to the folder with my_script.py
cd /home/my_username/my_folder

# Execute jobs in parallel
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=100
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=50
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=20
srun -p p100 -n 1 -c 2 --mem=5G --gres=gpu:1 python my_script.py --batch_size=10
wait
echo 'Finished running 4 parallel scripts!'
```

Submit the job to SLURM queue using this command:

```shell
sbatch 4_scripts_sequential
```

That's it. Hope this was helpful!
