# Introduction
To track slurm jobs and restart them if they fail. 

# Usage
1. `monitor_jobs.py` is used to monitor running jobs in slurm. It starts an infinite loop and check slurm state every 30 sec.  
2. `monitor_client.py` sends commands to ongoing `monitor_jobs` process (e.g., cancel some jobs with specific job id / regex). 


Start monitoring jobs. If `--job_id` is not specified, it will monitor all RUNNING and PENDING jobs
```
python -u monitor_jobs.py | tee log.log 
```

Specifying job ids. 
```
python -u monitor_jobs.py --job_ids 2345-2348,2510,2502 | tee log.log 
```

From another bash window, send command, e.g., cancel jobs with specific wandb uis. 
```
python monitor_client.py "cancel_wandb_urls `paste -sd, cancel_list.txt`" 
```
