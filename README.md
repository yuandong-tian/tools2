# Example Usage

Assuming you have already installed hydra. 

Suppose your sweep directory is in `/checkpoint/$USER/outputs/2019-12-27/08-38-43`, then you can run:

```
path=2019-12-27/08-38-43; python analyze.py --logdirs $path; python stats.py --logdir $path --key_stats eval_score
```

And it will automatically save to `/checkpoint/$USER/summary/2019-12-27_08-38-43.pkl`, which is a pandas DataFrame that contains all statistics.

# Usage without `/checkpoint`

For other usage, try the following with a folder generated by hydra
```
python analyze.py --logdirs [your saved directory with hydra]  --output_dir [your output folder] --num_process 1  --no_sub_folder --path_outside_checkpoint
```
For sweep generated by hydra, run the following:
```
python analyze.py --logdirs [your sweep directory]  --output_dir [your output folder] --num_process 1  --path_outside_checkpoint
```

# Automatic generate md file to visualize result
Prepare a file (e.g., `check_list.txt`) with the content like:
```
Test hyper parameter a
[your sweep filename]

Test hyper parameter b
[your sweep filename]
```

Then you run in your current code repo:
```
python ~/tools2/serve_result.py check_list.txt --match_file `pwd`/match_log3.json --output_prefix progress --key_stats acc --descending --check_freq 60
```
Then it will create `progress.md` in your current folder every 60 sec. You can use `markserv` to serve the file for web browsing, e.g.,  
```
markserv -p 5000 progress.md
```
Please check https://github.com/markserv/markserv

# Using distributed sweep tools
Install Ray and then run the following:

On the host machine.
```
sh ./start_ray_server.sh
```

On other machines (note that the ip address of the host machine is hard-coded for now and would need to change)
```
sh ./start_ray_client.sh
```

Then prepare a task list (called `tasks.txt`) and run
```
RAY_ADDRESS="auto" python [your path to tools2]/distribute_ray.py tasks.txt
```
