# Example Usage

Assuming you have already installed hydra. 

Suppose your sweep directory is in `/checkpoint/$USER/outputs/2019-12-27/08-38-43`, then you can run:

```
path=2019-12-27/08-38-43; python analyze.py --logdirs $path; python stats.py --logdir $path --key_stats eval_score
```

And it will automatically save to `/checkpoint/$USER/summary/2019-12-27_08-38-43.pkl`, which is a pandas DataFrame that contains all statistics.
