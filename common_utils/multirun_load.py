import glob
import re
import multiprocessing as mp
import tqdm

class MeanStdAggFunc:
    def __init__(self, precision=2, use_latex=False):
        self.precision = precision
        self.use_latex = use_latex

    def agg(self, series):
        mean_val = series.mean()
        std_val = series.std()
        nona_cnt = series.count()
        if self.use_latex:
            return fr"${mean_val:.{self.precision}f}\pm {std_val:.{self.precision}f}$"
        else:
            return f"{mean_val:.{self.precision}f} ± {std_val:.{self.precision}f} [{nona_cnt}]"


class FolderAggFunc:
    def __init__(self, max_simple=5):
        self.max_simple = max_simple

    def agg(self, series):
        try:
            ss = sorted([ int(os.path.basename(f)) for f in series ])
        except:
            ss = [0]

        if len(ss) == 0:
            return "[0]"

        # [6,7,8,10,12,13] -> 6-8,10,12-13
        last_start = ss[0]
        last_end = ss[0]
        output = []
        for s in ss[1:]:
            if s > last_end + 1:
                if last_end > last_start:
                    output.append(f"{last_start}-{last_end}")
                else:
                    output.append(f"{last_start}")
                last_start = last_end = s
            else:
                last_end = s

        if last_end > last_start:
            output.append(f"{last_start}-{last_end}")
        else:
            output.append(f"{last_start}")

        return ",".join(output) + f"[{series.size}]"

subfolder_matcher = re.compile(r"\d+")

def batch_load(root, load_one_func, num_process=16):
    '''
Example usage: 

First define load_one_func, e.g., 

def load_one_func(subfolder):
    data = torch.load(os.path.join(subfolder, "final.pth"), map_location="cpu")
    cfg = common_utils.MultiRunUtil.load_cfg(subfolder)
    cfg = { c.split("=")[0] : c.split("=")[1] for c in cfg }
    
    max_corrs = data["all_corrs_zero_mean"][0].max(dim=1)[0]
    # perfect_score measures how perfect the correlation is
    perfect_score = max_corrs[max_corrs > 0.01].mean()
    
    cfg["score"] = perfect_score.item()
    
    return cfg

Then call `results = batch_load(root, load_one_func)`

    '''
    subfolders = [ subfolder for subfolder in glob.glob(os.path.join(root, "*")) if subfolder_matcher.match(os.path.basename(subfolder)) ]
    
    pool = mp.Pool(num_process)
    num_folders = len(subfolders)
    chunksize = (num_folders + num_process - 1) // num_process

    print(f"Chunksize: {chunksize}")
    arguments = [ subfolder for subfolder in subfolders ]
    results = pool.imap_unordered(load_one_func, arguments, chunksize=chunksize)
    
    res = []
    for entry in tqdm.tqdm(results, total=num_folders):
        if entry is not None:
            res.append(entry)
    return res
