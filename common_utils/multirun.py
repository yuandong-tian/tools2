from subprocess import check_output 
from omegaconf import OmegaConf
from datetime import timedelta
from collections import defaultdict
import importlib
import time
import yaml
import re
import glob
import os
import sys

def get_git_hash():
    try:
        return check_output("git -C ./ log --pretty=format:'%H' -n 1", shell=True).decode('utf-8')
    except:
        return ""

def get_git_diffs():
    try:
        active_diff = check_output("git diff", shell=True).decode('utf-8')
        staged_diff = check_output("git diff --cached", shell=True).decode('utf-8')
        return active_diff + "\n" + staged_diff
    except:
        return ""

def get_all_files(root, file_extension):
    files = []
    for folder, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(file_extension):
                files.append(os.path.join(folder, f))
    return files

def pretty_print_cmd(args):
    s = ""
    for i, a in enumerate(args):
        if i > 0:
            s += "  "
        s += a + " \\\n"
    return s

def pretty_print_args(args):
    return OmegaConf.to_yaml(args)

def print_info(args):
    return f'''
Command line:
    
{pretty_print_cmd(sys.argv)}
    
Working dir: {os.getcwd()}
{get_git_hash()}
{get_git_diffs()}
{pretty_print_args(args)}
'''
    
class MultiRunUtil:
    filename_cfg_matcher = re.compile("[^=]+=[^=_]+")

    @classmethod
    def get_main_file(cls, subfolder):
        config = yaml.safe_load(open(os.path.join(subfolder, ".hydra/hydra.yaml"), "r"))
        filename = config["hydra"]["job"]["name"] 
        path = config["hydra"]["runtime"]["cwd"]
        return os.path.join(path, filename)

    @classmethod
    def load_full_cfg(cls, subfolder):
        return yaml.safe_load(open(os.path.join(subfolder, ".hydra/config.yaml"), "r"))

    @classmethod
    def load_cfg(cls, subfolder):
        ''' Return list [ "key=value" ] '''
        if not os.path.isdir(subfolder):
            # If it is not a dir, then just read from its filename.
            s = os.path.splitext(os.path.basename(subfolder))[0]
            cfg = [ m for m in cls.filename_cfg_matcher.findall(s) ]
            cfg = [ v.strip("_") for v in cfg ]
            return cfg

        if os.path.exists(os.path.join(subfolder, ".hydra")):
            return yaml.safe_load(open(os.path.join(subfolder, ".hydra/overrides.yaml"), "r"))

        if os.path.exists(os.path.join(subfolder, "multirun.yaml")):
            multirun_info = yaml.safe_load(open(os.path.join(subfolder, "multirun.yaml")))
            return multirun_info["hydra"]["overrides"]["task"]

        # Last resort.. read *.log file directly to get the parameters.
        num_files = list(glob.glob(os.path.join(subfolder, "*.log")))
        assert len(num_files) > 0
        log_file = num_files[0]
        text = None
        for line in open(log_file):
            if text is None:
                if not line.startswith("[") and not line.startswith(" "):
                    text = line
            else:
                if line.startswith("["):
                    break
                text += line

        assert text is not None
        overrides = yaml.safe_load(text)
        # From dict to list of key=value
        return [ f"{k}={v}" for k, v in overrides.items() if not isinstance(v, (dict, list)) ]
    
    @classmethod
    def get_log_file(cls, subfolder, load_submitit_log=False):
        if os.path.isdir(subfolder):
            if load_submitit_log:
                all_log_files = list(glob.glob(os.path.join(subfolder, "*.log")))
            else:
                parent_folder = os.path.dirname(subfolder)
                job_name = os.path.basename(subfolder)
                all_log_files = list(glob.glob(os.path.join(parent_folder, f".submitit/*_{job_name}/*.out")))

            if len(all_log_files) == 0:
                return None
            # First log file.
            log_file = all_log_files[0]
        else:
            log_file = subfolder

        sec_diff = time.time() - os.path.getmtime(log_file)
        td = timedelta(seconds=sec_diff)

        entry = defaultdict(list)
        entry["folder"] = subfolder
        entry["modified_since"] = str(td)

        return entry, log_file


    @classmethod
    def load_regex(cls, subfolder, regex_list, load_submitit_log=False):
        entry, log_file = cls.get_log_file(subfolder, load_submitit_log=load_submitit_log)

        has_one = False
        with open(log_file, "r") as f:
            for line in f:
                for d in regex_list:
                    m = d["match"].search(line)
                    if not m:
                        continue

                    for key_act, val_act in d["action"]:
                        entry[key_act].append(eval(val_act))
                    has_one = True

        if has_one:
            return [ dict(entry) ]
        else:
            return None

    @classmethod
    def load_inline_json(cls, subfolder, json_prefix="json_stats: ", load_submitit_log=False):
        entry, log_file = cls.get_log_file(subfolder, load_submitit_log=load_submitit_log)

        cnt = 0
        with open(log_file, "r") as f:
            for line in f:
                index = line.find(json_prefix)

                if index < 0:
                    continue

                this_entry = json.loads(line[index + len(json_prefix):])

                for k, v in this_entry.items():
                    entry[k].append(v)

                cnt += 1

        if cnt > 0:
            return [ dict(entry) ]
        else:
            return None

    @classmethod
    def load_tensorboard(cls, subfolder, tb_folder="stats.tb", tb_choice="largest"):
        event_files = [ glob.glob(os.path.join(subfolder, tb_folder, "*")) ]
        if tb_choice == "largest":
            # Use the largest event_file.
            files = [ (os.path.getsize(event_file), event_file) for event_file in event_files ] 
            files = sorted(files, key=lambda x: -x[0])[:1]
        elif tb_choice == "earliest":
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: x[0])[:1]
        elif tb_choice == "latest":
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: -x[0])[:1]
        elif tb_choice == "all":
            # All files, earliest first
            files = [ (os.path.getmtime(event_file), event_file) for event_file in event_files ]
            files = sorted(files, key=lambda x: x[0])
        else:
            raise RuntimeError(f"Unknown tb_choice: {tb_choice}")

        if len(files) == 0:
            return None

        entry = dict(folder=subfolder)
        for _, event_file in files:
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()

            for key_name in ea.Tags()["scalars"]:
                l = [ s.value for s in ea.Scalars(key_name) ]
                if key_name not in entry: 
                    entry[key_name] = []
                entry[key_name] += l

        # Format: 
        # List[Dict[str, List[value]]]: number of trials * (key, a list of values)
        return [entry]

    @classmethod
    def load_checkpoint(cls, subfolder, summary="summary.pth"):
        # [TODO] Hardcoded path.
        # sys.path.append("/private/home/yuandong/forked/luckmatters/catalyst")
        summary_file = os.path.join(subfolder, summary)

        stats = None
        for i in range(10):
            try:
                if os.path.exists(summary_file):
                    stats = torch.load(summary_file)
                    if hasattr(stats, "stats"):
                        stats = stats.stats
                break
            except Exception as e:
                time.sleep(2)

        if stats is None:
            # print(subfolder)
            # print(e)
            return None

        if not isinstance(stats, dict):
            stats = [stats]
        else:
            stats = stats.values()

        entries = []
        for one_stat in stats:
            if one_stat is not None:
                entry = dict(folder=subfolder)
                entry.update(listDict2DictList(one_stat))
                entries.append(to_cpu(entry))

        return entries

    @classmethod
    def load_check_module(cls, subfolder, filename=None):
        if filename is None:
            main_file = cls.get_main_file(subfolder)
            main_file_checkresult = main_file + "_checkresult.py"
            if not os.path.exists(main_file_checkresult):
                main_file_checkresult = main_file + ".py"
        else:
            main_file_checkresult = filename

        spec = importlib.util.spec_from_file_location("", main_file_checkresult)
        mdl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mdl)

        return mdl
    