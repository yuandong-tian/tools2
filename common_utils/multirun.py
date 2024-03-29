from subprocess import check_output 
from omegaconf import OmegaConf
from datetime import timedelta
from collections import defaultdict
import importlib
import time
import yaml
import json
import re
import glob
import os
import sys
from copy import deepcopy

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
    return OmegaConf.to_yaml(args, resolve=True)

gJobStartLine = "===*** Job start ***==="

def print_info(args):
    return f'''
{gJobStartLine}
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
        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path + ".py"):
            # second choice, check os.getcwd()
            curr_path = os.getcwd()
            full_path = os.path.join(curr_path, filename)
            assert os.path.exists(full_path + ".py"), f"{filename} cannot be found in either {path} or {curr_path}"
        return full_path 

    @classmethod
    def load_full_cfg(cls, subfolder):
        return yaml.safe_load(open(os.path.join(subfolder, ".hydra/config.yaml"), "r"))

    @classmethod
    def load_omega_conf(cls, subfolder):
        return OmegaConf.load(os.path.join(subfolder, ".hydra/config.yaml"))

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
            if not load_submitit_log:
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

        return log_file

    @classmethod
    def get_modified_since(cls, log_file):
        sec_diff = time.time() - os.path.getmtime(log_file)
        td = timedelta(seconds=sec_diff)
        return td

    @classmethod
    def get_logfile_longest_section(cls, log_file):
        def detect_job_preempt(line):
            if line.find(gJobStartLine) >= 0:
                return True

            if line.find("[submitit][WARNING]") >= 0 and line.find("Bypassing") < 0:
                return line.find("SIGTERM") >= 0 or line.find("SIGUSR1") >= 0 or line.find("SIGCONT") >= 0
            else:
                return False

        with open(log_file, "r") as f:
            lines = f.readlines() 

        sections = []
        last_start = 0
        for i, line in enumerate(lines):
            # if things are preempted, restart the record and keep only the longest one.
            if detect_job_preempt(line):
                sections.append((last_start, i))
                last_start = i

        # including the last line
        sections.append((last_start, i + 1))
        # pick the longest section.
        longest_start, longest_end = sorted(sections, key=lambda x: x[1] - x[0])[-1]
        return lines[longest_start:longest_end]

    @classmethod
    def load_regex(cls, config, lines, regex_list):
        # hack = os.path.basename(subfolder) in ["74"]
        entry = defaultdict(list)
        has_data = False
        for line in lines:
            for d in regex_list:
                m = d["match"].search(line)
                if m:
                    has_data = True
                    for key_act, val_act in d["action"]:
                        entry[key_act].append(eval(val_act))

        return entry if has_data else None

    @classmethod
    def load_df(cls, config, lines, df_matcher):
        for line in lines:
            m = df_matcher["match"].search(line)
            if m:
                return df_matcher["action"](config, m)

        return []

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
        return entry

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
        sys.path.append(os.path.abspath(os.path.dirname(main_file_checkresult)))
        spec.loader.exec_module(mdl)

        return mdl
    
