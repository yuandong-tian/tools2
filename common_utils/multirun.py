from subprocess import check_output 
import yaml
import re
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

filename_cfg_matcher = re.compile("[^=]+=[^=_]+")

class MultiRunUtil:
    def load_cfg(self, subfolder):
        ''' Return list [ "key=value" ] '''
        if not os.path.isdir(subfolder):
            # If it is not a dir, then just read from its filename.
            s = os.path.splitext(os.path.basename(subfolder))[0]
            cfg = [ m for m in filename_cfg_matcher.findall(s) ]
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

    