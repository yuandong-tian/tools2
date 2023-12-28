import datetime
import re
import argparse
import os
import glob
import sys
import time
import json
import zmq

from monitor_utils import *

def simple_encoder(a):
    return a.replace("\"", "@@@").replace(" ", "___")

def simple_decoder(a):
    return a.replace("@@@", "\"").replace("___", " ")


class JobTracker:
    def __init__(self, dry_run : bool, initial_job_ids=None, fallback_dir="./", user_id="yuandong"):
        self.dry_run = dry_run
        self.finished_job_ids = dict()
        self.fallback_dir = fallback_dir
        self.user_id = user_id

        if initial_job_ids is None or not isinstance(initial_job_ids, list):
            print("No job_ids specified. Load from sacct ...")
            job_stats = get_sacct(self.user_id)
            self.job_ids = set([ jobid for jobid, info in job_stats.items() if info["state"] in ["RUNNING", "PENDING"] ])
        else:
            print("Loading job_ids ...")
            self.job_ids = set(initial_job_ids)

    def _set_job_status(self, jobid, state):
        if jobid not in self.job_ids:
            return False

        self.job_ids.remove(jobid)
        self.finished_job_ids[jobid] = state
        return True

    def _cancel_jobs(self, jobs_to_be_cancelled):
        for jobid in jobs_to_be_cancelled:
            if self._set_job_status(jobid, "CANCELLED"):
                run_cmd(f"scancel {jobid}", self.dry_run, async_cmd=False)
            else:
                print(f"Jobid {jobid} not in the current watch list")

    def _add_jobs(self, jobs_to_be_added):
        self.job_ids = self.job_ids.union(set(jobs_to_be_added))

    def start_check(self):
        self.job_stats = get_sacct(self.user_id)
        print(f"[{datetime.datetime.now()}]: #jobs: {len(self.job_ids)}. jobs: {vis_seq(self.job_ids)}")
    
    def add_rerun_jobs(self):
        # If there is any jobs with "rerun" comment, add them to monitor
        new_jobs = []
        for jobid, info in self.job_stats.items():
            if jobid in self.job_ids or jobid in self.finished_job_ids:
                continue
            comment = info["comment"]
            if comment is not None and comment.startswith("rerun:") and info["state"] in ["RUNNING", "PENDING"]:
                new_jobs.append(jobid)

        self._add_jobs(new_jobs)

    def _match_jobs(self, regex):
        matcher = re.compile(regex)

        jobs_found = set()
        dup_job_ids = set(self.job_ids)
        for jobid in dup_job_ids:
            submit_line = self.job_stats[jobid]["submit_line"] 
            m = matcher.search(submit_line)
            if m:
                print(f"Find match: {jobid}: {submit_line}") 
                jobs_found.add(jobid)
        
        print(f"Total #job matched: {len(jobs_found)}")
        return jobs_found

    def _match_wandb_urls(self, urls):
        urls = set(urls.split(","))

        jobs_found = set()
        dup_job_ids = set(self.job_ids)
        for jobid in dup_job_ids:
            stderr_file = self.job_stats[jobid]["stderr_file"] 
            if stderr_file is None:
                continue

            submit_line = self.job_stats[jobid]["submit_line"] 
            url = get_wandb_url(stderr_file)
            if url in urls:
                print(f"Find match: {jobid}: {url}, {submit_line}") 
                jobs_found.add(jobid)
        
        print(f"Total #job matched: {len(jobs_found)}")
        return jobs_found


    def run_zmq_cmd(self, message):
        # Deal with message
        items = message.split(" ")
        cmd_executed = False

        if len(items) == 1:
            cmd = items[0]
            if cmd == "refresh_track":
                new_job_ids = [ jobid for jobid, info in self.job_stats.items() if info["state"] in ["RUNNING", "PENDING"] ]
                self._add_jobs(new_job_ids)
                print(f"Refresh tracker")
                cmd_executed = True

        elif len(items) == 2:
            cmd, params = items[0], items[1]
            if cmd == "cancel_ids":
                self._cancel_jobs([ int(i) for i in params.split(",") ])
                cmd_executed = True

            elif cmd == "cancel_regex":
                self._cancel_jobs(self._match_jobs(params))
                cmd_executed = True
            elif cmd == "cancel_wandb_urls":
                self._cancel_jobs(self._match_wandb_urls(params))
                cmd_executed = True

            elif cmd == "find_regex":
                self._match_jobs(params)
                cmd_executed = True

            elif cmd == "find_wandb_urls":
                self._match_wandb_urls(params)
                cmd_executed = True

        if not cmd_executed:
            print(f"Unknown command = {message}, skipping ...")

    def update(self): 
        job_stats = self.job_stats

        # for each job, check status
        dup_job_ids = set(self.job_ids)
        for jobid in dup_job_ids:
            state = job_stats[jobid]["state"]

            if state == "COMPLETED":
                # remove
                print(f"{jobid} has {state}!")
                self._set_job_status(jobid, state)

            elif state == "FAILED" or state == "CANCELLED":
                # restart the job
                print(f"{jobid} has status {state}! restarting")

                # all info there
                submit_line = job_stats[jobid]["submit_line"]
                comment = job_stats[jobid]["comment"]
                last_stderr_file = job_stats[jobid]["stderr_file"]
                last_stdout_file = job_stats[jobid]["stdout_file"]

                if comment is not None and comment.startswith("rerun:"):
                    # if comment is "rerun", use the path there. 
                    comment_data = json.loads(simple_decoder(comment[6:]))
                    std_base = comment_data["std_base"] 
                    failed = comment_data["fail_count"] + 1
                else:
                    # first failed
                    if last_stderr_file is None:
                        print(f"stderr file is none! {jobid}: {submit_line}")
                        # Fallback path
                        std_base = os.path.join(self.fallback_dir, str(jobid))
                    else:
                        std_base = os.path.join(os.path.dirname(last_stderr_file), str(jobid))
                    failed = 1

                stderr_path = std_base + f"_{failed}.err"
                stdout_path = std_base + f"_{failed}.out"

                new_comment = "rerun:" + simple_encoder(json.dumps(dict(std_base=std_base, fail_count=failed)))

                # remove previous comment if there is any
                submit_line = comment_matcher.sub("", submit_line)

                # replace stderr and stdout, and add new comment
                submit_line = stderr_matcher.sub("", submit_line)
                submit_line = stdout_matcher.sub("", submit_line)
                submit_line = submit_line + " 2>&1 1> /dev/null" 

                start_cmd, remaining = submit_line.split(" ", 1)
                submit_line = start_cmd + " -e " + stderr_path + " -o " + stdout_path + f" --comment='{new_comment}' " + remaining

                # print previous errors.. 
                all_errs = []
                try:
                    with open(last_stderr_file, "r") as f:
                        for line in f:
                            if "Error" in line:
                                all_errs.append(line)
                except:
                    pass

                if len(all_errs) > 0:
                    print(f"Possible reason: {all_errs[-3:]}")
                else:
                    print("Cannot find possible reason..")

                print(f"Resubmitting job with commit [{get_git_hash()}]: {submit_line}")
                # cmd = f'nohup bash -c "{submit_line}" &>> /dev/null < /dev/null &'
                run_cmd(submit_line, args.dry_run)

                self._set_job_status(jobid, state)


parser = argparse.ArgumentParser()
parser.add_argument('--job_ids', type=str, default=None, help="If none, then track all running and pending jobs") 
parser.add_argument('--dry_run', action="store_true")
parser.add_argument('--fallback_dir', type=str, default="./")
parser.add_argument('--user_id', type=str, default=os.environ["USER"])
parser.add_argument('--port', type=int, default=1579)

args = parser.parse_args()

# go through all stdout list, extract their first line as a command line, and resume if necessary. 
initial_job_ids = load_seq(args.job_ids) if args.job_ids is not None else None
tracker = JobTracker(args.dry_run, initial_job_ids=initial_job_ids, fallback_dir=args.fallback_dir, user_id=args.user_id)

# Open zmq port to collect request.
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind(f"tcp://*:{args.port}")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN) # POLLIN for recv, POLLOUT for send

# Monitor jobs
while True:
    tracker.start_check()
    tracker.add_rerun_jobs()

    # check whether there is a message coming from any client. 
    evts = poller.poll(30000)
    if len(evts) > 0:
        message = evts[0][0].recv_string()
        tracker.run_zmq_cmd(message)

    tracker.update()


