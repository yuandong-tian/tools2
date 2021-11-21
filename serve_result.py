import argparse
from collections import OrderedDict
from subprocess import check_output 
import time
from datetime import datetime
import os
import http.server
import socketserver
from queue import Empty
from multiprocessing import Process, Queue

from urllib.parse import urlparse, parse_qs

send_q = Queue()
recv_q = Queue()
last_reply = ""

class MyHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, *args):
        self.form_to_fill = '''
	<style>
	    pre {
		background: #f4f4f4;
		border: 1px solid #ddd;
		border-left: 3px solid #f36d33;
		color: #666;
		page-break-inside: avoid;
		font-family: monospace;
		font-size: 15px;
		line-height: 1.6;
		margin-bottom: 1.6em;
		max-width: 100%;
		overflow: auto;
		padding: 1em 1.5em;
		display: block;
		word-wrap: break-word;
	    }
	</style>
        <form method="POST">
            <p>Which run to check</p>
            <textarea name="content" id="content"></textarea>
            <input type="submit">
         </form>
        <form method="GET">
            <button>Refresh</button>
	</form>
        '''
        super(MyHTTPRequestHandler, self).__init__(*args)

    def _send_response(self):
        global last_reply
        while True:
            try:
                last_reply = recv_q.get_nowait()
                time.sleep(0.01)
            except Empty:
                break

        response = bytes(self.form_to_fill + last_reply, "utf-8") #create response

        self.send_response(200) #create header
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(response) #send response

    def do_GET(self):
        self._send_response()

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        data = str(self.rfile.read(length), "utf-8")
        send_q.put(data)

        self._send_response()


def run_http_server(port):
    Handler = MyHTTPRequestHandler

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"serving at port {port}")
        httpd.serve_forever()

parser = argparse.ArgumentParser()
parser.add_argument('--check_freq', type=int, default=60, help="check frequency (in sec)")
parser.add_argument('--match_file', type=str, default=None)
parser.add_argument('--key_stats', type=str, default="acc")
parser.add_argument('--descending', action="store_true")
parser.add_argument('--port', default=5000, type=int)

args = parser.parse_args()

abs_path = os.path.dirname(os.path.abspath(__file__))

proc = Process(target=run_http_server, args=(args.port,))
proc.start()

records = OrderedDict()

while True:
    try:
        request = send_q.get(timeout=args.check_freq)
        request = parse_qs(request)["content"][0]
        print(f"{datetime.now()}: Getting request..")
        state = 0
        records = OrderedDict()
        for line in request.split("\n"):
            line = line.strip(' ').strip("\n").strip("\r")
            if line == "":
                continue

            if state == 0:
                title = line
                state = 1
            elif state == 1:
                records[title] = line
                state = 0

        print(records)
    except Empty:
        print(f"{datetime.now()}: Reusing old request ... ")
    except:
        print(f"Error, restart...")
        continue
        
    # Then draw the records to a file and serve it. 
    f = open("_tmp.md", "w")
    f.write(f"# {datetime.now()} \n\n")
    for title, r in records.items():
        cmd = f"python {abs_path}/analyze.py --logdirs {r} --log_regexpr_json {args.match_file} --loader=log --num_process 1" 
        print(cmd)
        check_output(cmd, shell=True)
        cmd = f"python {abs_path}/stats.py --logdirs {r} --key_stats {args.key_stats} --topk_mean 1 --groups / "
        if args.descending:
            cmd += "--descending"
        print(cmd)
        output = check_output(cmd, shell=True).decode('utf-8')
        f.write(f"## {title}\n\n")
        f.write(f"```\n{output}\n```\n\n")
    f.close()
    check_output("pandoc _tmp.md --output _tmp.html", shell=True)
    with open("_tmp.html", "r") as f:
        reply = f.readlines()

    recv_q.put("".join(reply))

        
    
