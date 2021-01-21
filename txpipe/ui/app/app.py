#!/usr/bin/env python
from threading import Lock
from flask import Flask, render_template, session, request, \
    copy_current_request_context
from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect
import threading
import yaml
import ceci.utils
import sys
import queue
import contextlib
import os

TXPIPE_DIR = os.path.abspath("../../../")
sys.path.append(TXPIPE_DIR)
import txpipe

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode)
print("async mode", socketio.async_mode)

pipeline_thread = None
thread_lock = Lock()

@contextlib.contextmanager
def chdir(dirname=None):
    curdir = os.getcwd()
    try:
        if dirname is not None:
            print("Changing to dir", dirname)
            os.chdir(dirname)
        yield
    finally:
        if dirname is not None:
            print("Changing to dir", curdir)
            os.chdir(curdir)

print("Ident of main thread:", threading.get_ident())

class Session:
    def __init__(self):
        self.config = None
        self.pipeline = None
        self.queue = queue.SimpleQueue()
        self.logs = {}

    def load_pipeline(self, yml):
        self.config = yml
        self.pipeline = ceci.pipeline.MiniPipeline(yml["stages"], {"name": "mini"}, callback=self.callback, sleep=socketio.sleep)

    def get_graph(self):
        overall_inputs = self.config["inputs"]
        stages = self.pipeline.ordered_stages(overall_inputs)

        nodes = []
        edges = []

        sources = {}
        for stage in stages:
            nodes.append(stage.name)
            for tag in stage.output_tags():
                sources[tag] = stage.name

        for stage in stages:
            for tag in stage.input_tags():
                if tag in sources:
                    edges.append([sources[tag], stage.name])

        return nodes, edges

    def _run_task(self):
        inputs = self.config["inputs"]

        print("Override resume")
        self.config["resume"] = False

        site_config = self.config.get("site", {"name": "local"})

        # Override these with some of our own?
        run_config = {
            "output_dir": self.config["output_dir"],
            "log_dir": self.config["log_dir"],
            "resume": self.config["resume"],
        }

        stages_config = self.config["config"]

        paths = self.config.get("python_paths", [])
        if isinstance(paths, str):
            paths = paths.split()

        site_config["python_paths"] = paths
        ceci.sites.load({"name": "mini"}, [site_config])
        print("Ident of _run_task thread:", threading.get_ident())
        with chdir(TXPIPE_DIR):
            with ceci.utils.extra_paths(paths):
                self.pipeline.run(inputs, run_config, stages_config)

    def run(self):
        # create a minirunner
        # launch our pipeline

        with thread_lock:
            pipeline_thread = socketio.start_background_task(self._run_task)

    def request_log(self, name):
        stdout = self.logs.get(name)
        if stdout is None:
            return ""
        if not os.path.exists(stdout):
            return ""
        socketio.emit("read_log", {"stage":name, "data": open(stdout).read()})
        

    def callback(self, event, info):
        print(event)
        # update UI
        if event == "completed":
            socketio.emit("stage_complete", {"stage": info['job'].name})
        elif event == "fail":
            socketio.emit("stage_failed", {"stage": info['job'].name})
        elif event == "launch":
            name = info['job'].name
            self.logs[name] = info['stdout']
            socketio.emit("stage_launched", {"stage":name})


session = Session()


@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.event
def request_log(name):
    session.request_log(name)

@socketio.event
def log_python(msg):
    print(msg)

@socketio.event
def loaded_pipeline(d):
    yml = yaml.safe_load(d['yml'])
    session.load_pipeline(yml)
    nodes, edges = session.get_graph() 
    socketio.emit("parsed_pipeline", {"nodes":nodes, "edges":edges})

@socketio.event
def launch_pipeline(d):
    if not session.pipeline:
        print("No pipeline")
        socketio.emit("no_pipeline", {})
        return
    session.run()



# @socketio.event
# def disconnect_request():
#     @copy_current_request_context
#     def can_disconnect():
#         disconnect()

#     session['receive_count'] = session.get('receive_count', 0) + 1
#     # for this emit we use a callback function
#     # when the callback function is invoked we know that the message has been
#     # received and it is safe to disconnect
#     emit('my_response',
#          {'data': 'Disconnected!', 'count': session['receive_count']},
#          callback=can_disconnect)



# def connect():
#     print("Connected")
#     global thread
#     with thread_lock:
#         if thread is None:
#             thread = socketio.start_background_task(background_thread)
#     emit('my_response', {'data': 'Connected', 'count': 0})

# def background_thread():
#     """Example of how to send server generated events to clients."""
#     count = 0
#     while True:
#         socketio.sleep(10)
#         count += 1
#         socketio.emit('log_message',
#                       {'data': 'Server generated event', 'count': count})


# @socketio.on('disconnect')
# def test_disconnect():
#     print('Client disconnected', request.sid)


if __name__ == '__main__':
    print(socketio.run.__doc__)
    socketio.run(app, debug=True)


