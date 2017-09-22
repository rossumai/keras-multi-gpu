from __future__ import print_function
import datetime
import subprocess
import sys

def generate_id():
    return datetime.datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")

def arguments_from_dict(arguments):
    """
    dict to list of CLI options
    """
    args_list = []
    for key in sorted(arguments):
        args_list.append("--" + key)
        args_list.append(str(arguments[key]))
    return args_list

def run_benchmark(id, arguments):
    log_file = '%s.log' % id
    command = ' '.join(["python", "tf_cnn_benchmarks.py"] + arguments_from_dict(arguments))
    command += " 2>&1 |tee " + log_file
    print('Running experiment:', id)
    print(command)
    subprocess.call(command, shell=True)

run_benchmark(generate_id(), {})
