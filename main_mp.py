import argparse
import subprocess
import multiprocessing as mp

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

# 参数分别为rank和总进程数
def run(rank: int, number: int):
    p = subprocess.Popen(f"GLOO_SOCKET_IFNAME=lo torchrun --nproc_per_node=1 \
                            --nnodes={number} --node_rank={rank} --rdzv_id=456 --rdzv_endpoint=127.0.0.1:3214 main.py --gpu -1",
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    stdout_queue = mp.Queue()
    stderr_queue = mp.Queue()

    stdout_thread = mp.Process(target=enqueue_output, args=(p.stdout, stdout_queue))
    stderr_thread = mp.Process(target=enqueue_output, args=(p.stderr, stderr_queue))

    stdout_thread.daemon = True
    stderr_thread.daemon = True

    stdout_thread.start()
    stderr_thread.start()

    while p.poll() is None:
        while not stdout_queue.empty():
            line = stdout_queue.get()
            print(f"[Client {rank}] {line.decode('utf-8').strip()}")
        
        while not stderr_queue.empty():
            err = stderr_queue.get()
            print(f"[Client {rank}] {err.decode('utf-8').strip()}")

    stdout_thread.join()
    stderr_thread.join()

    if p.returncode == 0:
        print(f"[Client {rank}] Command completed successfully.")
    else:
        print(f"[Client {rank}] Command failed with return code {p.returncode}")

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('-n', '--process_threshold', type=str, default=4, dest='process_threshold')
args = parser.parse_args()
process_threshold_value = int(args.process_threshold)
server_process = mp.Process(target=run, args=(0, process_threshold_value,))
server_process.start()

if process_threshold_value > 1:
    clients_process = [mp.Process(target=run, args=(i, process_threshold_value,))
                       for i in range(1, process_threshold_value)]
    for c in clients_process:
        c.start()

server_process.join()
