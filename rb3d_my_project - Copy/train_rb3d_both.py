#!/usr/bin/env python3
"""
train_rb3d_both.py
================================================================================
Train the CONDITIONED and UNCONDITIONED RB3D models in parallel, one per GPU.

Because they are two SEPARATE models, this is embarrassingly parallel: each
runs as an independent process pinned to its own GPU (conditioned -> cuda:0,
unconditioned -> cuda:1). There is NO gradient sync, NO data sharding, NO
inter-GPU traffic -- each trains at full single-GPU speed, so both finish in
the wall-clock time of one. (Splitting a SINGLE model across two GPUs would
instead pay sync/transfer overhead for little gain; that is not what this does.)

Each worker writes its own checkpoints (best + _last) and resumes independently,
so a session restart just reruns this and both pick up where they left off.

Logs from both workers are interleaved with a [cond]/[uncond] prefix and also
saved to <out>/train_cond.log and <out>/train_uncond.log.

USAGE
    python train_rb3d_both.py --bank ./datasets/rb3d_multisolution/splits/train_bank.pt
    python train_rb3d_both.py --bank train_bank.pt --iters 30000 --batch 12
    # pass any train_* flag through; it goes to BOTH workers:
    python train_rb3d_both.py --bank train_bank.pt --hidden 48 --batch 8
If only 1 GPU is visible, the two models run SEQUENTIALLY on it (still correct,
just not parallel) -- a message says so.
"""

import argparse
import os
import subprocess
import sys
import threading
import time


def stream(proc, tag, logfile):
    with open(logfile, 'w') as lf:
        for line in proc.stdout:
            sys.stdout.write(f'[{tag}] {line}')
            sys.stdout.flush()
            lf.write(line); lf.flush()


def launch(script, device, out, passthrough):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        script),
           '--device', device, '--out', out, *passthrough]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=('/kaggle/working'
                                      if os.path.isdir('/kaggle/working') else '.'))
    ap.add_argument('--gpus', default='0,1',
                    help='comma list: which GPUs for [cond,uncond] (default 0,1)')
    ap.add_argument('--sequential', action='store_true',
                    help='force one-after-the-other (each model still uses its '
                         'own GPU, but only one runs at a time -- needed when '
                         'the dataset is too big to hold TWO copies in RAM)')
    ap.add_argument('--parallel', action='store_true',
                    help='force both-at-once even if the RAM estimate says it '
                         'will not fit (use only if you know the box has RAM)')
    args, passthrough = ap.parse_known_args()   # everything else -> both workers
    os.makedirs(args.out, exist_ok=True)

    try:
        import torch
        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    gpus = args.gpus.split(',')

    jobs = [('train_rb3d_pcfm_conditioned.py', 'cond'),
            ('train_rb3d_pcfm_unconditioned.py', 'uncond')]

    # --- RAM guard: each RB3DData holds ~4*Nx*Ny*Nz*N*4 bytes; two parallel
    #     trainers need two copies. On a ~13 GB Kaggle box the 128x64x49 train
    #     bank (~8.6 GB each) does NOT fit twice, so parallel training is
    #     OOM-killed (exit -9). Estimate and fall back to sequential.
    def _bank_ram_gb():
        for tok in passthrough:
            pass
        bank = None
        for i, tok in enumerate(passthrough):
            if tok == '--bank' and i + 1 < len(passthrough):
                bank = passthrough[i + 1]
        if not bank or not os.path.exists(bank):
            return None
        # file size is a good proxy for the in-RAM tensor size (float32 fields)
        return os.path.getsize(bank) / 1e9

    def _avail_ram_gb():
        try:
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemAvailable'):
                        return int(line.split()[1]) / 1e6
        except Exception:
            return None
        return None

    run_parallel = (n_gpu >= 2) and not args.sequential
    if run_parallel and not args.parallel:
        need = _bank_ram_gb()
        avail = _avail_ram_gb()
        if need is not None and avail is not None:
            # The killer is not the STEADY footprint (~need+1.5 GB each, which
            # usually fits twice) but the TRANSIENT peak of torch.load, which
            # briefly holds the deserialized pickle buffer AND the materialized
            # tensors (~2x the file) -- and if both trainers load at the same
            # instant those peaks STACK. Budget for two concurrent load peaks.
            load_peak = 2.0 * need + 1.5
            if 2 * load_peak > avail:
                print(f'[both] RAM guard: two concurrent dataset loads peak at '
                      f'~{2*load_peak:.1f} GB > {avail:.1f} GB available -> '
                      f'running SEQUENTIALLY (one model at a time, each on its '
                      f'own GPU). Override with --parallel if you have more RAM.',
                      flush=True)
                run_parallel = False

    if not run_parallel:
        if n_gpu < 2 and not args.sequential:
            print(f'[both] only {n_gpu} GPU(s) visible -> sequential', flush=True)
        for i, (script, tag) in enumerate(jobs):
            # each model still gets a full GPU; they just don't overlap in time
            dev = f'cuda:{gpus[i % max(n_gpu,1)]}' if n_gpu >= 1 else 'cpu'
            print(f'[both] === {tag} on {dev} (sequential) ===', flush=True)
            p = launch(script, dev, args.out, passthrough)
            stream(p, tag, os.path.join(args.out, f'train_{tag}.log'))
            if p.wait() != 0:
                sys.exit(f'[both] {tag} failed')
        print('[both] both models done (sequential).')
        return

    # parallel: one model per GPU
    procs, threads = [], []
    for (script, tag), gpu in zip(jobs, gpus):
        dev = f'cuda:{gpu}'
        print(f'[both] launching {tag} on {dev}', flush=True)
        p = launch(script, dev, args.out, passthrough)
        t = threading.Thread(target=stream, args=(
            p, tag, os.path.join(args.out, f'train_{tag}.log')), daemon=True)
        t.start()
        procs.append((tag, p)); threads.append(t)
        time.sleep(2)   # stagger cuDNN autotune so the two don't collide

    rc = 0
    for tag, p in procs:
        if p.wait() != 0:
            print(f'[both] {tag} exited with code {p.returncode}', flush=True)
            rc = 1
    for t in threads:
        t.join(timeout=5)
    print('[both] both models done.' if rc == 0 else '[both] a worker failed.',
          flush=True)
    sys.exit(rc)


if __name__ == '__main__':
    main()