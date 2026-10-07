"""Wrapper that runs SimCLR training with output to file, handling SIGPIPE gracefully."""
import sys, os, signal

# Ignore SIGPIPE so broken pipes don't crash us
signal.signal(signal.SIGPIPE, signal.SIG_IGN)

# Redirect stdout/stderr to log file
log = open('/home/anik-server/txmorph-wsi/logs/simclr_train.log', 'w', buffering=1)
sys.stdout = log
sys.stderr = log

# Also print to original stderr for the SSH session
orig_stderr = os.fdopen(os.dup(2), 'w', buffering=1)

os.chdir('/home/anik-server/txmorph-wsi')
sys.path.insert(0, 'src')

from txmorph.utils.config import load_config
from txmorph.utils.seed import seed_everything

cfg = load_config(overrides=['simclr.batch_size=384'])
seed_everything(cfg.get('seed', 0))

from pathlib import Path
from txmorph.training.simclr import train_simclr

ec = cfg.get('simclr', cfg.get('encoder', {}))
ckpt_path = str(Path(cfg.paths.ckpt) / 'simclr.pt')

# Print to both log and SSH
def dual_print(msg):
    print(msg, flush=True)
    try:
        orig_stderr.write(msg + '\n')
        orig_stderr.flush()
    except:
        pass

dual_print(f'[wrapper] Starting training, ckpt={ckpt_path}, batch_size=384')

train_simclr(
    tiles_dir=cfg.paths.tiles, ckpt_path=ckpt_path,
    epochs=ec.get('epochs', 100), batch_size=384,
    lr=ec.get('lr', 1e-3), temperature=ec.get('temperature', 0.07),
    proj_dim=ec.get('proj_dim', 128), device=cfg.get('device', 'cuda'),
    num_workers=ec.get('num_workers', cfg.get('num_workers', 4)),
    seed=cfg.get('seed', 0),
)
dual_print('[wrapper] Training complete!')
