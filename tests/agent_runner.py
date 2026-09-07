"""Load the task extension from PyTorch's distributed.py without a compiled Torch build.

The runner supplies only the minimal module objects required for the pre-existing imports in distributed.py.
The agent's added signature function receives only public query objects and never sees verifier ground truth.
"""
import importlib.util
import json
import os
import sys
import traceback
import types
from pathlib import Path

TARGET = Path("/app/torch/utils/data/distributed.py")
_BLOCKED_PREFIXES = ("os.exec", "os.spawn", "os.posix_spawn", "subprocess.", "ctypes.")
_BLOCKED_EVENTS = frozenset({"os.system", "os.startfile", "os.fork", "os.forkpty", "pty.spawn"})


def _audit(event, args):
    if event in _BLOCKED_EVENTS or event.startswith(_BLOCKED_PREFIXES):
        raise RuntimeError(f"blocked external execution: {event}")


sys.addaudithook(_audit)


def _install_torch_stubs():
    torch = types.ModuleType("torch")
    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.get_world_size = lambda: 1
    distributed.get_rank = lambda: 0
    torch.distributed = distributed

    utils = types.ModuleType("torch.utils")
    data = types.ModuleType("torch.utils.data")
    dataset = types.ModuleType("torch.utils.data.dataset")
    sampler = types.ModuleType("torch.utils.data.sampler")

    class Dataset:
        pass

    class Sampler:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    dataset.Dataset = Dataset
    sampler.Sampler = Sampler
    sys.modules.update({
        "torch": torch,
        "torch.distributed": distributed,
        "torch.utils": utils,
        "torch.utils.data": data,
        "torch.utils.data.dataset": dataset,
        "torch.utils.data.sampler": sampler,
    })


def run(payload):
    _install_torch_stubs()
    spec = importlib.util.spec_from_file_location("task_distributed_sampler", TARGET)
    if spec is None or spec.loader is None:
        return {"error": f"could not load {TARGET}"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "sharding_signature", None)
    if not callable(fn):
        return {"error": "distributed.py does not expose callable sharding_signature(query)"}
    answers = []
    for query in payload["queries"]:
        value = fn(query)
        answers.append({"t": type(value).__name__, "v": value if type(value) is int else None})
    return {"answers": answers}


def main():
    try:
        result = run(json.load(sys.stdin))
    except BaseException:
        result = {"error": traceback.format_exc()[-2000:]}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
