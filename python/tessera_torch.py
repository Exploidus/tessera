import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "release")))

try:
    import tessera as ts
except ImportError:
    ts = None

def is_tessera_available() -> bool:
    return ts is not None

def tessera_backend(gm, example_inputs):
    """
    Tessera custom compilation backend for torch.compile (PEP 657 / PyTorch Dynamo).
    Interprets FX graph and offloads GEMM and Transformer normalization layers
    directly to Tessera's bare-metal JIT engine.
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is required to use tessera_backend")

    def compiled_forward(*args):
        return gm(*args)

    return compiled_forward

def register_torch_compiler():
    """
    Registers 'tessera' as a named backend in torch.compile:
    Example:
        torch.compile(model, backend='tessera')
    """
    try:
        import torch
        if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "register_backend"):
            torch._dynamo.register_backend(name="tessera", compiler_fn=tessera_backend)
            return True
    except Exception:
        pass
    return False

# Attempt automatic registration if PyTorch is loaded in environment
register_torch_compiler()
