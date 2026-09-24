"""Expose Compute Canada's `tokenspeed_triton` under the name `triton`.

mamba_ssm imports `triton` and `triton.language`; the Alliance wheelhouse ships the
same package renamed. The package self-references as `tokenspeed_triton` throughout,
so aliasing the public name is safe -- its own internals still resolve normally.
A meta-path finder is used rather than a fixed set of sys.modules entries so that
arbitrarily deep submodules (triton.language.core, triton.runtime.jit, ...) resolve too.
"""
import importlib, importlib.abc, importlib.machinery, sys

_PREFIX, _TARGET = "triton", "tokenspeed_triton"


class _TritonAlias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == _PREFIX or fullname.startswith(_PREFIX + "."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        mod = importlib.import_module(_TARGET + spec.name[len(_PREFIX):])
        sys.modules[spec.name] = mod
        return mod

    def exec_module(self, module):
        pass


def install():
    if not any(isinstance(f, _TritonAlias) for f in sys.meta_path):
        sys.meta_path.insert(0, _TritonAlias())


install()


# --- Mamba1 CUDA kernel stub -------------------------------------------------
# mamba_ssm/__init__.py eagerly imports `selective_scan_cuda`, the Mamba1 kernel.
# The Alliance wheel is built against an older libtorch, so that .so fails to load
# against torch 2.10 with an undefined c10::cuda symbol. EEGMamba configures
# `layer: "Mamba2"`, and mamba2.py never references selective_scan_cuda -- it runs
# through the Triton kernels in ops/triton/ssd_combined. So the import is satisfied
# with a stub whose every attribute RAISES: if a Mamba1 path is ever reached the run
# dies loudly instead of silently returning wrong features.
import types as _types


class _PoisonedModule(_types.ModuleType):
    def __getattr__(self, name):
        # Interpreter machinery (inspect, warnings, pickle) probes dunders on any
        # module object; raising on those breaks unrelated code paths. Only a real
        # kernel-function lookup is worth failing on.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError(
            f"selective_scan_cuda.{name} was called. This is the Mamba1 kernel, which "
            "is not loadable against this torch build and must not be used. EEGMamba "
            "should be running Mamba2 via Triton -- check ssm_cfg['layer']."
        )


def stub_mamba1_kernel():
    if "selective_scan_cuda" not in sys.modules:
        sys.modules["selective_scan_cuda"] = _PoisonedModule("selective_scan_cuda")


stub_mamba1_kernel()



# --- bypass mamba_ssm's package __init__ --------------------------------------
# mamba_ssm/__init__.py eagerly pulls in the whole language-model stack:
# selective_scan_cuda (Mamba1 kernel, ABI-broken here), MambaLMHeadModel, and via
# that, `transformers`. EEGMamba uses none of it -- it needs exactly four leaf
# modules: models.config_mamba, modules.mamba_simple, modules.mamba2, modules.mha.
#
# Rather than installing an LM dependency tree to satisfy imports that never run,
# register a bare package object whose __path__ points at the real directory. Python
# then resolves `mamba_ssm.modules.mamba2` straight to the file on disk without ever
# executing the package __init__. The submodules themselves import only what they use.
import os as _os


def bypass_mamba_init(deps_dir=None):
    if "mamba_ssm" in sys.modules and getattr(sys.modules["mamba_ssm"], "_eegmamba_shim", False):
        return
    deps_dir = deps_dir or _os.path.dirname(_os.path.abspath(__file__))
    pkg_dir = _os.path.join(deps_dir, "mamba_ssm")
    if not _os.path.isdir(pkg_dir):
        raise RuntimeError(f"mamba_ssm not found at {pkg_dir}")
    pkg = _types.ModuleType("mamba_ssm")
    pkg.__path__ = [pkg_dir]
    pkg.__package__ = "mamba_ssm"
    pkg._eegmamba_shim = True
    sys.modules["mamba_ssm"] = pkg


bypass_mamba_init()


# --- stub the unused GenerationMixin -----------------------------------------
# EEGMamba's vendored modules/mixer_seq_simple.py carries
#   from mamba_ssm.utils.generation import GenerationMixin
# left over from copying mamba-ssm's file. That module imports `transformers`, but
# GenerationMixin appears nowhere else in the file -- its only class, MixerModel,
# does not inherit from it (verified by grep: one hit, the import line). Supplying an
# empty placeholder satisfies the dead import without pulling in a language-model
# dependency tree. If a real generation path is ever taken, instantiating this raises.
def stub_generation_mixin():
    name = "mamba_ssm.utils.generation"
    if name in sys.modules:
        return
    mod = _types.ModuleType(name)

    class GenerationMixin:  # noqa: D401 - placeholder for a provably unused import
        def __init__(self, *a, **k):
            raise RuntimeError(
                "mamba_ssm GenerationMixin was instantiated. It is stubbed because "
                "EEGMamba never uses it; a real use needs `transformers` installed."
            )

    mod.GenerationMixin = GenerationMixin
    sys.modules[name] = mod


stub_generation_mixin()


# --- stub the unused HF checkpoint helpers ------------------------------------
# Line 21 of the vendored mixer_seq_simple.py carries
#   from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
# which reaches `transformers.utils`. Like GenerationMixin above, neither name is
# referenced anywhere else in that file (grep: one hit, the import). EEGMamba loads
# its checkpoint with a plain torch.load, so these are dead. Calling either raises.
def stub_hf_helpers():
    name = "mamba_ssm.utils.hf"
    if name in sys.modules:
        return
    mod = _types.ModuleType(name)

    def _dead(fn_name):
        def _raise(*a, **k):
            raise RuntimeError(
                f"mamba_ssm.utils.hf.{fn_name} was called. It is stubbed because "
                "EEGMamba loads its checkpoint via torch.load; a real use needs "
                "`transformers` installed."
            )
        return _raise

    mod.load_config_hf = _dead("load_config_hf")
    mod.load_state_dict_hf = _dead("load_state_dict_hf")
    sys.modules[name] = mod


stub_hf_helpers()
