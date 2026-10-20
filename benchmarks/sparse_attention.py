"""Opt-in sparse attention runtime timing (scope_version=2).

START precedes the native attention input communication stage; END follows
its last result return/aggregation and existing stream dependencies. Routing,
sparse execution, KV/metadata transfers, coarse-result folding and exposed
waits inside that interval are included. Input Norm/modulation, projections
outside the communication interval, output projection, residual, cross-attn
and FFN are excluded. Single GPU uses the corresponding runtime interval.

NABLA DTensor input exchanges are interleaved with Q/K/V projections: the
continuous first-exchange -> output-redistribution interval necessarily also
contains K/V projection and QK Norm/RoPE. This exception is explicit in its
manifest; single-GPU NABLA has no exchange and measures the attention call.

AST insertion retains native operations and validates equality after erasing
event calls. Existing decorators, hooks and compiled helpers are retained.
DTensor communication hook boundaries add explicit compiler breaks for event
recording; no collective or per-layer synchronization is added. Whole-block
compile/graphs are rejected except Omni H3 with its native recursive eager
attention boundary and Omni Wan with explicit eager event breaks. Events are preallocated/primed outside requests;
results are read after the caller's post-request synchronization. Side-stream
work is covered by its existing consumer dependency, not double-counted.

Version 1 (Norm -> residual) results are legacy and must never be relabeled or
combined with version 2. Canonical source: benchmark_support/sparse_attention.py.
"""

import ast
import __future__
import copy
import functools
import hashlib
import inspect
import math
import os
import textwrap
import types
import weakref


ENV = "VIDEO_BENCH_SPARSE_ATTENTION"
SCOPE_VERSION = 2
SCOPE = "attention input communication through final result return/aggregation and output readiness"
SUPPORTED = {
    "MiniMaxH3TransformerBlock", "MiniMaxH3DiTBlock", "WanTransformerBlock_VSA",
    "WanTransformerBlock", "LongCatTransformerBlock", "LongCatSingleStreamBlock",
    "TransformerDecoderBlock", "MMDoubleStreamBlock",
}


def enabled():
    return os.environ.get(ENV) == "1"


def source_text(node):
    return ast.unparse(node)


def _contains(node, text):
    return text in source_text(node)


def _marker(kind, part):
    return ast.Expr(value=ast.Call(
        func=ast.Attribute(value=ast.Name(id="__sparse_clock", ctx=ast.Load()),
                           attr=kind, ctx=ast.Load()),
        args=[ast.Name(id="__sparse_owner", ctx=ast.Load()), ast.Constant(part)], keywords=[]))


def _is_marker(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id in {"__sparse_clock", "__disco_clock"})


def instrument_source(source, profile, module_name):
    """Instrument the actual communication/runtime method, never its DiT block."""
    original = ast.parse(textwrap.dedent(source))
    tree = copy.deepcopy(original)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    # Decorators are reapplied by _patched_function, not evaluated again here.
    decorators = [source_text(n) for n in fn.decorator_list]
    if any(d not in {"_maybe_compiler_disable"} for d in decorators):
        raise ValueError(f"Unsupported runtime decorator: {decorators}")
    fn.decorator_list = []
    original.body[0].decorator_list = []
    anchors = []

    def insert(node, kind, before):
        matches = []
        class Inject(ast.NodeTransformer):
            def generic_visit(self, current):
                if current is node:
                    matches.append(current)
                    mark = _marker(kind, "runtime")
                    return [mark, current] if before else [current, mark]
                return super().generic_visit(current)
        Inject().visit(fn)
        if len(matches) != 1:
            raise ValueError(f"Ambiguous runtime boundary: {profile}/{kind}")
        anchors.append(dict(kind=kind, part="runtime", line=node.lineno if before else node.end_lineno,
                            position="before" if before else "after", statement=source_text(node)))

    def one(predicate):
        found = [n for n in ast.walk(fn) if isinstance(n, ast.stmt) and predicate(n)]
        if len(found) != 1:
            raise ValueError(f"Missing/ambiguous runtime anchor: {profile} ({len(found)})")
        return found[0]

    def call_assign(text):
        return one(lambda n: isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                   and text in source_text(n.value.func))

    if profile in {"DistributedAttention_VSA", "UlyssesAttention_VSA"}:
        start = one(lambda n: isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and _contains(n.value.func, "sequence_model_parallel_all_to_all_4D")
                    and any(k.arg == "scatter_dim" and isinstance(k.value, ast.Constant)
                            and k.value.value == 2 for k in n.value.keywords))
        end = one(lambda n: isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                  and _contains(n.value.func, "sequence_model_parallel_all_to_all_4D")
                  and any(k.arg == "scatter_dim" and isinstance(k.value, ast.Constant)
                          and k.value.value == 1 for k in n.value.keywords))
    elif profile == "LongCatSelfAttention":
        # Conditional SP stage is also the single-GPU reference boundary.
        start = one(lambda n: isinstance(n, ast.If) and _contains(n, "scatter_dim=2"))
        end = one(lambda n: isinstance(n, ast.If) and _contains(n, "scatter_dim=1"))
    elif profile == "LongCatUlysses":
        start = call_assign("ulysses_a2a_in")
        end = call_assign("ulysses_a2a_out")
    elif profile == "HunyuanSequenceParallel":
        start = one(lambda n: isinstance(n, ast.If) and _contains(n, "query = all_to_all_4D"))
        end = one(lambda n: isinstance(n, ast.If) and _contains(n, "hidden_states = all_to_all_4D"))
        if not _contains(end, "encoder_hidden_states = all_gather"):
            raise ValueError("SSTA text-head return communication is missing")
    elif profile == "OmniAttention":
        start = call_assign("strategy.pre_attention")
        end = call_assign("strategy.post_attention")
    elif profile == "SGLangH3Core":
        # Includes the post-A2A coarse all-gather and gate fold before return.
        start = one(lambda n: isinstance(n, ast.If) and _contains(n.test, "ulysses_active")
                    and _contains(n.test, "gate_compress is None"))
        insert(start, "start", True)
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        if not returns or any(not isinstance(n.value, ast.Name) or n.value.id != "out" for n in returns):
            raise ValueError("H3 runtime return changed")
        for node in returns:
            insert(node, "end", True)
        start = end = None
    elif profile == "NablaLocal":
        start = one(lambda n: isinstance(n, ast.If) and _contains(n.test, "sparse_params is not None"))
        end = start
    else:
        raise ValueError(f"Unsupported sparse runtime profile: {profile}")
    if start is not None:
        insert(start, "start", True)
        insert(end, "end", False)

    class Erase(ast.NodeTransformer):
        def visit_Expr(self, node):
            return None if _is_marker(node) else self.generic_visit(node)
    erased = Erase().visit(copy.deepcopy(tree))
    if ast.dump(erased, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("Timing instrumentation changed original operations")
    ast.fix_missing_locations(tree)
    return tree, dict(profile=profile, module=module_name, scope_version=SCOPE_VERSION,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      anchors=anchors, decorators_preserved=decorators, ast_erasure_equal=True)


def _method_owner(module):
    """Preserve native offload/cache dispatch, patch only its inner callable."""
    manager = getattr(module, "_hook_manager", None)
    if hasattr(module, "_omni_original_forward"):
        registry = getattr(module, "_hook_registry", None)
        if getattr(registry, "_new_fwd_impl_hook", None) is not None:
            raise ValueError("Forward-replacing cache hook unsupported")
        return module, "_omni_original_forward"
    return (manager, "original_forward") if manager is not None else (module, "forward")


def _clone_globals(function, additions):
    clone = types.FunctionType(function.__code__, dict(function.__globals__, **additions),
                               function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return functools.update_wrapper(clone, function)


def _patch_bcg_passthrough(wrapper, original, patched):
    """Preserve the native eager wrapper, rejecting actual BCG capture contexts."""
    capture = wrapper.__globals__.get("_current_capture_var")
    closure = wrapper.__closure__ or ()
    if (getattr(wrapper, "__wrapped__", None) is not original or capture is None
            or sum(cell.cell_contents is original for cell in closure) != 1):
        raise ValueError("H3 native BCG wrapper changed; eager passthrough is unverified")
    cells = tuple(types.CellType(patched) if cell.cell_contents is original else cell
                  for cell in closure)
    native = types.FunctionType(wrapper.__code__, wrapper.__globals__, wrapper.__name__,
                                wrapper.__defaults__, cells)
    native.__kwdefaults__ = wrapper.__kwdefaults__
    functools.update_wrapper(native, wrapper)

    @functools.wraps(wrapper)
    def eager_only(*args, **kwargs):
        # A breakpoint flag alone does not enable BCG. Reject before the native
        # wrapper tears down a capture segment (where CUDA capture becomes false).
        if capture.get() is not None:
            raise ValueError("H3 BCG capture is unsupported for sparse runtime timing")
        return native(*args, **kwargs)
    return eager_only


def _compiled_eager_boundary(module, cls, name):
    """Keep compiled blocks; require native eager attention or explicit event breaks."""
    if cls.__name__ == "WanTransformerBlock" and cls.__module__.startswith("vllm_omni"):
        return dict(compiled_block_preserved=True, compile_event_breaks=True,
                    compilation_note="Eager event bookkeeping adds breaks before/after attention runtime")
    if cls.__name__ != "MiniMaxH3DiTBlock" or not cls.__module__.startswith("vllm_omni"):
        raise ValueError(f"Compiled block forward unsupported: {name}")
    boundary = getattr(module.attn, "_run_packed_attention", None)
    if (not getattr(boundary, "_torchdynamo_disable", False)
            or not getattr(boundary, "_torchdynamo_disable_recursive", False)):
        raise ValueError(f"Compiled H3 requires its native recursive eager attention boundary: {name}")
    original = getattr(boundary, "_torchdynamo_orig_callable", None)
    if original is None or getattr(boundary, "__wrapped__", None) is not original:
        raise ValueError(f"Unverified native H3 eager wrapper: {name}")
    source = inspect.getsource(original)
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    if ([source_text(d) for d in fn.decorator_list] != ["torch.compiler.disable"]
            or not any(isinstance(n, ast.Call) and source_text(n.func) == "self.attention"
                       for n in ast.walk(fn))):
        raise ValueError(f"Native H3 eager attention dispatch changed: {name}")
    return dict(compiled_block_preserved=True, native_eager_boundary=original.__qualname__,
                native_eager_boundary_source_sha256=hashlib.sha256(source.encode()).hexdigest())


# Reuse the exact compiled runtime function across warmup/formal observers.
# Values retain only a weak owner reference; finished models can be collected.
_COMPILED_RUNTIME_CACHE = weakref.WeakKeyDictionary()
_NABLA_HOOK_CACHE = weakref.WeakKeyDictionary()


class _EagerEventProxy:
    def __init__(self, torch, owner):
        self.owner = weakref.ref(owner)
        self.clock = None
        self.start = torch.compiler.disable(self._start)
        self.end = torch.compiler.disable(self._end)

    def _start(self, unused_owner, part):
        if self.clock is None or self.owner() is None:
            raise ValueError("Compiled attention has no active event observer")
        self.clock.start(self.owner(), part)

    def _end(self, unused_owner, part):
        if self.clock is None or self.owner() is None:
            raise ValueError("Compiled attention has no active event observer")
        self.clock.end(self.owner(), part)


class SparseAttentionEvents:
    """One request's range events; root timer supplies the forward index."""

    def __init__(self, torch, model, forwards):
        if not isinstance(forwards, int) or isinstance(forwards, bool) or forwards <= 0:
            raise ValueError("Sparse timing requires a positive forward count")
        self.torch = torch
        self.forwards = forwards
        self.forward_index = -1
        self.rows = []
        self.pending = {}
        self.restore = []
        self.names = {}
        self.manifest = []
        self.hook_restore = []
        self.event_proxies = []
        self.active_forward = False
        self.completed_forwards = 0
        self.seen = set()
        try:
            for name, module in model.named_modules():
                classes = [c for c in type(module).__mro__ if c.__name__ in SUPPORTED]
                if not classes:
                    continue
                cls = classes[0]
                if cls.__name__ == "WanTransformerBlock" and not cls.__module__.startswith("vllm_omni"):
                    continue
                owner, attribute = _method_owner(module)
                native = getattr(owner, attribute, None)
                compile_evidence = (_compiled_eager_boundary(module, cls, name)
                                    if hasattr(native, "_torchdynamo_orig_callable") else {})
                self.names[id(module)] = name
                self._attach(module, cls, name, compile_evidence.get("compile_event_breaks", False))
                self.manifest[-1].update(compile_evidence)
            if not self.manifest:
                raise ValueError("No supported sparse attention runtime found")
            self.expected_ranges = {(key, "runtime") for key in self.names}
            self.expected_per_forward = len(self.expected_ranges)
            if len(self.manifest) != self.expected_per_forward:
                raise ValueError("Sparse boundary manifest does not match model layers")
            self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                           for _ in range(forwards * self.expected_per_forward)]
            for start, end in self.events:
                start.record()
                end.record()
            torch.cuda.synchronize()
        except BaseException:
            self.detach()
            raise

    def _replace(self, owner, attribute, value):
        self.restore.append((owner, attribute, getattr(owner, attribute), attribute in vars(owner)))
        setattr(owner, attribute, value)

    def _patched_function(self, function, profile, module, name, event_breaks=False):
        function = getattr(function, "__func__", function)
        if profile != "LongCatUlysses":
            function = inspect.unwrap(function)
            if function.__closure__:
                raise ValueError(f"Compiled/wrapped runtime unsupported: {name}/{profile}")
        source = inspect.getsource(function)
        tree, evidence = instrument_source(source, profile, function.__module__)
        evidence.update(layer=name, class_name=type(module).__name__,
                        source_file=inspect.getsourcefile(function),
                        source_first_line=inspect.getsourcelines(function)[1])
        clock = self
        if event_breaks:
            cached = _COMPILED_RUNTIME_CACHE.get(module)
            if cached is not None:
                native, patched, clock = cached
                if native is not function:
                    raise ValueError(f"Native compiled runtime changed between requests: {name}")
                if clock.clock is not None:
                    raise ValueError(f"Compiled runtime already has an event observer: {name}")
                clock.clock = self
                self.event_proxies.append(clock)
                self.manifest.append(evidence)
                return patched
            clock = _EagerEventProxy(self.torch, module)
            clock.clock = self
            self.event_proxies.append(clock)
        namespace = dict(function.__globals__, __sparse_clock=clock,
                         __sparse_owner=None if event_breaks else module)
        if profile == "LongCatUlysses":
            namespace.update(inspect.getclosurevars(function).nonlocals)
        exec(compile(tree, f"<sparse-runtime:{name}:{profile}>", "exec",
                     flags=function.__code__.co_flags & __future__.annotations.compiler_flag), namespace)
        patched = functools.update_wrapper(namespace[function.__name__], function)
        for decorator in reversed(evidence["decorators_preserved"]):
            patched = function.__globals__[decorator](patched)
        if event_breaks:
            _COMPILED_RUNTIME_CACHE[module] = (function, patched, clock)
        self.manifest.append(evidence)
        return patched

    def _patch_method(self, target, attribute, profile, module, name, event_breaks=False):
        patched = self._patched_function(getattr(target, attribute), profile, module, name, event_breaks)
        self._replace(target, attribute, types.MethodType(patched, target))

    def _attach(self, module, cls, name, event_breaks=False):
        framework = cls.__module__
        if cls.__name__ == "MiniMaxH3TransformerBlock":
            self._patch_method(module.attn.distributed_attention, "forward", "DistributedAttention_VSA", module, name)
        elif cls.__name__ == "WanTransformerBlock_VSA":
            profile = "UlyssesAttention_VSA" if framework.startswith("sglang") else "DistributedAttention_VSA"
            self._patch_method(module.attn1, "forward", profile, module, name)
        elif cls.__name__ == "LongCatTransformerBlock":
            self._patch_method(module.self_attn, "forward", "LongCatSelfAttention", module, name)
        elif cls.__name__ == "LongCatSingleStreamBlock":
            self._patch_method(module.attn, "_process_attn", "LongCatUlysses", module, name)
        elif cls.__name__ == "MiniMaxH3DiTBlock" and framework.startswith("sglang"):
            target = module.attn
            original = target._attend.__func__
            core = original.__globals__["_minimax_h3_attention_core_impl"]
            patched = self._patched_function(core, "SGLangH3Core", module, name)
            replacements = {"_minimax_h3_attention_core_impl": patched}
            if getattr(target, "bcg_breakpoint", False):
                replacements["_minimax_h3_attention_core_bcg"] = _patch_bcg_passthrough(
                    original.__globals__["_minimax_h3_attention_core_bcg"], core, patched)
            cloned = _clone_globals(original, replacements)
            self._replace(target, "_attend", types.MethodType(cloned, target))
        elif framework.startswith("vllm_omni"):
            target = module.attn.attention if cls.__name__ == "MiniMaxH3DiTBlock" else module.attn1.attn
            self._patch_method(target, "_forward_impl", "OmniAttention", module, name, event_breaks)
        elif cls.__name__ == "MMDoubleStreamBlock":
            owner, attribute = _method_owner(module)
            original = getattr(owner, attribute).__func__
            dispatch = original.__globals__["parallel_attention"]
            inner = inspect.unwrap(dispatch)
            runtime = inner.__globals__["sequence_parallel_attention"]
            patched = self._patched_function(runtime, "HunyuanSequenceParallel", module, name)
            cloned = _clone_globals(inner, {"sequence_parallel_attention": patched})
            # Keep the original compiler-disable boundary around this dispatcher.
            cloned = self.torch.compiler.disable(cloned)
            forward = _clone_globals(original, {"parallel_attention": cloned})
            self._replace(owner, attribute, types.MethodType(forward, module))
        elif cls.__name__ == "TransformerDecoderBlock":
            target = module.self_attention
            q_hooks = getattr(target.to_query, "_forward_hooks", {})
            output_hooks = getattr(target.out_layer, "_forward_pre_hooks", {})
            if q_hooks or output_hooks:
                self._attach_nabla_distributed(target, module, name, q_hooks, output_hooks)
            else:
                self._patch_method(target, "forward", "NablaLocal", module, name)
        else:
            raise ValueError(f"Unsupported sparse runtime: {framework}.{cls.__name__}")

    def _attach_nabla_distributed(self, target, module, name, q_hooks, output_hooks):
        """Bracket the actual DTensor redistribution hooks, excluding both edge GEMMs."""
        def select(hooks, kind):
            found = []
            for key, hook in hooks.items():
                if kind in getattr(hook, "__qualname__", ""):
                    found.append((key, hook))
            if len(found) != 1:
                raise ValueError(f"NABLA DTensor {kind} hook changed: {name}")
            return found[0]
        q_key, q_hook = select(q_hooks, "PrepareModuleOutput")
        o_key, o_hook = select(output_hooks, "PrepareModuleInput")
        # Hook event bookkeeping runs eager even inside compiled get_qkv/out_l.
        # Existing DTensor hooks and compiled arithmetic remain in their order.
        cached = _NABLA_HOOK_CACHE.get(module)
        if cached is None:
            proxy = _EagerEventProxy(self.torch, module)
            def input_exchange(*args, **kwargs):
                proxy.start(None, "runtime")
                return q_hook(*args, **kwargs)
            def output_exchange(*args, **kwargs):
                result = o_hook(*args, **kwargs)
                proxy.end(None, "runtime")
                return result
            # Each layer has a different proxy. Skip only these bookkeeping
            # frames, avoiding per-proxy Dynamo guards/cache exhaustion while
            # permitting the original redistribution hooks to compile.
            input_exchange = self.torch.compiler.disable(input_exchange, recursive=False)
            output_exchange = self.torch.compiler.disable(output_exchange, recursive=False)
            _NABLA_HOOK_CACHE[module] = (q_hook, o_hook, input_exchange, output_exchange, proxy)
        else:
            native_q, native_o, input_exchange, output_exchange, proxy = cached
            if native_q is not q_hook or native_o is not o_hook:
                raise ValueError(f"Native NABLA DTensor hooks changed between requests: {name}")
        if proxy.clock is not None:
            raise ValueError(f"NABLA DTensor hooks already have an event observer: {name}")
        # Stable hooks prevent compiled get_qkv/out_l from guarding a new
        # closure/observer identity on every formal request after warmup.
        proxy.clock = self
        self.event_proxies.append(proxy)
        self.hook_restore.extend([(q_hooks, q_key, q_hook), (output_hooks, o_key, o_hook)])
        q_hooks[q_key] = input_exchange
        output_hooks[o_key] = output_exchange
        evidence = dict(layer=name, class_name=type(module).__name__, profile="NablaDTensor",
                        scope_version=SCOPE_VERSION, compile_event_breaks=True,
                        interleaved_work="K/V projection and QK Norm/RoPE between native DTensor exchanges",
                        anchors=[])
        for kind, hook in [("start", q_hook), ("end", o_hook)]:
            source = inspect.getsource(hook)
            evidence["anchors"].append(dict(kind=kind, statement=hook.__qualname__,
                position="before" if kind == "start" else "after", source_file=inspect.getsourcefile(hook),
                source_first_line=inspect.getsourcelines(hook)[1],
                source_sha256=hashlib.sha256(source.encode()).hexdigest()))
        self.manifest.append(evidence)

    def begin_forward(self, index):
        if self.active_forward or self.pending:
            raise ValueError("Unclosed sparse attention forward/range")
        if (not isinstance(index, int) or isinstance(index, bool)
                or index != self.completed_forwards or index >= self.forwards):
            raise ValueError("Sparse forward index is out of order or exceeds capacity")
        self.active_forward = True
        self.forward_index = index
        self.seen.clear()

    def end_forward(self):
        if not self.active_forward or self.pending or self.seen != self.expected_ranges:
            raise ValueError("Missing or duplicate sparse attention ranges")
        self.active_forward = False
        self.completed_forwards += 1

    def _stream_identity(self):
        stream = self.torch.cuda.current_stream()
        return stream.cuda_stream, str(stream.device)

    def start(self, module, part):
        if self.torch.compiler.is_compiling():
            raise ValueError("Sparse runtime event bookkeeping must execute eagerly")
        if self.torch.cuda.is_current_stream_capturing():
            raise ValueError("Sparse range timing requires native graph mode off")
        key = (id(module), part)
        if (not self.active_forward or key not in self.expected_ranges or key in self.seen
                or len(self.rows) >= len(self.events)):
            raise ValueError("Sparse attention event capacity/range mismatch")
        index = len(self.rows)
        stream_id, device = self._stream_identity()
        self.pending[key] = (index, stream_id, device)
        self.seen.add(key)
        self.rows.append(dict(forward_index=self.forward_index, layer=self.names[id(module)], part=part,
                              stream_id=stream_id, device=device))
        self.events[index][0].record()

    def end(self, module, part):
        key = (id(module), part)
        if key not in self.pending:
            raise ValueError("Sparse attention END has no matching START")
        index, stream_id, device = self.pending[key]
        if self._stream_identity() != (stream_id, device):
            raise ValueError("Sparse runtime did not restore its caller stream; output readiness is unverified")
        self.events[index][1].record()
        del self.pending[key]

    def result(self):
        if (self.active_forward or self.pending or self.completed_forwards != self.forwards
                or len(self.rows) != len(self.events)):
            raise ValueError("Incomplete sparse attention request")
        rows = [dict(row, elapsed_ms=a.elapsed_time(b)) for row, (a, b) in zip(self.rows, self.events)]
        totals = [0.0] * self.forwards
        for row in rows:
            totals[row["forward_index"]] += row["elapsed_ms"]
        return dict(metric="sparse_attention", scope_version=SCOPE_VERSION,
                    range=SCOPE,
                    includes_communication=True, excludes_output_projection_and_residual=True,
                    excludes_cross_attention_and_ffn=True,
                    forward_ms=totals, rows=rows, boundaries=self.manifest)

    def detach(self):
        for proxy in self.event_proxies:
            if proxy.clock is self:
                proxy.clock = None
        self.event_proxies.clear()
        for hooks, key, hook in reversed(self.hook_restore):
            hooks[key] = hook
        self.hook_restore.clear()
        for owner, attribute, original, shadowed in reversed(self.restore):
            if shadowed:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self.restore.clear()


def aggregate_sparse(record, ranks):
    """Keep one real rank's partition; never sum per-layer maxima."""
    values = [r.get("sparse_attention") for r in ranks]
    if not any(v is not None for v in values):
        return
    if not all(v is not None for v in values):
        raise ValueError("Sparse measurements missing on some ranks")
    if {value.get("scope_version", 1) for value in values} != {SCOPE_VERSION}:
        raise ValueError("Sparse scope versions differ or are legacy; v2 requires fresh measurements")
    record.update(sparse_attention_scope_version=SCOPE_VERSION, sparse_attention_range=SCOPE)
    for rank, value in zip(ranks, values):
        if len(value["forward_ms"]) != len(rank["dit_step_ms"]):
            raise ValueError("Sparse/DiT forward counts differ")
        if any(not math.isfinite(x) or x < 0
               for x in value["forward_ms"] + rank["dit_step_ms"]):
            raise ValueError("Invalid sparse/DiT elapsed time")
        if any(s > d + 1.0 for s, d in zip(value["forward_ms"], rank["dit_step_ms"])):
            raise ValueError("Sparse attention exceeds its enclosing DiT")
    slowest = max(range(len(ranks)), key=lambda i: sum(ranks[i]["dit_step_ms"]))
    on_dit_rank = sum(values[slowest]["forward_ms"]) / 1000
    dit = sum(ranks[slowest]["dit_step_ms"]) / 1000
    record.update(sparse_attention_sum_max_rank_s=max(sum(v["forward_ms"]) for v in values) / 1000,
                  sparse_attention_on_dit_max_rank_s=on_dit_rank,
                  dit_other_on_dit_max_rank_s=dit - on_dit_rank,
                  sparse_attention_fraction_of_dit=on_dit_rank / dit,
                  dit_partition_rank=ranks[slowest]["rank"])


def validate_record_scopes(records):
    versions = {r.get("sparse_attention_scope_version", 1) for r in records
                if "sparse_attention_on_dit_max_rank_s" in r}
    if len(versions) > 1:
        raise ValueError("Cannot summarize legacy and corrected sparse timing scopes together")


SPARSE_FIELDS = ("sparse_attention_sum_max_rank_s", "sparse_attention_on_dit_max_rank_s",
                 "dit_other_on_dit_max_rank_s", "sparse_attention_fraction_of_dit",
                 "sparse_attention_scope_version")


def report_lines(rows):
    observed = [r for r in rows if "sparse_attention_on_dit_max_rank_s" in r]
    if not observed:
        return []
    versions = {v for r in observed for v in r.get("sparse_attention_scope_version", {}).get("values", [1])}
    if len(versions) != 1:
        raise ValueError("Cannot combine legacy and corrected sparse timing scopes in one report")
    version = versions.pop()
    lines = ["", f"## Sparse attention runtime（scope v{version}）", "",
             "范围：attention 输入通信阶段开始，至最后结果回传/聚合和必要等待结束；包含路由、选块、稀疏计算及通信。",
             "不含输出投影/residual、cross-attention/FFN；通信间交错的计算与逐模型边界见 metrics.json 的 boundaries。",
             "Attention 和其余 DiT 使用 DiT 总耗时最大 rank 的同一条执行路径；单独 max(rank attention sum) 另存。",
             "", "| 模型/配方 | GPU | Attention (s) | 其余 DiT (s) | Attention/DiT |",
             "|---|---:|---:|---:|---:|"]
    if version == 1:
        lines[3:5] = ["旧口径：输入 Norm/调制 → 输出投影及 residual；不能与 scope v2 通信范围比较。"]
    for row in observed:
        def mean(key):
            return row[key]["mean"]
        label = row.get("model", row.get("offload", "native"))
        lines.append(f"| {label} | {row['gpus']} | {mean(SPARSE_FIELDS[1]):.6f} | "
                     f"{mean(SPARSE_FIELDS[2]):.6f} | {mean(SPARSE_FIELDS[3]):.2%} |")
    return lines


class DiscoSparseRangeProbe:
    """Disco: after layer_qkv, before write_qkv -> result combination before projection.

    Includes slab staging, pooled KV/metadata communication, routing, remote
    fetch/merge, fence/read_o and coarse gate combination. Excludes QKV/input
    Norm, output projection/residual, cross-attention and FFN. Deferred GPU
    validation: installing this probe does not start an experiment.
    """

    def __init__(self, timer, driver_class, bodies):
        self.timer = timer
        self.pending = None
        self.manifest = []
        self.restore = []

        def patch_method(cls, attribute, start):
            original = getattr(cls, attribute)
            source = inspect.getsource(original)
            before = ast.parse(textwrap.dedent(source))
            tree = copy.deepcopy(before)
            fn = tree.body[0]
            if fn.decorator_list:
                raise ValueError("Decorated Disco runtime unsupported")
            if start:
                targets = [n for n in fn.body if isinstance(n, ast.Assign)
                           and isinstance(n.value, ast.Call) and _contains(n.value.func, "self.adapter.layer_qkv")]
            else:
                targets = [n for n in fn.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id in {"combined", "token_major"}
                                   for t in n.targets)]
            if len(targets) != 1:
                raise ValueError(f"Disco communication boundary changed: {cls.__name__}/{attribute}")
            anchor = targets[0]
            kind = "start" if start else "end"
            fn.body.insert(fn.body.index(anchor) + 1,
                           ast.parse(f"__disco_clock.{kind}(layer)").body[0])
            class Erase(ast.NodeTransformer):
                def visit_Expr(self, node):
                    return None if _is_marker(node) else self.generic_visit(node)
            if ast.dump(Erase().visit(copy.deepcopy(tree))) != ast.dump(before):
                raise ValueError("Disco timing changed original operations")
            ast.fix_missing_locations(tree)
            namespace = dict(original.__globals__, __disco_clock=self)
            exec(compile(tree, f"<sparse-runtime:{cls.__module__}.{attribute}>", "exec",
                         flags=original.__code__.co_flags & __future__.annotations.compiler_flag), namespace)
            self.restore.append((cls, attribute, original))
            patched = functools.update_wrapper(namespace[original.__name__], original)
            setattr(cls, attribute, patched)
            self.manifest.append(dict(class_name=cls.__name__, method=attribute, scope_version=SCOPE_VERSION,
                source_file=inspect.getsourcefile(original), source_first_line=inspect.getsourcelines(original)[1],
                source_sha256=hashlib.sha256(source.encode()).hexdigest(), ast_erasure_equal=True,
                anchors=[dict(kind=kind, position="after", statement=source_text(anchor), line=anchor.end_lineno)]))
            return patched

        try:
            for cls in bodies:
                patch_method(cls, "layer_after_attention", False)
            patched = patch_method(driver_class, "run_layer", True)
            @functools.wraps(patched)
            def run_layer(driver, layer, *args, **kwargs):
                value = patched(driver, layer, *args, **kwargs)
                if timer.enabled and self.pending is not None:
                    raise ValueError("Missing Disco attention result-ready marker")
                return value
            driver_class.run_layer = run_layer
        except BaseException:
            self.detach()
            raise

    def start(self, layer):
        if not self.timer.enabled:
            return
        if self.pending is not None:
            raise ValueError("Unclosed Disco attention runtime")
        start, end = self.timer._take()
        self.pending = (layer, start, end)
        start.record()

    def end(self, layer):
        if not self.timer.enabled:
            return
        if self.pending is None or layer != self.pending[0]:
            raise ValueError("Mismatched Disco attention runtime layer")
        _, start, end = self.pending
        end.record()
        self.timer._records.append(("sparse_attention", start, end))
        self.pending = None

    def detach(self):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)
        self.restore.clear()
