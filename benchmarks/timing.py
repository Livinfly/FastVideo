"""Optional CUDA events around complete transformer forwards; never synchronize per step."""


class DiTEvents:

    def __init__(self, torch, steps):
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                       for _ in range(steps)]
        self.torch = torch
        self.sparse = None
        self.count = 0
        self.handles = []
        # CUDA event allocation is lazy. Prime every pair OUTSIDE request timing.
        for begin, end in self.events:
            begin.record()
            end.record()
        torch.cuda.synchronize()

    def attach(self, transformer):
        self.count = 0
        from sparse_attention import SparseAttentionEvents, enabled
        self.sparse = SparseAttentionEvents(self.torch, transformer, len(self.events)) if enabled() else None
        self.handles = [
            transformer.register_forward_pre_hook(self._begin, prepend=True),
            transformer.register_forward_hook(self._end)
        ]

    def _begin(self, module, inputs):
        # DiT START: before existing root pre-hooks; covers full DiT + SP/FSDP.
        if self.count >= len(self.events):
            raise ValueError('Unexpected extra DiT forward')
        self.events[self.count][0].record()
        if self.sparse is not None:
            self.sparse.begin_forward(self.count)

    def _end(self, module, inputs, output):
        # DiT END: after existing root post-hooks; excludes the scheduler.
        if self.sparse is not None:
            self.sparse.end_forward()
        self.events[self.count][1].record()
        self.count += 1

    def detach(self):
        if self.sparse is not None:
            self.sparse.detach()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def elapsed_ms(self):
        # Caller synchronizes AFTER E2E has ended, before querying these events.
        if self.count != len(self.events):
            raise ValueError(f"Expected {len(self.events)} DiT forwards, got {self.count}")
        return [begin.elapsed_time(end) for begin, end in self.events]

    def sparse_result(self):
        return self.sparse.result() if self.sparse is not None else None
