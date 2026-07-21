"""Fixed Blackwell 2SM/TMA kernels for Pi0/Pi0.5's two large GEMMs."""

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_barrier_count,
)
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


BM_PER_CTA = 128
CLUSTER_M = 2 * BM_PER_CTA
BLOCK_N = 128
BLOCK_K = 64
DEFAULT_TWO_CTA_MMA = os.environ.get(
    "REALTIME_VLA_GLUON_TWO_CTA_MMA", "1"
) != "0"
@gluon.aggregate
class _PipelineState:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return _PipelineState(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self):
        next_index = self.index + 1
        rollover = next_index == self.num_barriers
        index = gl.where(rollover, 0, next_index)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return _PipelineState(index, phase, self.num_barriers)


@gluon.jit
def _gate_load_partition(
    x_desc,
    w_desc,
    w2_desc,
    x_bufs,
    w_bufs,
    w2_bufs,
    gate_empty_bars,
    up_empty_bars,
    ready_bars,
    off_m,
    off_n,
    MULTICAST: gl.constexpr,
):
    block_k: gl.constexpr = x_desc.block_shape[1]
    stages: gl.constexpr = ready_bars.shape[0]
    state = _PipelineState.create(1, stages)
    for k in range(0, x_desc.shape[1], block_k):
        reuse = k >= block_k * stages
        mbarrier.wait(
            gate_empty_bars.index(state.index), state.phase, pred=reuse
        )
        mbarrier.wait(up_empty_bars.index(state.index), state.phase, pred=reuse)
        ready_bar = ready_bars.index(state.index)
        mbarrier.expect(
            ready_bar,
            x_desc.nbytes_per_cta
            + w_desc.nbytes_per_cta
            + w2_desc.nbytes_per_cta,
        )
        tma.async_load(
            x_desc,
            [off_m, k],
            ready_bar,
            x_bufs.index(state.index),
            multicast=MULTICAST,
        )
        tma.async_load(
            w_desc,
            [k, off_n],
            ready_bar,
            w_bufs.index(state.index),
            multicast=MULTICAST,
        )
        tma.async_load(
            w2_desc,
            [k, off_n],
            ready_bar,
            w2_bufs.index(state.index),
            multicast=MULTICAST,
        )
        state = state.next()


@gluon.jit
def _gate_mma_partition(
    x_bufs,
    w_bufs,
    w2_bufs,
    gate_empty_bars,
    up_empty_bars,
    ready_bars,
    acc,
    acc2,
    acc_ready_bar,
    num_k_tiles,
    MULTICAST: gl.constexpr,
):
    state = _PipelineState.create(0, ready_bars.shape[0])
    use_acc = False
    for _ in range(num_k_tiles):
        mbarrier.wait(ready_bars.index(state.index), state.phase)
        x_buf = x_bufs.index(state.index)
        w_buf = w_bufs.index(state.index)
        w2_buf = w2_bufs.index(state.index)
        tcgen05_mma(
            x_buf,
            w_buf,
            acc,
            use_acc=use_acc,
            multicast=MULTICAST,
            mbarriers=[gate_empty_bars.index(state.index)],
        )
        tcgen05_mma(
            x_buf,
            w2_buf,
            acc2,
            use_acc=use_acc,
            multicast=MULTICAST,
            mbarriers=[up_empty_bars.index(state.index)],
        )
        state = state.next()
        use_acc = True
    if MULTICAST:
        tcgen05_commit(
            acc_ready_bar,
            descs=[x_bufs.index(0), w_bufs.index(0), w2_bufs.index(0)],
        )
    else:
        tcgen05_commit(acc_ready_bar)


@gluon.jit
def _gate_epilogue_partition(
    acc, acc2, acc_ready_bar, out_desc, off_m, off_n, tile_n: gl.constexpr
):
    mbarrier.wait(acc_ready_bar, phase=0)
    split_n: gl.constexpr = out_desc.block_shape[1]
    split_count: gl.constexpr = tile_n // split_n
    out_smem = gl.allocate_shared_memory(
        out_desc.dtype, out_desc.block_shape, out_desc.layout
    )
    for split in gl.static_range(split_count):
        gate = acc.slice(split * split_n, split_n).load()
        up = acc2.slice(split * split_n, split_n).load()
        gate = gate / (
            1.0
            + libdevice.exp(
                -1.5957691216057308 * gate * (1.0 + 0.044715 * gate * gate)
            )
        )
        # A single subtile buffer cuts Gate shared memory by 16 KiB per CTA.
        tma.store_wait(0)
        out_smem.store((gate * up).to(out_desc.dtype))
        tma.async_copy_shared_to_global(
            out_desc, [off_m, off_n + split * split_n], out_smem
        )
    tma.store_wait(0)
    out_smem._keep_alive()


@gluon.jit
def gate_encoder_pipeline_kernel(
    x_desc,
    w_desc,
    w2_desc,
    out_desc,
    M,
    N,
    K,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EPILOGUE_N: gl.constexpr,
    STAGES: gl.constexpr,
    LOAD_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    GROUP_M: gl.constexpr,
    TWO_CTA_MMA: gl.constexpr,
    MULTICAST: gl.constexpr,
):
    gl.static_assert(x_desc.block_shape[1] == BLOCK_K)
    gl.static_assert(w_desc.block_shape[1] == BLOCK_N)
    gl.static_assert(out_desc.block_shape[1] == EPILOGUE_N)
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, x_desc.block_shape[0])
    num_pid_n = gl.cdiv(N, w_desc.block_shape[1])
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    off_m = pid_m * x_desc.block_shape[0]
    off_n = pid_n * w_desc.block_shape[1]

    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype, [STAGES] + x_desc.block_shape, x_desc.layout
    )
    w_bufs = gl.allocate_shared_memory(
        w_desc.dtype, [STAGES] + w_desc.block_shape, w_desc.layout
    )
    w2_bufs = gl.allocate_shared_memory(
        w2_desc.dtype, [STAGES] + w2_desc.block_shape, w2_desc.layout
    )
    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=1,
        cga_layout=((1, 0),),
        two_ctas=TWO_CTA_MMA,
    )
    acc = allocate_tensor_memory(
        gl.float32, [x_desc.block_shape[0], w_desc.block_shape[1]], acc_layout
    )
    acc2 = allocate_tensor_memory(
        gl.float32, [x_desc.block_shape[0], w_desc.block_shape[1]], acc_layout
    )

    gate_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    up_empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    ready_bars = mbarrier.allocate_mbarrier(
        batch=STAGES, two_ctas=TWO_CTA_MMA
    )
    gate_count: gl.constexpr = tcgen05_mma_barrier_count(
        [x_bufs.index(0), w_bufs.index(0)],
        multicast=MULTICAST,
        two_ctas=TWO_CTA_MMA,
    )
    up_count: gl.constexpr = tcgen05_mma_barrier_count(
        [x_bufs.index(0), w2_bufs.index(0)],
        multicast=MULTICAST,
        two_ctas=TWO_CTA_MMA,
    )
    for i in gl.static_range(STAGES):
        mbarrier.init(gate_empty_bars.index(i), count=gate_count)
        mbarrier.init(up_empty_bars.index(i), count=up_count)
        mbarrier.init(ready_bars.index(i), count=1)

    acc_ready_bar = mbarrier.allocate_mbarrier()
    acc_ready_count: gl.constexpr = tcgen05_mma_barrier_count(
        [x_bufs.index(0), w_bufs.index(0), w2_bufs.index(0)],
        multicast=MULTICAST,
        two_ctas=TWO_CTA_MMA,
    )
    mbarrier.init(acc_ready_bar, count=acc_ready_count)

    num_k_tiles = gl.cdiv(K, x_desc.block_shape[1])
    gl.warp_specialize(
        [
            (
                _gate_epilogue_partition,
                (
                    acc,
                    acc2,
                    acc_ready_bar,
                    out_desc,
                    off_m,
                    off_n,
                    w_desc.block_shape[1],
                ),
            ),
            (
                _gate_load_partition,
                (
                    x_desc,
                    w_desc,
                    w2_desc,
                    x_bufs,
                    w_bufs,
                    w2_bufs,
                    gate_empty_bars,
                    up_empty_bars,
                    ready_bars,
                    off_m,
                    off_n,
                    MULTICAST,
                ),
            ),
            (
                _gate_mma_partition,
                (
                    x_bufs,
                    w_bufs,
                    w2_bufs,
                    gate_empty_bars,
                    up_empty_bars,
                    ready_bars,
                    acc,
                    acc2,
                    acc_ready_bar,
                    num_k_tiles,
                    MULTICAST,
                ),
            ),
        ],
        [1, 1],
        [LOAD_REGS, MMA_REGS],
    )

    for i in gl.static_range(STAGES):
        mbarrier.invalidate(gate_empty_bars.index(i))
        mbarrier.invalidate(up_empty_bars.index(i))
        mbarrier.invalidate(ready_bars.index(i))
    mbarrier.invalidate(acc_ready_bar)


@gluon.jit
def _down_load_partition(
    x_desc,
    w_desc,
    x_bufs,
    w_bufs,
    empty_bars,
    ready_bars,
    off_m,
    off_n,
    MULTICAST: gl.constexpr,
):
    block_k: gl.constexpr = x_desc.block_shape[1]
    stages: gl.constexpr = ready_bars.shape[0]
    state = _PipelineState.create(1, stages)
    for k in range(0, x_desc.shape[1], block_k):
        mbarrier.wait(
            empty_bars.index(state.index),
            state.phase,
            pred=(k >= block_k * stages),
        )
        ready_bar = ready_bars.index(state.index)
        mbarrier.expect(
            ready_bar, x_desc.nbytes_per_cta + w_desc.nbytes_per_cta
        )
        tma.async_load(
            x_desc,
            [off_m, k],
            ready_bar,
            x_bufs.index(state.index),
            multicast=MULTICAST,
        )
        tma.async_load(
            w_desc,
            [k, off_n],
            ready_bar,
            w_bufs.index(state.index),
            multicast=MULTICAST,
        )
        state = state.next()


@gluon.jit
def _down_mma_partition(
    x_bufs,
    w_bufs,
    empty_bars,
    ready_bars,
    acc,
    acc_ready_bar,
    num_k_tiles,
    MULTICAST: gl.constexpr,
):
    state = _PipelineState.create(0, ready_bars.shape[0])
    use_acc = False
    for _ in range(num_k_tiles):
        mbarrier.wait(ready_bars.index(state.index), state.phase)
        x_buf = x_bufs.index(state.index)
        w_buf = w_bufs.index(state.index)
        tcgen05_mma(
            x_buf,
            w_buf,
            acc,
            use_acc=use_acc,
            multicast=MULTICAST,
            mbarriers=[empty_bars.index(state.index)],
        )
        state = state.next()
        use_acc = True
    if MULTICAST:
        tcgen05_commit(acc_ready_bar, descs=[x_bufs.index(0), w_bufs.index(0)])
    else:
        tcgen05_commit(acc_ready_bar)


@gluon.jit
def _down_epilogue_partition(
    acc,
    acc_ready_bar,
    residual_bar,
    out_desc,
    off_m,
    off_n,
    tile_n: gl.constexpr,
    MULTICAST: gl.constexpr,
):
    mbarrier.wait(acc_ready_bar, phase=0)
    split_n: gl.constexpr = out_desc.block_shape[1]
    split_count: gl.constexpr = tile_n // split_n
    residual_smem = gl.allocate_shared_memory(
        out_desc.dtype, out_desc.block_shape, out_desc.layout
    )
    residual_phase = 0
    for split in gl.static_range(split_count):
        tma.store_wait(0)
        mbarrier.expect(residual_bar, out_desc.nbytes_per_cta)
        tma.async_load(
            out_desc,
            [off_m, off_n + split * split_n],
            residual_bar,
            residual_smem,
            multicast=MULTICAST,
        )
        mbarrier.wait(residual_bar, phase=residual_phase, deps=[residual_smem])
        residual_phase ^= 1
        acc_slice = acc.slice(split * split_n, split_n)
        result = acc_slice.load() + residual_smem.load(
            acc_slice.get_reg_layout()
        ).to(gl.float32)
        residual_smem.store(result.to(out_desc.dtype))
        tma.async_copy_shared_to_global(
            out_desc,
            [off_m, off_n + split * split_n],
            residual_smem,
        )
    tma.store_wait(0)
    residual_smem._keep_alive()


@gluon.jit
def res_ffndown_pipeline_kernel(
    x_desc,
    w_desc,
    out_desc,
    out_restore,
    M,
    N,
    K,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    EPILOGUE_N: gl.constexpr,
    STAGES: gl.constexpr,
    LOAD_REGS: gl.constexpr,
    MMA_REGS: gl.constexpr,
    GROUP_M: gl.constexpr,
    TWO_CTA_MMA: gl.constexpr,
    MULTICAST: gl.constexpr,
):
    gl.static_assert(x_desc.block_shape[1] == BLOCK_K)
    gl.static_assert(w_desc.block_shape[1] == BLOCK_N)
    gl.static_assert(out_desc.block_shape[1] == EPILOGUE_N)
    pid = gl.program_id(0)
    num_pid_m = gl.cdiv(M, x_desc.block_shape[0])
    num_pid_n = gl.cdiv(N, w_desc.block_shape[1])
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = gl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_in_group % group_size_m
    pid_n = pid_in_group // group_size_m
    off_m = pid_m * x_desc.block_shape[0]
    off_n = pid_n * w_desc.block_shape[1]

    x_bufs = gl.allocate_shared_memory(
        x_desc.dtype, [STAGES] + x_desc.block_shape, x_desc.layout
    )
    w_bufs = gl.allocate_shared_memory(
        w_desc.dtype, [STAGES] + w_desc.block_shape, w_desc.layout
    )
    acc_layout: gl.constexpr = TensorMemoryLayout(
        block=(128, BLOCK_N),
        col_stride=1,
        cga_layout=((1, 0),),
        two_ctas=TWO_CTA_MMA,
    )
    acc = allocate_tensor_memory(
        gl.float32, [x_desc.block_shape[0], w_desc.block_shape[1]], acc_layout
    )

    empty_bars = mbarrier.allocate_mbarrier(batch=STAGES)
    # The TMA producer feeds one cta_group::2 MMA consumer.  Both CTAs must
    # therefore observe the same ready barrier, while the MMA completion
    # barriers remain per-CTA and are multicast by tcgen05.
    ready_bars = mbarrier.allocate_mbarrier(
        batch=STAGES, two_ctas=TWO_CTA_MMA
    )
    mma_count: gl.constexpr = tcgen05_mma_barrier_count(
        [x_bufs.index(0), w_bufs.index(0)],
        multicast=MULTICAST,
        two_ctas=TWO_CTA_MMA,
    )
    for i in gl.static_range(STAGES):
        mbarrier.init(empty_bars.index(i), count=mma_count)
        mbarrier.init(ready_bars.index(i), count=1)

    acc_ready_bar = mbarrier.allocate_mbarrier()
    residual_bar = mbarrier.allocate_mbarrier()
    mbarrier.init(acc_ready_bar, count=mma_count)
    mbarrier.init(residual_bar, count=1)

    num_k_tiles = gl.cdiv(K, x_desc.block_shape[1])
    gl.warp_specialize(
        [
            (
                _down_epilogue_partition,
                (
                    acc,
                    acc_ready_bar,
                    residual_bar,
                    out_desc,
                    off_m,
                    off_n,
                    w_desc.block_shape[1],
                    MULTICAST,
                ),
            ),
            (
                _down_load_partition,
                (
                    x_desc,
                    w_desc,
                    x_bufs,
                    w_bufs,
                    empty_bars,
                    ready_bars,
                    off_m,
                    off_n,
                    MULTICAST,
                ),
            ),
            (
                _down_mma_partition,
                (
                    x_bufs,
                    w_bufs,
                    empty_bars,
                    ready_bars,
                    acc,
                    acc_ready_bar,
                    num_k_tiles,
                    MULTICAST,
                ),
            ),
        ],
        [1, 1],
        [LOAD_REGS, MMA_REGS],
    )

    for i in gl.static_range(STAGES):
        mbarrier.invalidate(empty_bars.index(i))
        mbarrier.invalidate(ready_bars.index(i))
    mbarrier.invalidate(acc_ready_bar)
    mbarrier.invalidate(residual_bar)


def _set_descriptor_tiles_group1(nargs):
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    epilogue_n = nargs["EPILOGUE_N"]
    x_block = [CLUSTER_M, block_k]
    w_block = [block_k, block_n]
    out_block = [CLUSTER_M, epilogue_n]

    nargs["x_desc"].block_shape = x_block
    nargs["w_desc"].block_shape = w_block
    nargs["out_desc"].block_shape = out_block
    if "w2_desc" in nargs:
        nargs["w2_desc"].block_shape = w_block

    cga_m = ((1, 0),)
    cga_broadcast = ((0, 0),)
    nargs["x_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        x_block, gl.bfloat16, cga_layout=cga_m
    )
    nargs["w_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        w_block, gl.bfloat16, cga_layout=cga_broadcast
    )
    if "w2_desc" in nargs:
        nargs["w2_desc"].layout = nargs["w_desc"].layout
    nargs["out_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        out_block, gl.bfloat16, cga_layout=cga_m
    )


def _set_descriptor_tiles_group2(nargs):
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    epilogue_n = nargs["EPILOGUE_N"]
    x_block = [CLUSTER_M, block_k]
    w_block = [block_k, block_n]
    out_block = [CLUSTER_M, epilogue_n]

    nargs["x_desc"].block_shape = x_block
    nargs["w_desc"].block_shape = w_block
    nargs["out_desc"].block_shape = out_block
    if "w2_desc" in nargs:
        nargs["w2_desc"].block_shape = w_block

    # A cta_group::2 MMA is an outer product across the CTA pair: A is
    # distributed across M and B across N.  This is deliberately different
    # from the group-1 multicast path, which broadcasts all of B to both CTAs.
    cga_a = ((1, 0),)
    cga_b = ((0, 1),)
    cga_c = ((1, 0),)
    nargs["x_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        x_block, gl.bfloat16, cga_layout=cga_a
    )
    nargs["w_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        w_block, gl.bfloat16, cga_layout=cga_b
    )
    if "w2_desc" in nargs:
        nargs["w2_desc"].layout = nargs["w_desc"].layout
    nargs["out_desc"].layout = gl.NVMMASharedLayout.get_default_for(
        out_block, gl.bfloat16, cga_layout=cga_c
    )


def _config(
    block_n,
    block_k,
    epilogue_n,
    stages,
    warps,
    mma_regs,
    group_m,
    two_cta_mma,
):
    return triton.Config(
        {
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "EPILOGUE_N": epilogue_n,
            "STAGES": stages,
            "LOAD_REGS": 24,
            "MMA_REGS": mma_regs,
            "GROUP_M": group_m,
            "TWO_CTA_MMA": two_cta_mma,
        },
        num_warps=warps,
        num_stages=1,
        num_ctas=2,
        pre_hook=(
            _set_descriptor_tiles_group2
            if two_cta_mma
            else _set_descriptor_tiles_group1
        ),
    )


gate_encoder_autotuned_kernel = triton.autotune(
    configs=[
        _config(
            256,
            64,
            128,
            4,
            8,
            32,
            4,
            True,
        )
    ],
    key=["M", "N", "K"],
)(gate_encoder_pipeline_kernel)

res_ffndown_autotuned_kernel = triton.autotune(
    configs=[
        _config(
            256,
            128,
            128,
            3,
            8,
            48,
            4,
            True,
        )
    ],
    key=["M", "N", "K"],
    restore_value=["out_restore"],
)(res_ffndown_pipeline_kernel)

# Keep the exact pre-cta_group::2 implementation available for controlled A/B
# runs.  The production wrappers select the group-2 variants by default.
gate_encoder_group1_kernel = triton.autotune(
    configs=[_config(128, 64, 128, 4, 8, 32, 4, False)],
    key=["M", "N", "K"],
)(gate_encoder_pipeline_kernel)

res_ffndown_group1_kernel = triton.autotune(
    configs=[_config(256, 128, 128, 2, 8, 48, 4, False)],
    key=["M", "N", "K"],
    restore_value=["out_restore"],
)(res_ffndown_pipeline_kernel)


def _layouts(dtype, block_n=BLOCK_N, block_k=BLOCK_K, out_block_n=BLOCK_N):
    # Adjacent M tiles belong to CTA 0/1; a zero basis broadcasts B/weights.
    cga_m = ((1, 0),)
    cga_broadcast = ((0, 0),)
    x_layout = gl.NVMMASharedLayout.get_default_for(
        [CLUSTER_M, block_k], dtype, cga_layout=cga_m
    )
    w_layout = gl.NVMMASharedLayout.get_default_for(
        [block_k, block_n], dtype, cga_layout=cga_broadcast
    )
    out_layout = gl.NVMMASharedLayout.get_default_for(
        [CLUSTER_M, out_block_n], dtype, cga_layout=cga_m
    )
    acc_layout = TensorMemoryLayout(
        block=(BM_PER_CTA, block_n),
        col_stride=1,
        cga_layout=cga_m,
        two_ctas=False,
    )
    return x_layout, w_layout, out_layout, acc_layout


def _check_bf16(*tensors):
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("Gluon large-GEMM experiment currently requires BF16 tensors")


def launch_gate_encoder(
    x,
    weight_gate,
    weight_up,
    out,
    *,
    two_cta_mma=None,
):
    """Launch Gate+Up, using Blackwell cta_group::2 by default."""
    _check_bf16(x, weight_gate, weight_up, out)
    if two_cta_mma is None:
        two_cta_mma = DEFAULT_TWO_CTA_MMA
    m, k = x.shape
    n = weight_gate.shape[1]
    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.bfloat16)
    x_desc = TensorDescriptor.from_tensor(x, dummy_block, dummy_layout)
    w_desc = TensorDescriptor.from_tensor(weight_gate, dummy_block, dummy_layout)
    w2_desc = TensorDescriptor.from_tensor(weight_up, dummy_block, dummy_layout)
    out_desc = TensorDescriptor.from_tensor(out, dummy_block, dummy_layout)
    grid = lambda meta: (
        triton.cdiv(m, CLUSTER_M) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    kernel = (
        gate_encoder_autotuned_kernel
        if two_cta_mma
        else gate_encoder_group1_kernel
    )
    return kernel[grid](
        x_desc,
        w_desc,
        w2_desc,
        out_desc,
        m,
        n,
        k,
        MULTICAST=True,
    )


def launch_res_ffndown(x, weight, out, *, two_cta_mma=None):
    """Launch FFN-down, using Blackwell cta_group::2 by default."""
    _check_bf16(x, weight, out)
    if two_cta_mma is None:
        two_cta_mma = DEFAULT_TWO_CTA_MMA
    m, k = x.shape
    n = weight.shape[1]
    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.bfloat16)
    x_desc = TensorDescriptor.from_tensor(x, dummy_block, dummy_layout)
    w_desc = TensorDescriptor.from_tensor(weight, dummy_block, dummy_layout)
    out_desc = TensorDescriptor.from_tensor(out, dummy_block, dummy_layout)
    grid = lambda meta: (
        triton.cdiv(m, CLUSTER_M) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    kernel = (
        res_ffndown_autotuned_kernel
        if two_cta_mma
        else res_ffndown_group1_kernel
    )
    return kernel[grid](
        x_desc,
        w_desc,
        out_desc,
        out,
        m,
        n,
        k,
        MULTICAST=True,
    )
