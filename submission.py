#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X

"""
FP4 quant + FP4 GEMM reference: bf16 A, MXFP4 B -> MXFP4 per-1x32 quant A -> gemm_a4w4 -> bf16 C.
Quant logic follows aiter op_tests/test_gemm_a4w4.py (get_triton_quant(QuantType.per_1x32)).
"""

from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Reference: MXFP4 per-1x32 quant on A; B_shuffle, B_scale_sh from generate_input.
    gemm_a4w4 with bpreshuffle=True.
    """
    import aiter
    import torch
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    def _quant_mxfp4(x):
        x_fp4, bs_e8m0 = dynamic_mxfp4_quant(x)
        bs_e8m0 = e8m0_shuffle(bs_e8m0)
        return x_fp4.view(dtypes.fp4x2), bs_e8m0.view(dtypes.fp8_e8m0)

    A, _, _, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, _ = A.shape
    n = B_shuffle.shape[0]

    A_q, A_scale_sh = _quant_mxfp4(A)
    if m < 32:
        out = torch.empty(((m + 31) // 32 * 32, n), dtype=dtypes.bf16, device=A.device)
        out_gemm = aiter.gemm_a4w4_asm(
            A_q,
            B_shuffle,
            A_scale_sh,
            B_scale_sh,
            out,
            "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E",
            bpreshuffle=True,
            log2_k_split=0,
        )
        return out_gemm[:m]

    return aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )
