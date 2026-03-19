# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for batched Code2Wav decode.

Verifies that decoding multiple requests via a single batched
decoder.chunked_decode() call (with right-padding to Fmax) produces
per-request outputs equivalent to sequential BS=1 decode.
"""

import pytest
import torch
import torch.nn as nn

NUM_QUANTIZERS = 8
TOTAL_UPSAMPLE = 4
DEVICE = torch.device("cpu")


class SyntheticDecoder(nn.Module):
    """Minimal decoder with Conv1d layers and chunked_decode support.

    Uses standard (non-causal) Conv1d to stress-test padding effects --
    the real decoder uses causal convolutions where padding impact is
    even smaller.
    """

    def __init__(self, num_quantizers=NUM_QUANTIZERS, total_upsample=TOTAL_UPSAMPLE):
        super().__init__()
        hidden = 32
        self.total_upsample = total_upsample
        self.embed = nn.Conv1d(num_quantizers, hidden, kernel_size=3, padding=1)
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose1d(hidden, hidden, kernel_size=total_upsample, stride=total_upsample)
        self.out = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, codes):
        x = codes.float()
        x = torch.relu(self.embed(x))
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.upsample(x)
        return self.out(x).clamp(min=-1, max=1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        wavs = []
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            wavs.append(wav_chunk[..., context_size * self.total_upsample :])
            start_index = end_index
        return torch.cat(wavs, dim=-1)


@pytest.fixture(scope="module")
def decoder():
    torch.manual_seed(42)
    return SyntheticDecoder().to(DEVICE).eval()


def _random_codes(seq_len, batch=1, device=DEVICE):
    return torch.randint(0, 100, (batch, NUM_QUANTIZERS, seq_len), dtype=torch.long, device=device)


def _sequential_decode(decoder, codes_list):
    """Decode each [Q, F] tensor individually at BS=1, return list of 1-D wavs."""
    results = []
    for codes_qf in codes_list:
        codes_bqf = codes_qf.unsqueeze(0)  # [1, Q, F]
        wav = decoder.chunked_decode(codes_bqf)  # [1, 1, wav_len]
        results.append(wav.squeeze(0).squeeze(0))  # [wav_len]
    return results


def _assert_output_lengths(bat_wavs, lengths):
    """Assert each batched output has the correct length (L * TOTAL_UPSAMPLE)."""
    expected = [L * TOTAL_UPSAMPLE for L in lengths]
    actual = [w.shape[0] for w in bat_wavs]
    assert expected == actual, f"Shape mismatch: expected {expected}, got {actual}"


def _assert_interior_close(bat_wavs, seq_wavs, boundary=3 * TOTAL_UPSAMPLE, atol=1e-5, rtol=1e-5):
    """Assert interior samples (away from padding boundary) match between batched and sequential."""
    for bat_wav, seq_wav in zip(bat_wavs, seq_wavs):
        if seq_wav.shape[0] > boundary:
            torch.testing.assert_close(bat_wav[:-boundary], seq_wav[:-boundary], atol=atol, rtol=rtol)


def _batched_decode(decoder, codes_list, upsample):
    """Pad, stack, single batched chunked_decode, trim per-request."""
    actual_frames = [c.shape[1] for c in codes_list]
    f_max = max(actual_frames)

    if len(codes_list) == 1:
        codes_bqf = codes_list[0].unsqueeze(0)
    else:
        padded = []
        for codes_qf in codes_list:
            f_i = codes_qf.shape[1]
            if f_i < f_max:
                codes_qf = torch.nn.functional.pad(codes_qf, (0, f_max - f_i))
            padded.append(codes_qf)
        codes_bqf = torch.stack(padded, dim=0)

    wav_batch = decoder.chunked_decode(codes_bqf)  # [B, 1, wav_len_max]

    results = []
    for j, f_actual in enumerate(actual_frames):
        wav_j = wav_batch[j].squeeze(0)
        expected_len = f_actual * upsample
        results.append(wav_j[:expected_len])
    return results


# ──────────────────────────────────────────────────────────────────
# 1. Same-length requests: batched == sequential (bit-identical)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [25, 50, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_same_length_close(decoder, seq_len, batch_size):
    """All requests have the same frame count -- no padding needed.

    BS=1 is bit-identical.  BS>1 may have tiny floating-point diffs
    (~1e-6) from batched GEMM accumulation order in Conv1d.
    """
    torch.manual_seed(7)
    codes_list = [
        torch.randint(0, 100, (NUM_QUANTIZERS, seq_len), dtype=torch.long, device=DEVICE) for _ in range(batch_size)
    ]
    with torch.no_grad():
        seq_wavs = _sequential_decode(decoder, codes_list)
        bat_wavs = _batched_decode(decoder, codes_list, TOTAL_UPSAMPLE)

    atol, rtol = (0, 0) if batch_size == 1 else (1e-5, 1e-4)
    for bat_wav, seq_wav in zip(bat_wavs, seq_wavs):
        torch.testing.assert_close(bat_wav, seq_wav, atol=atol, rtol=rtol)


# ──────────────────────────────────────────────────────────────────
# 2. Variable-length requests: correct output lengths + bounded diff
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lengths", [[10, 25], [25, 50, 100], [7, 13, 50, 80]])
def test_variable_length_output_shapes(decoder, lengths):
    """Each request's output must be trimmed to its actual frame count."""
    torch.manual_seed(11)
    codes_list = [torch.randint(0, 100, (NUM_QUANTIZERS, L), dtype=torch.long, device=DEVICE) for L in lengths]
    with torch.no_grad():
        bat_wavs = _batched_decode(decoder, codes_list, TOTAL_UPSAMPLE)

    _assert_output_lengths(bat_wavs, lengths)


@pytest.mark.parametrize("lengths", [[10, 25], [25, 50, 100], [7, 13, 50, 80]])
def test_variable_length_interior_close(decoder, lengths):
    """Interior positions (away from padding boundary) should be close.

    With non-causal convolutions, the boundary region is affected by
    right-padding.  Interior positions more than (receptive_field * upsample)
    samples from the end should match sequential decode closely.
    """
    torch.manual_seed(11)
    codes_list = [torch.randint(0, 100, (NUM_QUANTIZERS, L), dtype=torch.long, device=DEVICE) for L in lengths]
    with torch.no_grad():
        seq_wavs = _sequential_decode(decoder, codes_list)
        bat_wavs = _batched_decode(decoder, codes_list, TOTAL_UPSAMPLE)

    _assert_interior_close(bat_wavs, seq_wavs)


# ──────────────────────────────────────────────────────────────────
# 3. Single valid request (BS=1 fast path, no padding)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seq_len", [25, 50])
def test_single_request_bit_identical(decoder, seq_len):
    """BS=1 fast path should be bit-identical to sequential decode."""
    torch.manual_seed(3)
    codes_qf = torch.randint(0, 100, (NUM_QUANTIZERS, seq_len), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        seq_wavs = _sequential_decode(decoder, [codes_qf])
        bat_wavs = _batched_decode(decoder, [codes_qf], TOTAL_UPSAMPLE)

    torch.testing.assert_close(bat_wavs[0], seq_wavs[0], atol=0, rtol=0)


# ──────────────────────────────────────────────────────────────────
# 4. Long sequences that trigger internal chunking
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lengths", [[350, 350], [200, 400], [100, 350, 500]])
def test_long_sequences_chunked_internally(decoder, lengths):
    """Sequences exceeding chunk_size=300 trigger internal chunking.

    Batched decode should still produce correct per-request output
    lengths, and interior positions should be close to sequential.
    """
    torch.manual_seed(17)
    codes_list = [torch.randint(0, 100, (NUM_QUANTIZERS, L), dtype=torch.long, device=DEVICE) for L in lengths]
    with torch.no_grad():
        seq_wavs = _sequential_decode(decoder, codes_list)
        bat_wavs = _batched_decode(decoder, codes_list, TOTAL_UPSAMPLE)

    _assert_output_lengths(bat_wavs, lengths)
    _assert_interior_close(bat_wavs, seq_wavs)


# ──────────────────────────────────────────────────────────────────
# 5. Output values bounded
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lengths", [[10, 25], [25, 50, 100]])
def test_output_bounded(decoder, lengths):
    """All batched outputs must remain in [-1, 1]."""
    torch.manual_seed(5)
    codes_list = [torch.randint(0, 100, (NUM_QUANTIZERS, L), dtype=torch.long, device=DEVICE) for L in lengths]
    with torch.no_grad():
        bat_wavs = _batched_decode(decoder, codes_list, TOTAL_UPSAMPLE)

    for i, wav in enumerate(bat_wavs):
        assert wav.min() >= -1.0 and wav.max() <= 1.0, f"Request {i}: values out of [-1, 1] range"
