"""C1 qwen4_exp device-resident decode parity tests (SKIPPED unless the built
sllm_gpu.so exports the sllm_q4_*_dev set and a CUDA device is present).

The device decode (kernels/q4_device_decode.Q4DeviceDecodeState) is compared
against the numpy oracle (ref/qwen4_exp_pipeline.decode_step), the single
source of truth. Runs on the DGX cluster after `kernels/cuda/build.sh`.
"""

from __future__ import annotations

import unittest

import numpy as np

from kernels import _q4_cuda as q4
from kernels import _sllm_cuda as ck
from kernels.q4_device_decode import Q4DeviceDecodeState, Q4DeviceWeightTable
from ref import qwen4_exp_pipeline as qp
from serving.dev_model import tiny_qwen4_exp_cfg, tiny_qwen4_exp_weights

_NEEDS_GPU = not (q4.dev_available() and ck.device_count() >= 1)


@unittest.skipIf(_NEEDS_GPU, "sllm_q4_*_dev kernels not built / no CUDA")
class TestQ4DeviceDecode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = tiny_qwen4_exp_cfg()
        cls.cfg.validate()
        cls.weights = tiny_qwen4_exp_weights(cls.cfg)
        cls.ids = [1, 2, 3, 4]

    def _fresh_state(self):
        st, _ = qp.prefill(self.ids, self.weights, self.cfg)
        return st

    def test_step_parity(self):
        # two independent states so oracle and device both start from prefill
        st_o = self._fresh_state()
        st_d = self._fresh_state()
        table = Q4DeviceWeightTable(self.weights, self.cfg, dtype="fp32")
        dev = Q4DeviceDecodeState(table, st_d, self.cfg)
        try:
            last = self.ids[-1]
            for _ in range(6):
                lo = qp.decode_step(st_o, self.weights, self.cfg, last)
                ld = dev.step(last)
                np.testing.assert_allclose(
                    ld, lo, rtol=2e-3, atol=2e-3,
                    err_msg=f"decode step mismatch at token {last}")
                last = int(np.argmax(lo))
        finally:
            dev.free()
            table.free()

    def test_state_advances(self):
        st_d = self._fresh_state()
        table = Q4DeviceWeightTable(self.weights, self.cfg, dtype="fp32")
        dev = Q4DeviceDecodeState(table, st_d, self.cfg)
        try:
            n0 = st_d.n_ctx
            dev.step(self.ids[-1])
            self.assertEqual(st_d.n_ctx, n0 + 1)
            self.assertGreater(st_d.layers[2]["k"].shape[1], 0)
        finally:
            dev.free()
            table.free()


if __name__ == "__main__":
    unittest.main(verbosity=2)
