from __future__ import annotations

import unittest

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from orchestrations.reference_v7 import MANIFEST, solve


class ReferenceV7Tests(unittest.IsolatedAsyncioTestCase):
    async def test_static_subtasks_use_no_decomposer(self) -> None:
        async def handler(request):
            if request.role == "solver":
                return f"candidate from {request.key}"
            if request.role == "verifier":
                return {"accepted": True, "critique": ""}
            if request.role == "synthesizer":
                return "final answer"
            raise AssertionError(request.role)

        executor = FakeWorkerExecutor(
            default_handler=handler,
            allowed_roles={"solver", "verifier", "synthesizer"},
        )
        result = await ScriptRuntime(executor).run(
            solve, task="question", run_id="reference-v7",
        )

        self.assertEqual(result.value, "final answer")
        self.assertNotIn("decomposer", [request.role for request in executor.requests])
        self.assertEqual(
            MANIFEST["roles"], ("solver", "verifier", "synthesizer"),
        )


if __name__ == "__main__":
    unittest.main()
