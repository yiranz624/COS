from __future__ import annotations

import unittest

from benchmarks.core.registry import get_config
from benchmarks.judges import has_judge


class BrowseComp200ConfigTests(unittest.TestCase):
    def test_dataset_and_judge_are_registered(self) -> None:
        config = get_config("browsecomp_200")

        self.assertEqual(config.key, "BrowseComp-200")
        self.assertEqual(config.default_pipeline, "react_base")
        self.assertTrue(has_judge("browsecomp_200"))


if __name__ == "__main__":
    unittest.main()
