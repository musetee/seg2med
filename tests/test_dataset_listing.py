"""Dataset discovery should describe configuration without scanning files."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from seg2med_workflow import prepare_datasets
from seg2med_datasets.datasets import (
    available_dataset_kinds,
    describe_dataset_requirements,
)


class DatasetListingTests(unittest.TestCase):
    def test_requirements_cover_every_kind_and_alias(self):
        expected = {
            "anika": "ct_root + mr_root",
            "anish": "csv OR root",
            "csv": "csv",
            "generic": "name",
            "synthrad": "root",
            "trial": "root",
            "umm": "csv",
            "uitm_ms_nmo": "image_root + mask_root",
        }
        self.assertEqual(describe_dataset_requirements(), expected)
        self.assertEqual(sorted(expected), available_dataset_kinds())

    def test_print_list_needs_no_config_and_does_not_scan(self):
        output = io.StringIO()
        with patch("sys.argv", ["seg2med_workflow.prepare_datasets", "--print-list"]), \
                patch.object(prepare_datasets, "prepare_from_config") as prepare, \
                redirect_stdout(output):
            self.assertEqual(prepare_datasets.main(), 0)
        prepare.assert_not_called()
        text = output.getvalue()
        for kind, requirements in describe_dataset_requirements().items():
            self.assertIn(f"  {kind}\n    required: {requirements}\n", text)
        self.assertIn("root modalities: ct (ID 0)", text)
        self.assertIn("Known modality IDs", text)
        self.assertIn("mr_vibe_in", text)


if __name__ == "__main__":
    unittest.main()
