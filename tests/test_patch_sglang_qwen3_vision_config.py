import textwrap
import unittest

from scripts.tools.patch_sglang_qwen3_vision_config import patch_source


class PatchSGLangQwen3VisionConfigTest(unittest.TestCase):
    def test_patch_converts_dict_vision_config_before_hidden_size_access(self) -> None:
        source = textwrap.dedent(
            """
            class Qwen3VLMoeVisionModel:
                def __init__(self, vision_config):
                    self.hidden_size = vision_config.hidden_size
                    self.patch_size = vision_config.patch_size
                    self.nested = vision_config.nested.value
            """
        )

        patched, changed = patch_source(source)
        namespace: dict[str, object] = {}
        exec(patched, namespace)
        model_cls = namespace["Qwen3VLMoeVisionModel"]
        model = model_cls({"hidden_size": 4096, "patch_size": 14, "nested": {"value": 7}})

        self.assertTrue(changed)
        self.assertEqual(model.hidden_size, 4096)
        self.assertEqual(model.patch_size, 14)
        self.assertEqual(model.nested, 7)

    def test_patch_source_is_idempotent(self) -> None:
        source = textwrap.dedent(
            """
            class Qwen3VLMoeVisionModel:
                def __init__(self, vision_config):
                    self.hidden_size = vision_config.hidden_size
            """
        )

        patched, changed = patch_source(source)
        patched_again, changed_again = patch_source(patched)

        self.assertTrue(changed)
        self.assertFalse(changed_again)
        self.assertEqual(patched, patched_again)


if __name__ == "__main__":
    unittest.main()
