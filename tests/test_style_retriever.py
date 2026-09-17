import unittest

from tools.rag.style_retriever import retrieve_local_demo_styles


class StyleRetrieverTests(unittest.TestCase):
    def test_empty_query_is_rejected_before_loading_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "query_must_not_be_empty"):
            retrieve_local_demo_styles("   ")

    def test_top_k_is_bounded_before_loading_dependencies(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k_must_be_between_1_and_5"):
            retrieve_local_demo_styles("国风", top_k=6)


if __name__ == "__main__":
    unittest.main()
