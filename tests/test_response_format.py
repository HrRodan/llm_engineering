import sys
import unittest
from typing import Dict, Any
from pydantic import BaseModel

# Ensure we can import ai_tools by adding project root to sys.path
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tools.tools import LLMQuery


class UserInfo(BaseModel):
    name: str
    age: int
    is_active: bool


class TestResponseFormat(unittest.TestCase):
    def test_pydantic_model_format(self):
        query = LLMQuery(response_format=UserInfo)
        kwargs = query._prepare_request_kwargs(
            messages=[], stream=False, json_format=False
        )

        self.assertIn("response_format", kwargs)
        rf = kwargs["response_format"]
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"], "UserInfo")
        self.assertTrue(rf["json_schema"]["strict"])
        self.assertIn("schema", rf["json_schema"])

        # Check schema content (simplified check)
        schema = rf["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertIn("name", schema["properties"])
        self.assertIn("age", schema["properties"])

    def test_dict_format(self):
        custom_format = {"type": "json_object"}
        query = LLMQuery(response_format=custom_format)
        kwargs = query._prepare_request_kwargs(
            messages=[], stream=False, json_format=False
        )

        self.assertIn("response_format", kwargs)
        self.assertEqual(kwargs["response_format"], custom_format)

    def test_json_format_precedence(self):
        # json_format=True should override response_format
        query = LLMQuery(json_format=True, response_format=UserInfo)
        kwargs = query._prepare_request_kwargs(
            messages=[], stream=False, json_format=True
        )  # json_format passed as arg

        self.assertIn("response_format", kwargs)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_json_format_instance_precedence(self):
        # json_format=True in init should also trigger the priority
        query = LLMQuery(json_format=True, response_format=UserInfo)
        # Note: _prepare_request_kwargs logic uses instance var if arg is None, but here we explicitly pass checks
        # Let's test checking what query() calls.
        # But for unit test, we can just check _prepare_request_kwargs

        # If we pass json_format=True to the method (which query() does if self.json_format is True)
        kwargs = query._prepare_request_kwargs(
            messages=[], stream=False, json_format=True
        )
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

        # If we pass json_format=False, it should fall back to response_format
        kwargs = query._prepare_request_kwargs(
            messages=[], stream=False, json_format=False
        )
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")


if __name__ == "__main__":
    unittest.main()
