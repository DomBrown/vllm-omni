"""Tests for OmniOpenAIServingChat.warmup() chat template handling."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_serving_chat():
    """Build a minimal OmniOpenAIServingChat without a real engine."""
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    instance = object.__new__(OmniOpenAIServingChat)
    instance.chat_template = None
    instance.chat_template_content_format = "auto"
    instance.default_chat_template_kwargs = {}

    mock_tokenizer = MagicMock()
    mock_tokenizer.name_or_path = "test-model"

    mock_renderer = MagicMock()
    mock_renderer.get_tokenizer.return_value = mock_tokenizer

    instance.renderer = mock_renderer

    mock_model_config = MagicMock()
    mock_model_config.trust_remote_code = False
    mock_model_config.hf_config.model_type = "test"
    instance.model_config = mock_model_config

    return instance


def test_warmup_skips_when_no_chat_template():
    """When no chat template is resolvable, warmup should skip
    without calling super().warmup() or raising an error."""
    instance = _make_mock_serving_chat()

    with patch(
        "vllm.renderers.hf.resolve_chat_template",
        return_value=None,
    ) as mock_resolve:
        with patch(
            "vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat.warmup",
            new_callable=AsyncMock,
        ) as mock_super_warmup:
            asyncio.run(instance.warmup())

            mock_resolve.assert_called_once()
            mock_super_warmup.assert_not_called()


def test_warmup_proceeds_when_chat_template_exists():
    """When a chat template IS resolvable, warmup should call
    super().warmup() normally."""
    instance = _make_mock_serving_chat()

    fake_template = "{% for m in messages %}{{ m.content }}{% endfor %}"
    with patch(
        "vllm.renderers.hf.resolve_chat_template",
        return_value=fake_template,
    ) as mock_resolve:
        with patch(
            "vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat.warmup",
            new_callable=AsyncMock,
        ) as mock_super_warmup:
            asyncio.run(instance.warmup())

            mock_resolve.assert_called_once()
            mock_super_warmup.assert_called_once()
