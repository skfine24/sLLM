from pathlib import Path

import pytest

from encoding_dsv4 import (
    IMAGE_PLACEHOLDER,
    encode_messages,
    load_cases,
    parse_tagged_text,
)


def test_plain_text_prompt_is_unchanged():
    prompt = encode_messages(
        [{"role": "user", "content": "hello"}],
        thinking_mode="chat",
    )
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>hello"
        "<｜Assistant｜></think>"
    )


def test_multiturn_text_prompt_is_unchanged():
    prompt = encode_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ],
        thinking_mode="chat",
    )
    assert prompt == (
        "<｜begin▁of▁sentence｜>sys<｜User｜>q1<｜Assistant｜></think>"
        "a1<｜end▁of▁sentence｜><｜User｜>q2<｜Assistant｜></think>"
    )


def test_top_level_image_block_returns_matching_placeholder_and_record():
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "images/image_1.jpeg"}},
            {"type": "text", "text": "describe"},
        ],
    }]
    prompt, media = encode_messages(
        messages,
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜>\n\n"
        "describe<｜Assistant｜></think>"
    )
    assert prompt.count(IMAGE_PLACEHOLDER) == len(media["images"]) == 1
    assert media["images"][0]["url"] == "images/image_1.jpeg"


def test_tagged_text_matches_standard_image_content_blocks():
    tagged_content = parse_tagged_text(
        "before<image>images/image_1.jpeg</image>after"
    )
    standard_content = [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "images/image_1.jpeg"}},
        {"type": "text", "text": "after"},
    ]
    tagged = encode_messages(
        [{"role": "user", "content": tagged_content}],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    standard = encode_messages(
        [{"role": "user", "content": standard_content}],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert tagged == standard


def test_tagged_text_preserves_multiple_image_order():
    content = parse_tagged_text(
        "<image>first.png</image>middle<image>second.png</image>"
    )
    prompt, media = encode_messages(
        [{"role": "user", "content": content}],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert prompt.count(IMAGE_PLACEHOLDER) == 2
    assert [image["url"] for image in media["images"]] == [
        "first.png",
        "second.png",
    ]


def test_txt_and_json_examples_encode_identically():
    root = Path(__file__).parent.parent
    examples = root / "inference" / "examples"
    text = (examples / "example_vl.txt").read_text().rstrip("\n")
    json_case = load_cases(str(examples / "example_vl_harmony.json"))[0]

    txt_encoded = encode_messages(
        [{"role": "user", "content": parse_tagged_text(text)}],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    json_encoded = encode_messages(
        json_case["messages"],
        thinking_mode="chat",
        return_multi_modal_data=True,
    )

    assert txt_encoded == json_encoded
    prompt, media = txt_encoded
    assert prompt.count(IMAGE_PLACEHOLDER) == 2
    assert [image["url"] for image in media["images"]] == [
        "examples/images/carrots.jpeg",
        "examples/images/corn.jpeg",
    ]


def test_malformed_tagged_text_is_rejected():
    with pytest.raises(ValueError, match="Malformed"):
        parse_tagged_text("<image>missing end tag")


def test_nested_tool_result_preserves_image_placeholder():
    messages = [{
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call-1",
            "content": [
                {"type": "image_url", "image_url": {"url": "images/image_1.jpeg"}},
                {"type": "text", "text": "nested"},
            ],
        }],
    }]
    prompt, media = encode_messages(
        messages,
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert "<tool_result><｜deepseek_image｜>\n\nnested</tool_result>" in prompt
    assert prompt.count(IMAGE_PLACEHOLDER) == len(media["images"]) == 1


def test_tool_role_with_image_blocks_preserves_placeholder():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "inspect", "arguments": "{}"},
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": [
                {"type": "image_url", "image_url": {"url": "images/image_1.jpeg"}},
                {"type": "text", "text": "tool image"},
            ],
        },
    ]
    prompt, media = encode_messages(
        messages,
        thinking_mode="chat",
        return_multi_modal_data=True,
    )
    assert "<tool_result><｜deepseek_image｜>\n\ntool image</tool_result>" in prompt
    assert prompt.count(IMAGE_PLACEHOLDER) == len(media["images"]) == 1


def test_context_images_are_not_returned_as_current_media():
    context = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "images/image_1.jpeg"}},
            {"type": "text", "text": "previous"},
        ],
    }]
    prompt, media = encode_messages(
        [{"role": "user", "content": "now"}],
        thinking_mode="chat",
        context=context,
        return_multi_modal_data=True,
    )
    assert IMAGE_PLACEHOLDER not in prompt
    assert media == {"images": []}


def test_user_supplied_placeholder_is_rejected():
    with pytest.raises(ValueError, match="image special token"):
        encode_messages(
            [{"role": "user", "content": IMAGE_PLACEHOLDER}],
            thinking_mode="chat",
        )


def test_image_block_without_source_is_rejected():
    with pytest.raises(ValueError, match="valid source"):
        encode_messages(
            [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": {}}],
            }],
            thinking_mode="chat",
            return_multi_modal_data=True,
        )
