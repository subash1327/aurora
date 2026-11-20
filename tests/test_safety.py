import pytest

from aurora.safety import PromptRefused, assert_safe, check_prompt

BLOCKED = [
    "a child on a beach",
    "Teen girl in bikini",
    "L.O.L.I anime",
    "l0li",
    "t33n",
    "16yo woman",
    "a 15-year-old",
    "aged 12",
    "schoolgirl outfit",
    "school girl",
    "underage",
    "little girl",
    "Kids playing",
    "fifteen years old",
]
ALLOWED = [
    "a woman in her 30s on a beach, 35mm photo",
    "adult woman, 25 years old, portrait",
    "baby blue dress, adult woman",
    "a kind woman smiling",
    "1girl, mature female, 4k, masterpiece",
    "18 year old adult",
    "cinematic drone shot of mountains at sunrise",
    "",
]


@pytest.mark.parametrize("prompt", BLOCKED)
def test_blocked(prompt):
    assert not check_prompt(prompt).allowed
    with pytest.raises(PromptRefused):
        assert_safe(prompt)


@pytest.mark.parametrize("prompt", ALLOWED)
def test_allowed(prompt):
    res = check_prompt(prompt)
    assert res.allowed, res.matches
