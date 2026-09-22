import json
from pathlib import Path

from neuro_os.domain import Intent
from neuro_os.sources.synthetic import TARGET_FREQUENCIES


def test_browser_targets_match_decoder_configuration() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "apps" / "ssvep-stimulus" / "targets.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    browser_targets = {
        Intent(item["intent"]): float(item["frequency_hz"])
        for item in payload["targets"]
    }

    assert browser_targets == TARGET_FREQUENCIES
