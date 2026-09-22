import pytest

from neuro_os.decoders.ssvep import SSVEPDecoder
from neuro_os.domain import Intent
from neuro_os.sources.synthetic import generate_ssvep


@pytest.mark.parametrize("intent", [Intent.LEFT, Intent.RIGHT, Intent.SELECT, Intent.BACK])
def test_decoder_recognizes_synthetic_target(intent: Intent) -> None:
    decoded = SSVEPDecoder().decode(generate_ssvep(intent))
    assert decoded.intent is intent
    assert decoded.confidence >= 0.55
