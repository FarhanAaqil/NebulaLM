from nebulalm.data.acquire import RateLimiter, document_id, raw_checksum, within_cutoff
from nebulalm.data.adapters.openstax import record_from_html
from nebulalm.data.adapters.wikipedia import record_from_payload
from nebulalm.data.normalize import normalize


def test_document_id_and_checksum_are_stable():
    assert document_id("wikipedia_en_astronomy", "123 45") == "doc_wikipedia_en_astronomy_123_45"
    text = "NGC 1300"
    assert raw_checksum(text) == raw_checksum(text)
    assert len(raw_checksum(text)) == 64


def test_cutoff_rejects_later_revisions():
    assert within_cutoff("2023-12-31T23:59:59Z", "2024-01-01T00:00:00Z")
    assert not within_cutoff("2024-01-01T00:00:01Z", "2024-01-01T00:00:00Z")
    assert within_cutoff(None, "2024-01-01T00:00:00Z", allow_null=True)
    assert not within_cutoff(None, "2024-01-01T00:00:00Z", allow_null=False)


def test_rate_limiter_waits():
    sleeps = []
    clock = {"now": 0.0}

    def sleeper(delay):
        sleeps.append(delay)
        clock["now"] += delay

    limiter = RateLimiter(1.0, sleeper=sleeper, clock=lambda: clock["now"])
    limiter.wait()
    clock["now"] += 0.2
    limiter.wait()
    assert sleeps == [0.8]


def test_openstax_html_keeps_notation():
    html = """
    <html><body><nav>Home</nav>
    <h1>Stars</h1>
    <p>α Centauri has a toy velocity of -2.5 km/s. Mass is written M<sub>⊙</sub>.</p>
    <p>NGC 1300 and M31 are catalog examples. Distance 1.3 pc.</p>
    <table><tr><td>drop me</td></tr></table>
    </body></html>
    """
    record = record_from_html(
        section_id="1-1",
        title="Stars",
        canonical_url="https://openstax.org/books/astronomy-2e/pages/1-1",
        html=html,
        retrieved_at="2026-10-25T00:00:00Z",
    )
    prose = normalize(record["text"])
    assert "α Centauri" in prose
    assert "-2.5 km/s" in prose
    assert "M_⊙" in prose
    assert "NGC 1300" in prose
    assert "drop me" not in prose
    assert record["document_id"] == "doc_openstax_astronomy_2e_1-1"
    assert record["license_id"] == "CC-BY-4.0"


def test_wikipedia_payload_uses_old_revision():
    payload = {
        "query": {
            "pages": {
                "1": {
                    "pageid": 26751,
                    "title": "Sun",
                    "revisions": [
                        {
                            "revid": 119000,
                            "timestamp": "2023-12-01T00:00:00Z",
                            "slots": {
                                "main": {
                                    "content": "== Overview ==\nThe Sun is a star.\n\nA second paragraph mentions α Centauri and -2.5 km/s without a citation [1]. {{Infobox|x=1}}"
                                }
                            },
                        }
                    ],
                }
            }
        }
    }
    record = record_from_payload(payload, "Sun", "2026-10-25T00:00:00Z")
    prose = normalize(record["text"])
    assert record["revision_id"] == "119000"
    assert record["published_at"] == "2023-12-01T00:00:00Z"
    assert "Infobox" not in prose
    assert "[1]" not in prose
    assert "α Centauri" in prose
    assert "Overview" in prose
    assert len(record["raw_sha256"]) == 64
