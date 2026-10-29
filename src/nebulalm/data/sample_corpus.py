"""Authored Phase 1 corpus.

The prose is original fixture text, including seeded survey-id sentences so a
byte-level BPE run can actually reach the 4k and 8k vocabulary targets.
It is not OpenStax or Wikipedia text.
"""

from __future__ import annotations

import random

RETRIEVED_AT = "2026-10-25T00:00:00Z"
PIPELINE_VERSION = "phase1-slice"

_BODIES = [
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Ceres", "Europa", "Titan", "Ganymede", "Io", "Triton", "Enceladus", "Luna",
]
_STARS = [
    "Sirius", "Vega", "Polaris", "Betelgeuse", "Rigel", "Altair", "Deneb",
    "Procyon", "Antares", "Spica", "Arcturus", "Alpha Centauri", "Beta Pictoris",
]


def _paragraph(sentences: list[str]) -> str:
    return " ".join(sentences)


def hand_written_documents() -> list[dict]:
    """Notation fixtures, near-duplicates, and records that must be excluded."""
    alpha = (
        "Alpha Centauri is a nearby stellar system, and Beta Pictoris is a young star with a debris disk. "
        "Observers record the pair when they discuss proper motion, radial velocity, and catalog cross-matches. "
        "The Greek letters α and β stay in the prose because they identify the stars in Bayer notation.\n\n"
        "A report may say that the relative speed is -2.5 km/s and that the distance is about 1.3 pc. "
        "The solar mass is written M_⊙ and the solar luminosity is written L_⊙. "
        "Coordinates in this note are 14h 39m and −60° 50′, with the minus sign kept on the declination. "
        "NGC 1300 and M31 appear here only as examples of catalog style, not as claims about those galaxies."
    )
    alpha_near = (
        "Alpha Centauri is a nearby stellar system, and Beta Pictoris is a young star with a debris disk. "
        "Observers record the pair when they discuss proper motion, radial velocity, and catalog cross-matches. "
        "The Greek letters α and β stay in the prose because they identify the stars in Bayer notation.\n\n"
        "A report may say that the relative speed is -2.5 km/s and that the distance is about 1.3 pc. "
        "The solar mass is written M_⊙ and the solar luminosity is written L_⊙. "
        "Coordinates in this note are 14h 39m and −60° 50′, with the minus sign kept on the declination. "
        "NGC 1300 and M31 appear in the second copy with one added clause about duplicate detection."
    )
    shared = (
        "The transit notebook lists the same field stars on three nights so the deduplicator can see a chain. "
        "Night logs keep the magnitude, the filter, and the airmass next to the object name. "
        "Nothing in this paragraph is taken from a textbook page or a wiki revision. "
        "The numbers stay small and the sentences stay long enough for a five-gram overlap. "
        "The observer wrote the start time, the end time, and the sky quality in the same order each night. "
        "A second line records the comparison star, the filter wheel, and the exposure count. "
        "A third line records the wind, the humidity, and the moon altitude. "
        "Those repeated lines are the body of the chain and they dominate the five-gram set.\n\n"
    )
    chain_a = shared + (
        "Chain member A adds that the comparison star was HR 5459 and the sky was photometric. "
        "The observer rejected one frame because a satellite crossed the aperture."
    )
    chain_b = shared + (
        "Chain member B adds that the comparison star was HR 5459 and the sky was photometric. "
        "The observer rejected one frame because a satellite crossed the aperture. "
        "Member B keeps almost every five-gram from member A."
    )
    chain_c = shared + (
        "Chain member B adds that the comparison star was HR 5459 and the sky was photometric. "
        "The observer rejected one frame because a satellite crossed the aperture. "
        "Member C copies member B and changes only this closing sentence about the transitive cluster."
    )
    exact = (
        "Messier 31 is also called the Andromeda Galaxy in this authored fixture. "
        "The sentence is duplicated on purpose so the exact-hash stage has a pair to cluster. "
        "Units in the note include 110 km/s for a toy radial velocity and 752 kpc for a toy distance.\n\n"
        "The second paragraph repeats the catalog style NGC 224 and M31 without adding new claims. "
        "Negative values such as -1.5 magnitudes remain in the line so the normalizer can be checked."
    )
    return [
        _doc("openstax_astronomy_2e", "alpha-centauri", "Alpha Centauri fixture", alpha, "2022-03-09T00:00:00Z", "CC-BY-4.0"),
        _doc("wikipedia_en_astronomy", "alpha-centauri-near", "Alpha Centauri near duplicate", alpha_near, "2023-06-01T00:00:00Z", "CC-BY-SA-4.0"),
        _doc("openstax_astronomy_2e", "chain-a", "Chain A", chain_a, "2022-03-09T00:00:00Z", "CC-BY-4.0"),
        _doc("openstax_astronomy_2e", "chain-b", "Chain B", chain_b, "2022-03-09T00:00:00Z", "CC-BY-4.0"),
        _doc("wikipedia_en_astronomy", "chain-c", "Chain C", chain_c, "2023-06-01T00:00:00Z", "CC-BY-SA-4.0"),
        _doc("openstax_astronomy_2e", "m31-exact", "M31 exact", exact, "2022-03-09T00:00:00Z", "CC-BY-4.0"),
        _doc("wikipedia_en_astronomy", "m31-exact-copy", "M31 exact copy", exact, "2023-06-01T00:00:00Z", "CC-BY-SA-4.0"),
        _doc(
            "wikipedia_en_astronomy",
            "post-cutoff",
            "Post cutoff challenge",
            "This post-cutoff fixture mentions a mission status that was invented for the challenge set. "
            "It must not be copied into the training split, and it does not reuse the Alpha Centauri fixture. "
            "The date on the record is after 2024-01-01, so the cutoff filter excludes it.\n\n"
            "A second paragraph keeps the record long enough that only the timestamp, not the quality filter, removes it. "
            "The cloze item would have asked for a fictional launch window that never enters the baseline splits.",
            "2025-05-01T00:00:00Z",
            "CC-BY-SA-4.0",
        ),
        _doc(
            "astronomy_stackexchange",
            "excluded-question",
            "Excluded forum note",
            "Stack Exchange is excluded from the baseline even when the timestamp is old. "
            "This paragraph is original fixture prose and not a copied question. "
            "The source policy keeps the whole source out of train, validation, and test.\n\n"
            "The second paragraph records the exclusion reason source_not_approved. "
            "No five-gram from the approved astronomy fixtures is repeated here on purpose.",
            "2020-01-01T00:00:00Z",
            "CC-BY-SA-4.0",
        ),
    ]


def rejection_documents() -> list[dict]:
    rows = []
    for index in range(50):
        if index % 5 == 0:
            text = "Too short."
        elif index % 5 == 1:
            text = ("nav nav nav\n" * 30) + "\n\n" + ("nav nav nav\n" * 10)
        elif index % 5 == 2:
            text = "<div><span>" + ("{template|x} " * 80) + "</span></div>\n\n" + ("[[link]] " * 40)
        elif index % 5 == 3:
            text = ("42 42 42 99 100 3.14\n" * 20) + "\n\n" + ("7 8 9 10\n" * 10)
        else:
            line = "Repeated footer line for the calibration reject set."
            text = (line + "\n") * 12 + "\n" + (line + "\n") * 8
        rows.append(
            _doc(
                "openstax_astronomy_2e",
                f"reject-{index:02d}",
                f"Reject {index}",
                text,
                "2022-03-09T00:00:00Z",
                "CC-BY-4.0",
            )
        )
    return rows


def volume_documents(count: int = 700, seed: int = 17013) -> list[dict]:
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    rows = []
    for index in range(count):
        unique = [
            "".join(rng.choice(letters) for _ in range(rng.choice((5, 6, 7, 8))))
            for _ in range(12)
        ]
        body = _BODIES[index % len(_BODIES)]
        star = _STARS[index % len(_STARS)]
        velocity = -((index * 17) % 400) - 0.5
        ra_h = index % 24
        ra_m = (index * 3) % 60
        first = (
            f"Survey note {index} follows {body} against {star}. "
            f"The radial velocity is {velocity:.1f} km/s at coordinates {ra_h}h {ra_m}m. "
            f"Catalog NGC {2000 + index} and identifier M{(index % 110) + 1} stay in the sentence. "
            "The Greek letter α is kept, and the mass unit M_⊙ is kept."
        )
        second = (
            "The second paragraph exists so the quality filter sees real prose rather than a fragment. "
            "This is the part of the note that is written for the reader, and it is here because the "
            "language check should see ordinary English rather than a dump of identifiers. "
            "The observer checked the frame and the sky, and the reduction was done with the bias and the flat. "
            f"Synthetic survey tokens {' '.join(unique)} distinguish this note from the others. "
            "Those tokens are authored identifiers for the slice, and they are not observations."
        )
        source = "openstax_astronomy_2e" if index % 2 == 0 else "wikipedia_en_astronomy"
        license_id = "CC-BY-4.0" if source.startswith("openstax") else "CC-BY-SA-4.0"
        published = "2022-03-09T00:00:00Z" if source.startswith("openstax") else "2023-11-01T00:00:00Z"
        rows.append(
            _doc(source, f"survey-{index:04d}", f"Survey {index}", f"{first}\n\n{second}", published, license_id)
        )
    return rows


def build_sample_records() -> list[dict]:
    return hand_written_documents() + volume_documents() + rejection_documents()


def _doc(source: str, record_id: str, title: str, text: str, published_at: str, license_id: str) -> dict:
    from nebulalm.data.acquire import document_id

    license_url = (
        "https://creativecommons.org/licenses/by/4.0/"
        if license_id == "CC-BY-4.0"
        else "https://creativecommons.org/licenses/by-sa/4.0/"
    )
    authors = (
        ["Andrew Fraknoi", "David Morrison", "Sidney Wolff"]
        if source == "openstax_astronomy_2e"
        else ["NebulaLM fixture authors"]
    )
    if source == "astronomy_stackexchange":
        authors = ["Excluded fixture"]
        license_url = "https://creativecommons.org/licenses/by-sa/4.0/"
    return {
        "document_id": document_id(source, record_id),
        "source": source,
        "source_record_id": record_id,
        "title": title,
        "canonical_url": f"https://example.invalid/nebulalm/{source}/{record_id}",
        "revision_id": f"fixture-{record_id}",
        "published_at": published_at,
        "retrieved_at": RETRIEVED_AT,
        "author_attribution": authors,
        "license_id": license_id,
        "license_url": license_url,
        "rights_notes": "Authored fixture. Safe to store in the repository. Not a redistribution of OpenStax or Wikipedia.",
        "raw_text": text,
        "text": text,
    }
