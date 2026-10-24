import json
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_registered_protocol_matches_schema():
    schema = json.loads((ROOT / "schemas" / "experiment.schema.json").read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "experiments" / "protocols" / "EXP-TOK.yaml").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(protocol)
