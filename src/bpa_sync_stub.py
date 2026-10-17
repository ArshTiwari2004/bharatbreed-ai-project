"""
bpa_sync_stub.py
---------------------------------
Mocked integration with the Bharat Pashudhan App (BPA) data pipeline.

The synopsis explicitly scopes this as "simulated/mocked for the minor-project
scope where direct government access is not available" (section 4.3). This
module is that mock: it defines the exact request/response contract a real
integration would use, and instead of calling a live government endpoint it
appends the "synced" record to a local JSON file so the whole pipeline -
capture -> predict -> confirm -> sync - can be demonstrated end-to-end.

Swapping this for a real integration later means replacing `_mock_post`
with an authenticated HTTPS call to the actual BPA sync endpoint; nothing
else in the pipeline needs to change.
"""

import json
import os
import time

import config

MOCK_BPA_DB_PATH = os.path.join(config.LOGS_DIR, "bpa_sync_mock.jsonl")


def _mock_post(payload):
    """Stands in for `requests.post(BPA_SYNC_ENDPOINT, json=payload, headers=auth_headers)`."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    with open(MOCK_BPA_DB_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")
    return {"status": 200, "synced": True, "record_id": payload["animal_id"]}


def sync_breed_to_bpa(animal_id, breed, confidence, flw_id, farmer_village=None):
    """
    Pushes a confirmed breed label into the (mocked) BPA record for an animal.

    animal_id       : the animal's unique ID / ear-tag number as recorded in BPA
    breed            : final, human-confirmed breed label
    confidence       : model confidence for the accepted label (for audit trail)
    flw_id           : Field Level Worker who confirmed the record
    farmer_village   : optional metadata for traceability
    """
    payload = {
        "animal_id": animal_id,
        "breed": breed,
        "model_confidence": confidence,
        "confirmed_by": flw_id,
        "farmer_village": farmer_village,
        "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_system": "BharatBreedAI",
    }
    response = _mock_post(payload)
    return response


if __name__ == "__main__":
    resp = sync_breed_to_bpa(
        animal_id="UP-2024-00981234",
        breed="Sahiwal",
        confidence=0.91,
        flw_id="FLW_1042",
        farmer_village="Barabanki, UP",
    )
    print("BPA sync response:", resp)
    print("Mock BPA store ->", MOCK_BPA_DB_PATH)
