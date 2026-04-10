import numpy as np

from hirag._dedup import build_entity_resolution_candidates, build_neighborhood_map


def test_embedding_match_keeps_distinct_numeric_entities_separate():
    entities = [
        {
            "entity_name": "2004 Summer Olympics",
            "entity_type": "EVENT",
            "description": "The 2004 Summer Olympics were held in Athens.",
            "embedding": np.array([1.0, 0.0, 0.0]),
        },
        {
            "entity_name": "2008 Summer Olympics",
            "entity_type": "EVENT",
            "description": "The 2008 Summer Olympics were held in Beijing.",
            "embedding": np.array([1.0, 0.0, 0.0]),
        },
    ]

    resolved = build_entity_resolution_candidates(
        entities,
        embedding_threshold=0.9,
    )

    assert len(resolved) == 2
    assert {entity["entity_name"] for entity in resolved} == {
        "2004 Summer Olympics",
        "2008 Summer Olympics",
    }


def test_embedding_match_still_merges_aliases_when_identity_tokens_agree():
    entities = [
        {
            "entity_name": "Apollo 11",
            "entity_type": "MISSION",
            "description": "Apollo 11 was the first crewed Moon landing mission.",
            "embedding": np.array([0.0, 1.0, 0.0]),
        },
        {
            "entity_name": "Project Apollo 11",
            "entity_type": "MISSION",
            "description": "Project Apollo 11 landed humans on the Moon.",
            "embedding": np.array([0.0, 1.0, 0.0]),
        },
    ]

    resolved = build_entity_resolution_candidates(
        entities,
        embedding_threshold=0.9,
    )

    assert len(resolved) == 1
    assert resolved[0]["entity_name"] == "Project Apollo 11"
    assert set(resolved[0]["aliases"]) == {"Apollo 11", "Project Apollo 11"}


def test_neighborhood_overlap_supports_embedding_merge():
    entities = [
        {
            "entity_name": "Sherlock Holmes",
            "entity_type": "PERSON",
            "description": "Sherlock Holmes is a detective.",
            "embedding": np.array([0.0, 0.0, 1.0]),
        },
        {
            "entity_name": "Holmes",
            "entity_type": "PERSON",
            "description": "Holmes is the detective consulting at Baker Street.",
            "embedding": np.array([0.0, 0.0, 1.0]),
        },
    ]
    neighborhood_map = build_neighborhood_map(
        [entity["entity_name"] for entity in entities],
        [
            {"src_id": "Sherlock Holmes", "tgt_id": "Dr. Watson"},
            {"src_id": "Sherlock Holmes", "tgt_id": "221B Baker Street"},
            {"src_id": "Holmes", "tgt_id": "Dr. Watson"},
            {"src_id": "Holmes", "tgt_id": "221B Baker Street"},
        ],
    )

    resolved = build_entity_resolution_candidates(
        entities,
        embedding_threshold=0.9,
        neighborhood_map=neighborhood_map,
    )

    assert len(resolved) == 1
    assert resolved[0]["dedup_source"] == "embedding_neighborhood_match"


def test_neighborhood_overlap_blocks_unsupported_embedding_merge():
    entities = [
        {
            "entity_name": "Washington University",
            "entity_type": "ORGANIZATION",
            "description": "Washington University is a research university.",
            "embedding": np.array([0.6, 0.6, 0.0]),
        },
        {
            "entity_name": "George Washington University",
            "entity_type": "ORGANIZATION",
            "description": "George Washington University is a private research university.",
            "embedding": np.array([0.6, 0.6, 0.0]),
        },
    ]
    neighborhood_map = build_neighborhood_map(
        [entity["entity_name"] for entity in entities],
        [
            {"src_id": "Washington University", "tgt_id": "St. Louis"},
            {"src_id": "Washington University", "tgt_id": "Missouri"},
            {"src_id": "George Washington University", "tgt_id": "Washington, D.C."},
            {"src_id": "George Washington University", "tgt_id": "Foggy Bottom"},
        ],
    )

    resolved = build_entity_resolution_candidates(
        entities,
        embedding_threshold=0.9,
        neighborhood_map=neighborhood_map,
        neighborhood_overlap_threshold=0.5,
        neighborhood_min_shared_neighbors=1,
    )

    assert len(resolved) == 2
