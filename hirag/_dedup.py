import math
import re
from collections import defaultdict
from typing import Callable

import numpy as np

from ._utils import compute_mdhash_id, split_string_by_multi_markers
from .prompt import GRAPH_FIELD_SEP


_GENERIC_ENTITY_NAMES = {
    "ENTITY",
    "ITEM",
    "OBJECT",
    "SYSTEM",
    "PROCESS",
    "METHOD",
    "MODEL",
    "THING",
}


def normalize_entity_name(name: str) -> str:
    name = (name or "").strip().upper()
    name = re.sub(r"[^A-Z0-9]+", " ", name)
    return " ".join(name.split())


def normalize_text_content(content: str) -> str:
    content = (content or "").strip().lower()
    content = re.sub(r"\s+", " ", content)
    return content


def stable_unique_by(items: list, key_fn: Callable) -> list:
    seen = set()
    unique_items = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        unique_items.append(item)
    return unique_items


def stable_unique_strings(values: list[str]) -> list[str]:
    return stable_unique_by(
        [v.strip() for v in values if isinstance(v, str) and v.strip()],
        lambda value: value,
    )


def split_stored_multi_value(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        merged = []
        for item in value:
            merged.extend(split_stored_multi_value(item))
        return stable_unique_strings(merged)
    if not isinstance(value, str):
        value = str(value)
    return stable_unique_strings(split_string_by_multi_markers(value, [GRAPH_FIELD_SEP]))


def join_multi_value(values: list[str]) -> str:
    return GRAPH_FIELD_SEP.join(stable_unique_strings(values))


def merge_alias_lists(*alias_groups) -> list[str]:
    merged = []
    for alias_group in alias_groups:
        merged.extend(split_stored_multi_value(alias_group))
    return stable_unique_strings(merged)


def merge_source_ids(*source_groups) -> str:
    merged = []
    for source_group in source_groups:
        merged.extend(split_stored_multi_value(source_group))
    return join_multi_value(merged)


def merge_descriptions(*description_groups) -> str:
    merged = []
    for description_group in description_groups:
        merged.extend(split_stored_multi_value(description_group))
    return join_multi_value(merged)


def build_canonical_id(entity_name: str, layer: int = 0) -> str:
    normalized_name = normalize_entity_name(entity_name)
    return compute_mdhash_id(f"{layer}:{normalized_name}", prefix="canon-")


def cosine_similarity(left, right) -> float:
    if left is None or right is None:
        return -1.0
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    denom = np.linalg.norm(left_arr) * np.linalg.norm(right_arr)
    if denom == 0:
        return -1.0
    return float(np.dot(left_arr, right_arr) / denom)


def are_entity_types_compatible(left: str, right: str) -> bool:
    left = (left or "").strip().upper()
    right = (right or "").strip().upper()
    if not left or not right:
        return True
    if left == right:
        return True
    unknown_values = {'"UNKNOWN"', "UNKNOWN", "OTHER", "MISC"}
    return left in unknown_values or right in unknown_values


def is_generic_entity_name(name: str) -> bool:
    normalized_name = normalize_entity_name(name)
    return (
        not normalized_name
        or len(normalized_name) <= 2
        or normalized_name in _GENERIC_ENTITY_NAMES
    )


def has_meaningful_name_overlap(left: str, right: str) -> bool:
    left_tokens = set(normalize_entity_name(left).split())
    right_tokens = set(normalize_entity_name(right).split())
    if not left_tokens or not right_tokens:
        return False
    return bool(left_tokens & right_tokens)


def have_description_overlap(left: str, right: str) -> bool:
    left_tokens = set(normalize_text_content(left).split())
    right_tokens = set(normalize_text_content(right).split())
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    minimum = max(1, math.ceil(min(len(left_tokens), len(right_tokens)) * 0.25))
    return len(overlap) >= minimum


def prefer_display_name(left: str, right: str) -> str:
    candidates = stable_unique_strings([left, right])
    return sorted(
        candidates,
        key=lambda value: (
            -len(normalize_entity_name(value)),
            -len(value),
            value,
        ),
    )[0]


def _copy_entity(entity: dict) -> dict:
    copied = dict(entity)
    if copied.get("embedding") is not None:
        copied["embedding"] = np.asarray(copied["embedding"], dtype=float)
    aliases = merge_alias_lists(copied.get("aliases"), copied.get("entity_name"))
    copied["aliases"] = aliases
    copied["layer"] = int(copied.get("layer", 0) or 0)
    copied["dedup_source"] = copied.get("dedup_source", "exact_name")
    copied["canonical_id"] = copied.get(
        "canonical_id",
        build_canonical_id(copied.get("entity_name", ""), copied["layer"]),
    )
    return copied


def _merge_entity_records(canonical_entity: dict, candidate_entity: dict, dedup_source: str) -> dict:
    merged = dict(canonical_entity)
    merged["entity_name"] = prefer_display_name(
        canonical_entity["entity_name"], candidate_entity["entity_name"]
    )
    merged["entity_type"] = canonical_entity.get("entity_type") or candidate_entity.get(
        "entity_type"
    )
    merged["aliases"] = merge_alias_lists(
        canonical_entity.get("aliases"),
        candidate_entity.get("aliases"),
        canonical_entity.get("entity_name"),
        candidate_entity.get("entity_name"),
    )
    merged["description"] = merge_descriptions(
        canonical_entity.get("description"), candidate_entity.get("description")
    )
    merged["source_id"] = merge_source_ids(
        canonical_entity.get("source_id"), candidate_entity.get("source_id")
    )
    merged["layer"] = int(
        min(canonical_entity.get("layer", 0), candidate_entity.get("layer", 0))
    )
    merged["dedup_source"] = dedup_source
    merged["canonical_id"] = build_canonical_id(merged["entity_name"], merged["layer"])

    left_embedding = canonical_entity.get("embedding")
    right_embedding = candidate_entity.get("embedding")
    if left_embedding is not None and right_embedding is not None:
        merged["embedding"] = (
            np.asarray(left_embedding, dtype=float) + np.asarray(right_embedding, dtype=float)
        ) / 2.0
    elif right_embedding is not None:
        merged["embedding"] = np.asarray(right_embedding, dtype=float)
    else:
        merged["embedding"] = left_embedding

    return merged


def _match_candidate_to_group(
    candidate: dict,
    groups: list[dict],
    embedding_threshold: float,
    same_layer_only: bool,
    lexical_gate_for_embeddings: bool,
) -> tuple[int | None, str | None]:
    candidate_layer = int(candidate.get("layer", 0) or 0)
    candidate_name = candidate.get("entity_name", "")
    candidate_normalized = normalize_entity_name(candidate_name)
    candidate_description = candidate.get("description", "")
    candidate_type = candidate.get("entity_type", "")

    for index, group in enumerate(groups):
        group_layer = int(group.get("layer", 0) or 0)
        if same_layer_only and candidate_layer != group_layer:
            continue
        if not are_entity_types_compatible(candidate_type, group.get("entity_type", "")):
            continue

        if candidate_normalized == normalize_entity_name(group.get("entity_name", "")):
            return index, "normalized_name"

        similarity = cosine_similarity(candidate.get("embedding"), group.get("embedding"))
        if similarity < embedding_threshold:
            continue
        if is_generic_entity_name(candidate_name) or is_generic_entity_name(group.get("entity_name", "")):
            continue
        if lexical_gate_for_embeddings and not (
            has_meaningful_name_overlap(candidate_name, group.get("entity_name", ""))
            or have_description_overlap(candidate_description, group.get("description", ""))
        ):
            continue
        return index, "embedding_match"

    return None, None


def build_entity_resolution_candidates(
    all_entities: list[dict],
    embedding_threshold: float,
    same_layer_only: bool = True,
    lexical_gate_for_embeddings: bool = True,
) -> list[dict]:
    groups = []
    for entity in all_entities:
        candidate = _copy_entity(entity)
        match_index, match_source = _match_candidate_to_group(
            candidate,
            groups,
            embedding_threshold=embedding_threshold,
            same_layer_only=same_layer_only,
            lexical_gate_for_embeddings=lexical_gate_for_embeddings,
        )
        if match_index is None:
            groups.append(candidate)
            continue
        groups[match_index] = _merge_entity_records(
            groups[match_index], candidate, match_source
        )
    return groups


def resolve_entities_to_canonical(
    all_entities: list[dict],
    embedding_threshold: float,
    same_layer_only: bool = True,
    lexical_gate_for_embeddings: bool = True,
) -> tuple[list[dict], dict[str, str]]:
    canonical_entities = build_entity_resolution_candidates(
        all_entities,
        embedding_threshold=embedding_threshold,
        same_layer_only=same_layer_only,
        lexical_gate_for_embeddings=lexical_gate_for_embeddings,
    )
    alias_map = {}
    normalized_alias_map = {}
    for entity in canonical_entities:
        canonical_name = entity["entity_name"]
        for alias in merge_alias_lists(entity.get("aliases"), canonical_name):
            alias_map[alias] = canonical_name
            normalized_alias_map[normalize_entity_name(alias)] = canonical_name
    return canonical_entities, alias_map | normalized_alias_map


def rewrite_chunk_entity_lists(
    context_entities: dict[str, list[str]],
    alias_map: dict[str, str],
) -> dict[str, list[str]]:
    rewritten = {}
    for chunk_id, entity_names in context_entities.items():
        rewritten[chunk_id] = stable_unique_by(
            [alias_map.get(name, alias_map.get(normalize_entity_name(name), name)) for name in entity_names],
            lambda value: normalize_entity_name(value),
        )
    return rewritten


def rewrite_relation_endpoints(
    relation_results: list[tuple[dict, dict]],
    alias_map: dict[str, str],
) -> list[tuple[dict, dict]]:
    rewritten_results = []
    for maybe_nodes, maybe_edges in relation_results:
        rewritten_nodes = defaultdict(list)
        rewritten_edges = defaultdict(list)

        for entity_name, items in maybe_nodes.items():
            canonical_name = alias_map.get(
                entity_name, alias_map.get(normalize_entity_name(entity_name), entity_name)
            )
            for item in items:
                rewritten_item = dict(item)
                rewritten_item["entity_name"] = canonical_name
                rewritten_item["aliases"] = merge_alias_lists(
                    rewritten_item.get("aliases"),
                    entity_name,
                )
                rewritten_nodes[canonical_name].append(rewritten_item)

        for edge_key, items in maybe_edges.items():
            for item in items:
                src_id = alias_map.get(
                    item["src_id"],
                    alias_map.get(normalize_entity_name(item["src_id"]), item["src_id"]),
                )
                tgt_id = alias_map.get(
                    item["tgt_id"],
                    alias_map.get(normalize_entity_name(item["tgt_id"]), item["tgt_id"]),
                )
                rewritten_item = dict(item)
                rewritten_item["src_id"] = src_id
                rewritten_item["tgt_id"] = tgt_id
                rewritten_item["canonical_source_id"] = build_canonical_id(
                    src_id, int(rewritten_item.get("layer", 0) or 0)
                )
                rewritten_item["canonical_target_id"] = build_canonical_id(
                    tgt_id, int(rewritten_item.get("layer", 0) or 0)
                )
                rewritten_item["dedup_source"] = rewritten_item.get(
                    "dedup_source", "canonical_rewrite"
                )
                rewritten_edges[tuple(sorted((src_id, tgt_id)))].append(rewritten_item)
        rewritten_results.append((dict(rewritten_nodes), dict(rewritten_edges)))
    return rewritten_results


def fingerprint_content(content: str) -> str:
    return normalize_text_content(content)


def stable_group_counts(items: list[dict], group_key_fn: Callable[[dict], str]) -> dict[str, int]:
    counts = defaultdict(int)
    for item in items:
        counts[group_key_fn(item)] += 1
    return dict(counts)
