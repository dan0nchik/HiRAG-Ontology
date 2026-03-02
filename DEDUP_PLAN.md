# Deduplication Implementation Plan for HiRAG

## Goal

Introduce deduplication into HiRAG in a way that:

- reduces redundant entities, relations, chunks, and retrieved contexts,
- preserves the hierarchical design described in `PAPER.tex`,
- avoids collapsing valid higher-layer abstractions into lower-layer entities,
- improves retrieval quality and token efficiency without destabilizing the current pipeline.

This plan is based on the current implementation in:

- `hirag/_op.py`
- `hirag/_cluster_utils.py`
- `hirag/_storage/gdb_networkx.py`
- `hirag/hirag.py`
- `hirag/base.py`


## Current State Summary

The codebase already performs a limited form of exact deduplication:

- entities are merged by exact `entity_name` in `_merge_nodes_then_upsert`,
- relations are merged by exact endpoint pair in `_merge_edges_then_upsert`,
- hierarchical clustering removes repeated summary nodes only by exact `entity_name`,
- retrieval does not apply any semantic or diversity-based deduplication.

This leaves four main redundancy sources:

1. same real-world entity extracted under different names,
2. same summary concept regenerated across clusters or layers,
3. duplicated or near-duplicated retrieved contexts at query time,
4. unstable bridge paths caused by non-deterministic or order-breaking selection logic.


## Design Principles

1. Deduplicate aggressively only within the same semantic level.
   Higher-layer summary entities are abstractions, not necessarily duplicates of lower-layer entities.

2. Separate canonicalization from graph structure.
   Canonical identity should be metadata, not a new graph topology primitive unless later proven useful.

3. Prefer deterministic heuristics first.
   Start with lexical normalization, stable ordering, embedding similarity, and source overlap before adding LLM-based entity resolution.

4. Preserve reversibility.
   Keep alias/provenance metadata so merged entities can still be traced back to extracted mentions and source chunks.

5. Stage rollout by risk.
   Start with low-risk retrieval-time dedup and indexing scaffolding, then add semantic entity resolution.


## Scope

### In Scope

- entity canonicalization during indexing,
- summary-entity dedup within hierarchical clustering,
- retrieval-time redundancy suppression for entities, communities, edges, paths, and chunks,
- deterministic bridge path construction,
- metadata additions needed to support deduplication and evaluation.

### Out of Scope for Initial Rollout

- changing the paper’s overall HiIndex / HiRetrieval design,
- deduplicating parent-child hierarchy edges as if they were aliases,
- adding new external storage systems,
- training a learned entity-resolution model,
- large prompt redesign for extraction or summarization.


## High-Level Architecture Changes

### New Concept: Canonical Entity Identity

Introduce a canonical identity layer for extracted entities.

Each graph node should retain:

- `entity_name`: current display name used as graph node id for now,
- `canonical_id`: stable dedup identity,
- `aliases`: alternate names merged into the canonical entity,
- `layer`: hierarchy layer if available,
- `dedup_source`: how the merge was decided, such as `exact_name`, `normalized_name`, `embedding_match`, `manual`,
- `source_id`: existing chunk provenance,
- `description`: merged/canonicalized description,
- `entity_type`: merged dominant type.

The canonical layer should initially be metadata only. The graph may still use canonical display names as node ids, but the implementation should not assume `entity_name == canonical_id`.


## Implementation Phases

## Phase 0: Low-Risk Retrieval Optimizations

### Objective

Reduce redundant context without changing indexing semantics yet.

### Changes

1. Fix ordered dedup of bridge key entities.
   Current code converts key entities to `set(...)`, which destroys ranking/order before shortest-path chaining.

   Target locations:

   - `hirag/_op.py` in `_build_hierarchical_query_context`
   - `hirag/_op.py` in `_build_hibridge_query_context`

   Replace:

   - unordered `set(...)` dedup

   With:

   - stable first-seen dedup preserving ranking order across communities.

2. Add retrieval-time diversity reranking.

   Apply MMR-style or quota-based dedup to:

   - retrieved entities,
   - communities,
   - text chunks,
   - reasoning-path edges.

3. Add exact duplicate suppression by content string before prompt assembly.

   This is especially useful for:

   - community reports with nearly identical content,
   - repeated chunk content from overlapping provenance,
   - relation descriptions merged from multiple paths.

### New Parameters

Add to `QueryParam`:

- `enable_dedup: bool = True`
- `entity_diversity_lambda: float`
- `max_entities_per_canonical: int = 1`
- `max_chunks_per_canonical_group: int`
- `dedup_communities_by_content: bool = True`
- `dedup_paths_by_edge_pair: bool = True`

### Expected Benefit

- immediate token savings,
- cleaner bridge context,
- more stable retrieval,
- no graph migration required.


## Phase 1: Canonicalization Scaffolding in Indexing

### Objective

Prepare the data model and storage layer for semantic deduplication without yet changing merge logic too aggressively.

### Changes

1. Extend node metadata written by `_merge_nodes_then_upsert`.

   Target:

   - `hirag/_op.py`

   Add fields:

   - `canonical_id`
   - `aliases`
   - `layer`
   - `dedup_source`

2. Extend vector DB metadata for entities.

   Current `entities_vdb` stores only `entity_name`.

   Update construction in `hirag/hirag.py`:

   - include `canonical_id`
   - include `entity_type`
   - include `layer`

   Update upsert payloads in `hirag/_op.py` accordingly.

3. Add utility helpers for normalization and stable dedup.

   Likely in:

   - `hirag/_utils.py` or a new `hirag/_dedup.py`

   Suggested helpers:

- `normalize_entity_name(name: str) -> str`
- `stable_unique_by(items, key_fn)`
- `merge_alias_lists(...)`
- `merge_source_ids(...)`
- `merge_descriptions(...)`

4. Keep canonicalization decisions observable.

   Every merged entity should retain enough metadata to explain:

   - which aliases were merged,
   - which sources contributed,
   - which rule triggered the merge.

### Expected Benefit

- prepares code for semantic dedup,
- keeps graph/debuggability intact,
- minimizes refactor risk.


## Phase 2: Index-Time Entity Resolution Before Relation Extraction

### Objective

Canonicalize extracted entities before relation extraction so relation prompts use canonical entity names and avoid duplicate graph nodes.

### Why This Stage Matters

The current hierarchical extraction flow is already split into:

- entity extraction,
- entity embedding,
- relation extraction conditioned on per-chunk entity lists.

This is the best insertion point for dedup because relations can be rewritten to canonical ids before graph insertion.

### Target Flow Changes

Current:

1. extract entities from chunks,
2. embed extracted entities,
3. use raw per-chunk entity lists for relation extraction,
4. merge graph nodes by exact name only.

Planned:

1. extract entities from chunks,
2. build candidate dedup groups,
3. create canonical entities,
4. rewrite per-chunk entity lists to canonical names,
5. run relation extraction using canonical names,
6. rewrite returned relation endpoints to canonical ids,
7. merge canonical nodes/edges into graph.

### Candidate Matching Heuristics

Use a conservative staged matcher:

1. exact normalized name match,
2. exact normalized name + compatible type,
3. high embedding similarity above threshold,
4. embedding similarity + source overlap,
5. optional LLM tie-break only for ambiguous candidates.

### Important Guardrails

Do not merge if:

- entity types clearly conflict,
- names are generic and underspecified,
- embedding similarity is high but source contexts indicate different senses,
- one candidate is a summary abstraction and the other is a concrete lower-layer entity.

### Deliverables

1. New canonicalization pass in `extract_hierarchical_entities`.
2. Alias map:

   - raw extracted name -> canonical entity name / id.

3. Rewritten `context_entities` per chunk before relation extraction.
4. Rewritten relation endpoints before final graph upsert.

### Suggested New Functions

- `build_entity_resolution_candidates(all_entities, thresholds, config)`
- `resolve_entities_to_canonical(candidates, config)`
- `rewrite_chunk_entity_lists(context_entities, alias_map)`
- `rewrite_relation_endpoints(relation_results, alias_map)`

### Expected Benefit

- prevents alias duplication early,
- reduces graph noise,
- improves community quality,
- improves retrieval precision downstream.


## Phase 3: Summary-Entity Dedup Inside Hierarchical Clustering

### Objective

Prevent upper layers from accumulating semantically repeated summary entities.

### Current Problem

`Hierarchical_Clustering.perform_clustering` only removes duplicates by exact `entity_name`.
This is too weak because two clusters may independently generate equivalent summary concepts with different wording.

### Changes

1. After each layer’s summary generation, run same-layer dedup on generated summary entities.

2. Use a stricter policy than base entity resolution:

   - same layer only,
   - same or compatible `entity_type`,
   - high embedding similarity,
   - similar normalized names or strong description overlap.

3. Rewrite generated summary relations so they point to the surviving canonical summary entity.

4. Replace random cluster downsampling with deterministic representative selection.

   Current implementation randomly samples nodes when a cluster is too long.
   This makes upper-layer summaries unstable and undermines dedup reproducibility.

   Replace with one of:

   - top-central exemplars by similarity to cluster centroid,
   - top-diverse exemplars with deterministic ordering,
   - longest-description or highest-degree representatives only as fallback.

### Expected Benefit

- fewer duplicate semantic hubs,
- more stable hierarchy across runs,
- cleaner cross-layer community structure.


## Phase 4: Retrieval-Time Canonical and Redundancy Control

### Objective

Make the retriever aware of canonical identity and avoid redundant context assembly.

### Entity Retrieval

After `entities_vdb.query(...)`, perform:

1. stable score sort,
2. group by `canonical_id`,
3. keep best representative per canonical group,
4. optionally allow `max_entities_per_canonical > 1` for ambiguous queries,
5. optionally apply MMR over canonical representatives.

### Community Retrieval

Current community ranking counts how often retrieved entities point into each community.
This can still over-represent the same canonical concept.

Improve ranking by:

- counting unique canonical entities, not raw entity hits,
- suppressing near-duplicate community reports by normalized report content,
- optionally preferring communities that contribute novel entities.

### Chunk Retrieval

Current chunk selection is keyed by chunk id and relation counts, but many chunks may contain the same alias set or near-duplicate overlapping text.

Add suppression by:

- exact content match,
- normalized content fingerprint,
- excessive overlap in canonical entities covered,
- optional max chunks per canonical group.

### Reasoning Path Retrieval

Deduplicate:

- repeated nodes in paths while preserving order,
- repeated edges across path subgraphs,
- repeated relation descriptions in the final bridge context.

### Prompt Assembly

Before converting sections to CSV:

- remove duplicate rows by stable key,
- enforce deterministic ordering,
- log counts before and after dedup.

### Expected Benefit

- lower prompt redundancy,
- better use of token budget,
- more informative and less repetitive contexts.


## Phase 5: Optional LLM-Assisted Entity Resolution

### Objective

Handle ambiguous near-duplicates that deterministic heuristics cannot safely merge.

### When to Use

Only for candidate pairs/groups that satisfy:

- high semantic similarity,
- conflicting or uncertain lexical signals,
- insufficient type/source evidence for automatic merge.

### Prompt Task

Ask the LLM to decide whether candidates refer to:

- the same real-world entity,
- related but distinct entities,
- an abstraction/concept vs an instance.

### Constraints

- use only as tie-breaker,
- cache responses,
- never use this in the hot retrieval path,
- keep deterministic heuristics as default.

### Rollout

This should be feature-flagged and disabled by default until offline evaluation proves value.


## Data Model Changes

## Graph Node Schema

Add or standardize the following fields:

- `entity_name: str`
- `canonical_id: str`
- `aliases: str` or JSON list
- `entity_type: str`
- `description: str`
- `source_id: str`
- `layer: int`
- `dedup_source: str`

Optional later:

- `canonical_score`
- `dedup_confidence`

## Graph Edge Schema

Keep existing fields and consider adding:

- `canonical_source_id`
- `canonical_target_id`
- `dedup_source`

This is optional if edge endpoints are already rewritten to canonical ids before insertion.

## Vector DB Metadata

Entity VDB should include:

- `entity_name`
- `canonical_id`
- `entity_type`
- `layer`

This is needed for query-time canonical grouping.


## File-Level Work Breakdown

## `hirag/_op.py`

Primary changes:

- add dedup utilities import,
- insert canonicalization pass after entity extraction and embedding,
- rewrite `context_entities`,
- rewrite relation endpoints,
- expand node metadata in `_merge_nodes_then_upsert`,
- expand VDB upsert payload,
- add retrieval-time dedup helpers for:
  - communities,
  - text units,
  - edges,
  - path sections,
- fix ordered key-entity dedup in bridge retrieval functions.

## `hirag/_cluster_utils.py`

Primary changes:

- add same-layer summary dedup,
- replace random truncation with deterministic representative selection,
- carry `layer` metadata into generated summary entities,
- optionally annotate generated entities with `dedup_source="summary_layer"`.

## `hirag/hirag.py`

Primary changes:

- extend `HiRAG` config with dedup-related thresholds and toggles,
- expand `entities_vdb` metadata fields,
- expose feature flags for staged rollout.

Suggested config flags:

- `enable_entity_dedup`
- `enable_summary_dedup`
- `enable_retrieval_dedup`
- `entity_name_normalization`
- `entity_embedding_merge_threshold`
- `summary_embedding_merge_threshold`
- `retrieval_mmr_lambda`

## `hirag/base.py`

Primary changes:

- add dedup-related query parameters to `QueryParam`.

## `hirag/_storage/gdb_networkx.py`

Potential changes:

- no major structural changes required,
- ensure new node metadata fields survive read/write round-trip,
- optionally add helper methods if canonical grouping at graph level becomes necessary.

## `hirag/_utils.py` or new `hirag/_dedup.py`

Recommended new module for:

- normalization,
- candidate generation,
- canonical selection,
- stable unique operations,
- content fingerprinting,
- optional MMR helpers.


## Detailed Algorithm Proposal

## A. Entity Canonicalization Algorithm

Input:

- `all_entities` from extraction,
- embeddings,
- per-entity type,
- per-entity source ids,
- optional layer info.

Steps:

1. Normalize each entity name.
2. Build blocking keys:
   - normalized name,
   - normalized name + type,
   - optional first-token/last-token blocks for efficiency.
3. Within each block, evaluate candidate merge pairs.
4. Score each pair using:
   - normalized name equality,
   - type compatibility,
   - embedding cosine similarity,
   - source overlap,
   - description token overlap.
5. Form merge groups using connected components only for confident edges.
6. Select canonical representative per group.
   Suggested policy:
   - most frequent source coverage,
   - highest description richness,
   - shortest stable normalized display name as tie-breaker.
7. Merge group metadata.
8. Emit:
   - canonical entity records,
   - alias map,
   - merge provenance.

## B. Retrieval Diversity Algorithm

Input:

- ranked retrieved items,
- item embedding or content fingerprint,
- canonical grouping metadata.

Steps:

1. Drop exact duplicates.
2. Group by canonical id if available.
3. Keep top representative from each group.
4. Apply MMR or novelty-aware rerank over survivors.
5. Truncate to token budget.

## C. Summary-Layer Dedup Algorithm

Input:

- summary entities produced at a given layer,
- embeddings,
- types,
- descriptions.

Steps:

1. block by normalized name and type,
2. compare embeddings and description overlap,
3. merge only within same layer,
4. rewrite summary relations to canonical layer nodes,
5. continue clustering using deduplicated summaries only.


## Rollout Order

Recommended order:

1. Phase 0: retrieval-only dedup and bridge ordering fix.
2. Phase 1: metadata scaffolding and config plumbing.
3. Phase 2: pre-relation entity canonicalization.
4. Phase 3: summary-layer dedup and deterministic representative selection.
5. Phase 4: canonical-aware retrieval ranking.
6. Phase 5: optional LLM tie-breaker.

This order keeps risk low and makes regressions easier to isolate.


## Evaluation Plan

## Offline Structural Metrics

Measure before and after dedup:

- number of graph nodes,
- number of graph edges,
- average aliases per canonical entity,
- number of communities,
- community size distribution,
- cluster sparsity by layer,
- number of summary entities per layer,
- retrieval context token count.

## Retrieval Metrics

For held-out queries, measure:

- unique canonical entities in retrieved set,
- duplicate ratio in retrieved entities,
- duplicate ratio in retrieved chunks,
- duplicate ratio in community reports,
- bridge path edge redundancy,
- token budget utilization.

## Answer Quality

Reuse existing evaluation scripts and compare:

- `hi`
- `hi_nobridge`
- dedup-enabled `hi`
- dedup-enabled `hi_nobridge`

Check whether dedup:

- improves or preserves win rates,
- reduces context tokens,
- improves bridge coherence.

## Stability Tests

Especially for clustering:

- repeated indexing runs on same input should produce near-identical summary layers,
- bridge paths should be deterministic given same graph and query,
- retrieval section ordering should be stable.


## Testing Strategy

## Unit Tests

Add focused tests for:

- name normalization,
- stable dedup preserving order,
- canonical selection,
- alias merge behavior,
- relation endpoint rewriting,
- chunk/content fingerprint dedup,
- ordered bridge key selection.

## Integration Tests

Use a small controlled corpus with known aliases, such as:

- `AWS`, `Amazon Web Services`,
- `Amazon`, `AMAZON.COM`,
- distinct senses like `Amazon` company vs rainforest.

Validate:

- correct merges,
- correct non-merges,
- relation rewrites,
- cleaner retrieval output.

## Regression Tests

Ensure:

- baseline insertion still succeeds with dedup disabled,
- query modes `hi`, `hi_local`, `hi_global`, `hi_bridge`, `hi_nobridge` still function,
- stored graph and VDB files remain readable.


## Risks and Mitigations

## Risk 1: False Positive Entity Merges

Problem:

- semantically similar but distinct entities may be merged.

Mitigation:

- conservative thresholds,
- require compatible types,
- require source/context overlap for borderline matches,
- keep dedup feature-flagged,
- log merge provenance for inspection.

## Risk 2: Collapsing Abstractions with Instances

Problem:

- higher-layer summary entities may be mistaken for duplicates of lower-layer nodes.

Mitigation:

- never merge across layers in initial implementation,
- require same-layer matching for summary dedup.

## Risk 3: Retrieval Over-Pruning

Problem:

- aggressive dedup may remove useful corroborating evidence.

Mitigation:

- keep per-canonical quotas configurable,
- compare answer quality and token savings,
- allow `max_entities_per_canonical > 1`.

## Risk 4: Pipeline Complexity

Problem:

- dedup introduces many new branches and metadata dependencies.

Mitigation:

- isolate logic in `hirag/_dedup.py`,
- add feature flags,
- roll out in phases with tests after each phase.

## Risk 5: Instability from Random Cluster Truncation

Problem:

- current clustering can vary across runs because oversized clusters are randomly sampled.

Mitigation:

- replace random selection with deterministic representative selection before dedup rollout.


## Recommended Minimum Viable Dedup Release

The smallest high-value release should include:

1. stable ordered dedup in bridge path key-entity selection,
2. retrieval-time dedup for entities/chunks/communities,
3. canonical metadata scaffolding,
4. conservative exact and normalized-name canonicalization before relation extraction.

This gives immediate practical benefit while avoiding the highest-risk semantic merges.


## Recommended Follow-Up Release

After the minimum viable release is stable, implement:

1. embedding-assisted entity resolution,
2. same-layer summary dedup,
3. deterministic cluster representative selection,
4. optional LLM tie-breaker for ambiguous merges.


## Suggested Implementation Sequence by Commit

1. add `DEDUP_PLAN.md`
2. add dedup config flags and `QueryParam` options
3. add utility module for normalization and stable unique
4. fix bridge key-entity order-preserving dedup
5. add retrieval-time exact duplicate suppression
6. add canonical metadata fields to node/VDB payloads
7. add conservative indexing-time canonicalization before relation extraction
8. add tests for alias rewrite and retrieval dedup
9. replace random cluster truncation with deterministic representative selection
10. add same-layer summary dedup
11. run evaluation on one dataset, then all datasets


## Definition of Done

This work is complete when:

- dedup can be enabled/disabled by config,
- graph nodes and VDB entries expose canonical identity metadata,
- extracted aliases are merged conservatively before graph insertion,
- retrieval contexts show materially reduced redundancy,
- bridge path construction is stable and ordered,
- summary-layer duplication is reduced without collapsing hierarchy,
- offline and evaluation scripts show no major regression in answer quality.
