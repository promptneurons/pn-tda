"""Realistic sprint fixtures for corpus-level validation.

Sprint 26033: 10 documents in 3 clusters + 2 isolated
Sprint 26034: 8 documents, bridges clusters A+B (thread maturation)
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

_SCHEMA = """
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    sprint_id TEXT,
    sprint_year INTEGER,
    sprint_month INTEGER,
    sprint_in_month INTEGER,
    is_focus_hub BOOLEAN DEFAULT 0,
    title_dtg TEXT,
    parent_group TEXT
);

CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    signal_type TEXT NOT NULL,
    signal_value TEXT NOT NULL,
    line_number INTEGER,
    context TEXT,
    UNIQUE(doc_id, signal_type, signal_value, line_number)
);

CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    target_path TEXT NOT NULL,
    target_doc_id TEXT,
    edge_type TEXT NOT NULL,
    line_number INTEGER,
    UNIQUE(source_doc_id, target_path, edge_type, line_number)
);
"""

# --- Sprint 26033: 10 docs, 3 clusters + 2 isolated ---
#
# Cluster A (daynote chain): hub — note1 — note2 — note3
# Cluster B (CEC refs):      ref1 — ref2
# Cluster C (DDC refs):      ref3 — ref4
# Isolated:                   iso1, iso2

_DOCS_26033 = """
INSERT INTO documents VALUES
    ('hub1',  'Sprint/26033/hub-index.md',    'hub-index.md',    '26033', 2026, 3, 3, 1, '250800TMAR26', 'FY2026'),
    ('n1',    'Sprint/26033/note-0325.md',    'note-0325.md',    '26033', 2026, 3, 3, 0, '250930TMAR26', 'FY2026'),
    ('n2',    'Sprint/26033/note-0326.md',    'note-0326.md',    '26033', 2026, 3, 3, 0, '261015TMAR26', 'FY2026'),
    ('n3',    'Sprint/26033/note-0327.md',    'note-0327.md',    '26033', 2026, 3, 3, 0, '271400TMAR26', 'FY2026'),
    ('r1',    'Sprint/26033/ref-tda.md',      'ref-tda.md',      '26033', 2026, 3, 3, 0, NULL, 'FY2026'),
    ('r2',    'Sprint/26033/ref-homology.md',  'ref-homology.md', '26033', 2026, 3, 3, 0, NULL, 'FY2026'),
    ('r3',    'Sprint/26033/ref-finance.md',   'ref-finance.md',  '26033', 2026, 3, 3, 0, NULL, 'FY2026'),
    ('r4',    'Sprint/26033/ref-ontology.md',  'ref-ontology.md', '26033', 2026, 3, 3, 0, NULL, 'FY2026'),
    ('iso1',  'Sprint/26033/scratch-pad.md',   'scratch-pad.md',  '26033', 2026, 3, 3, 0, '281200TMAR26', 'FY2026'),
    ('iso2',  'Sprint/26033/external-ref.md',  'external-ref.md', '26033', 2026, 3, 3, 0, NULL, 'FY2026');
"""

_SIGNALS_26033 = """
INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
    -- Hub: broad signal coverage
    ('hub1', 'classification_ref', 'CEC B512', 3),
    ('hub1', 'classification_ref', 'JDN 3-01', 8),
    ('hub1', 'classification_ref', 'DDC 332.6', 12),
    ('hub1', 'dtg_content', '250800TMAR26', 1),

    -- Note1: shares CEC + JDN with hub
    ('n1', 'classification_ref', 'CEC B512', 5),
    ('n1', 'classification_ref', 'JDN 3-01', 10),
    ('n1', 'dtg_content', '250930TMAR26', 1),

    -- Note2: shares CEC with hub, adds SUMO
    ('n2', 'classification_ref', 'CEC B512', 4),
    ('n2', 'classification_ref', 'SUMO Process', 9),
    ('n2', 'dtg_content', '261015TMAR26', 1),

    -- Note3: shares SUMO with note2
    ('n3', 'classification_ref', 'SUMO Process', 6),
    ('n3', 'dtg_content', '271400TMAR26', 1),

    -- Ref1: CEC cluster
    ('r1', 'classification_ref', 'CEC B512', 2),
    ('r1', 'classification_ref', 'CEC B514', 7),

    -- Ref2: CEC cluster (shares CEC B514 with ref1)
    ('r2', 'classification_ref', 'CEC B514', 3),
    ('r2', 'classification_ref', 'CEC B516', 8),

    -- Ref3: DDC cluster
    ('r3', 'classification_ref', 'DDC 332.6', 4),
    ('r3', 'classification_ref', 'DDC 338.1', 9),

    -- Ref4: DDC cluster (shares DDC 338.1 with ref3)
    ('r4', 'classification_ref', 'DDC 338.1', 5),
    ('r4', 'classification_ref', 'DDC 510.0', 11),

    -- Iso1: unique signal
    ('iso1', 'classification_ref', 'Jefflib 7.2', 3),
    ('iso1', 'dtg_content', '281200TMAR26', 1),

    -- Iso2: unique signal
    ('iso2', 'classification_ref', 'MEC 4400', 2);
"""

_EDGES_26033 = """
INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
    -- Cluster A: hub — note1 — note2 — note3 (wiki_link chain)
    ('hub1', 'Sprint/26033/note-0325.md', 'n1', 'wiki_link', 5),
    ('n1', 'Sprint/26033/hub-index.md', 'hub1', 'uplink', 1),
    ('n1', 'Sprint/26033/note-0326.md', 'n2', 'wiki_link', 12),
    ('n2', 'Sprint/26033/hub-index.md', 'hub1', 'uplink', 1),
    ('n2', 'Sprint/26033/note-0327.md', 'n3', 'wiki_link', 8),
    ('n3', 'Sprint/26033/note-0326.md', 'n2', 'wiki_link', 4),

    -- Cluster B: ref1 — ref2 (obsidian_link)
    ('r1', 'ref-homology.md', 'r2', 'obsidian_link', 15),
    ('r2', 'ref-tda.md', 'r1', 'obsidian_link', 10),

    -- Cluster C: ref3 — ref4 (wiki_link)
    ('r3', 'Sprint/26033/ref-ontology.md', 'r4', 'wiki_link', 7),

    -- Iso2: external edge only (not resolved)
    ('iso2', 'https://dev.azure.com/pn/wiki/external', NULL, 'azure_devops_link', 3);
"""

# --- Sprint 26034: 8 docs, bridges clusters A+B ---
# Carries forward hub1, n1, r1 context; adds new docs + bridging edges

_DOCS_26034 = """
INSERT INTO documents VALUES
    ('hub2',  'Sprint/26034/hub-index.md',     'hub-index.md',     '26034', 2026, 3, 4, 1, '010800TAPR26', 'FY2026'),
    ('n4',    'Sprint/26034/note-0401.md',     'note-0401.md',     '26034', 2026, 3, 4, 0, '011030TAPR26', 'FY2026'),
    ('n5',    'Sprint/26034/note-0402.md',     'note-0402.md',     '26034', 2026, 3, 4, 0, '021400TAPR26', 'FY2026'),
    ('r5',    'Sprint/26034/ref-tda-v2.md',    'ref-tda-v2.md',    '26034', 2026, 3, 4, 0, NULL, 'FY2026'),
    ('r6',    'Sprint/26034/ref-topology.md',  'ref-topology.md',  '26034', 2026, 3, 4, 0, NULL, 'FY2026'),
    ('r7',    'Sprint/26034/ref-signals.md',   'ref-signals.md',   '26034', 2026, 3, 4, 0, NULL, 'FY2026'),
    ('n6',    'Sprint/26034/note-0403.md',     'note-0403.md',     '26034', 2026, 3, 4, 0, '031600TAPR26', 'FY2026'),
    ('r8',    'Sprint/26034/ref-maturity.md',  'ref-maturity.md',  '26034', 2026, 3, 4, 0, NULL, 'FY2026');
"""

_SIGNALS_26034 = """
INSERT INTO signals (doc_id, signal_type, signal_value, line_number) VALUES
    -- Hub2: broad coverage, overlaps with 26033 hub
    ('hub2', 'classification_ref', 'CEC B512', 2),
    ('hub2', 'classification_ref', 'JDN 3-01', 6),
    ('hub2', 'classification_ref', 'CEC B514', 10),
    ('hub2', 'dtg_content', '010800TAPR26', 1),

    -- n4: continues CEC thread
    ('n4', 'classification_ref', 'CEC B512', 4),
    ('n4', 'classification_ref', 'SUMO Process', 8),
    ('n4', 'dtg_content', '011030TAPR26', 1),

    -- n5: bridges CEC + DDC (cross-cluster signal)
    ('n5', 'classification_ref', 'CEC B514', 3),
    ('n5', 'classification_ref', 'DDC 332.6', 7),
    ('n5', 'dtg_content', '021400TAPR26', 1),

    -- r5: TDA-specific, shares with r1/r2 cluster
    ('r5', 'classification_ref', 'CEC B514', 5),
    ('r5', 'classification_ref', 'CEC B516', 9),

    -- r6: topology refs
    ('r6', 'classification_ref', 'SUMO Process', 3),
    ('r6', 'classification_ref', 'DDC 510.0', 7),

    -- r7: signal analysis
    ('r7', 'classification_ref', 'CEC B512', 4),
    ('r7', 'classification_ref', 'JDN 3-01', 8),

    -- n6: late sprint note
    ('n6', 'classification_ref', 'CEC B512', 3),
    ('n6', 'dtg_content', '031600TAPR26', 1),

    -- r8: maturity research
    ('r8', 'classification_ref', 'SUMO Process', 2),
    ('r8', 'classification_ref', 'DDC 338.1', 6);
"""

_EDGES_26034 = """
INSERT INTO edges (source_doc_id, target_path, target_doc_id, edge_type, line_number) VALUES
    -- Hub chain
    ('hub2', 'Sprint/26034/note-0401.md', 'n4', 'wiki_link', 3),
    ('hub2', 'Sprint/26034/note-0402.md', 'n5', 'wiki_link', 5),
    ('n4', 'Sprint/26034/hub-index.md', 'hub2', 'uplink', 1),
    ('n5', 'Sprint/26034/hub-index.md', 'hub2', 'uplink', 1),

    -- Dense connectivity (more edges = more mature)
    ('n4', 'Sprint/26034/note-0402.md', 'n5', 'wiki_link', 10),
    ('n5', 'Sprint/26034/ref-tda-v2.md', 'r5', 'obsidian_link', 12),
    ('r5', 'Sprint/26034/ref-topology.md', 'r6', 'obsidian_link', 8),
    ('r6', 'Sprint/26034/ref-signals.md', 'r7', 'wiki_link', 5),
    ('r7', 'Sprint/26034/hub-index.md', 'hub2', 'wiki_link', 14),
    ('n6', 'Sprint/26034/hub-index.md', 'hub2', 'uplink', 1),
    ('n6', 'Sprint/26034/note-0401.md', 'n4', 'wiki_link', 6),
    ('r8', 'Sprint/26034/ref-topology.md', 'r6', 'obsidian_link', 4),
    ('r8', 'Sprint/26034/ref-maturity.md', 'r8', 'wiki_link', 9);
"""


def _create_db(docs_sql, signals_sql, edges_sql, db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.executescript(docs_sql)
    conn.executescript(signals_sql)
    conn.executescript(edges_sql)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sprint_26033_db(tmp_path):
    return _create_db(
        _DOCS_26033, _SIGNALS_26033, _EDGES_26033,
        str(tmp_path / "signals_26033.db"),
    )


@pytest.fixture
def sprint_26034_db(tmp_path):
    return _create_db(
        _DOCS_26034, _SIGNALS_26034, _EDGES_26034,
        str(tmp_path / "signals_26034.db"),
    )
