"""Provenance: the real-world facts behind a generated scene, with their web sources.

Saved into the scene export (task.json) and the LeRobot dataset (meta/info.json), so anything
built from a sentence stays traceable to where its numbers came from.
"""
from __future__ import annotations

from spec2sim.schemas import GroundedFacts


def provenance(facts: GroundedFacts) -> dict:
    """Real-world facts behind a scene, with their sources, for exports and datasets."""
    return {
        "objects": {f.name: {"size_m": f.size, "mass_kg": f.mass, "material": f.material, "source_url": f.source_url,
                             "sources_agree": f"{f.sources_agree}/{f.sources_checked}", "note": f.note} for f in facts.facts},
        "materials": {m.material: {"friction": m.friction, "friction_source": m.friction_source,
                                   "restitution": m.restitution, "restitution_source": m.restitution_source,
                                   "note": m.note} for m in facts.materials},
        "robot": facts.robot.model_dump() if facts.robot else None,
    }
