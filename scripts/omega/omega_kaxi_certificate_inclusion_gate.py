#!/usr/bin/env python3
"""Verify Omega K-AXI certificate attestation inclusion proofs.

This is the independent verifier for the Kretikos/K-AXI evidence braid:
it consumes the Omega attestation JSON, recomputes the Merkle root, and
emits profile-level inclusion proofs for each required K-AXI profile.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "sounio.omega.kaxi-certificate-inclusion-proofs.v1"
ATTESTATION_SCHEMA = "sounio.omega.kaxi-certificate-attestation.v1"
DEFAULT_REQUIRED_PROFILES = (
    "vec_add_f32",
    "vec_sub_f32",
    "vec_mul_f32",
    "vec_div_f32",
    "fma_f32",
    "epistemic_elementwise_f32",
    "epistemic_dual_output_f32",
)
REQUIRED_PROFILE_LEAF_KINDS = (
    "hpc_source_result",
    "kaxi_assembly",
    "kaxi_certificate",
    "kaxi_witness",
    "runtime_bundle",
    "runtime_source_profile",
    "source",
    "source_witness",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def leaf_hash(kind: str, profile: str, name: str, digest: str) -> str:
    return sha256_text(f"{kind}\0{profile}\0{name}\0{digest}")


def parent_hash(left: str, right: str) -> str:
    return sha256_text(left + right)


def canonical_leaf_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("profile", "")),
        str(row.get("kind", "")),
        str(row.get("name", "")),
        str(row.get("sha256", "")),
    )


def recompute_root(leaves: list[dict[str, Any]]) -> str:
    level = [str(row.get("leaf_hash", "")) for row in leaves]
    if not level:
        return ""
    while len(level) > 1:
        nxt: list[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else left
            nxt.append(parent_hash(left, right))
        level = nxt
    return level[0]


def inclusion_proof(leaves: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    proof: list[dict[str, Any]] = []
    level = [str(row.get("leaf_hash", "")) for row in leaves]
    position = index
    depth = 0
    while len(level) > 1:
        if position % 2 == 0:
            sibling_index = position + 1 if position + 1 < len(level) else position
            sibling_position = "right"
        else:
            sibling_index = position - 1
            sibling_position = "left"
        proof.append(
            {
                "depth": depth,
                "sibling_position": sibling_position,
                "sibling_index": sibling_index,
                "sibling_hash": level[sibling_index],
            }
        )

        nxt: list[str] = []
        for idx in range(0, len(level), 2):
            left = level[idx]
            right = level[idx + 1] if idx + 1 < len(level) else left
            nxt.append(parent_hash(left, right))
        level = nxt
        position //= 2
        depth += 1
    return proof


def verify_inclusion(leaf: str, proof: list[dict[str, Any]], root: str) -> bool:
    cursor = leaf
    for step in proof:
        sibling = str(step.get("sibling_hash", ""))
        if step.get("sibling_position") == "left":
            cursor = parent_hash(sibling, cursor)
        else:
            cursor = parent_hash(cursor, sibling)
    return cursor == root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Omega K-AXI certificate inclusion proofs")
    parser.add_argument(
        "--attestation",
        default="artifacts/omega/kaxi_certificate_attestation.v1.json",
        help="Omega K-AXI certificate attestation JSON",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/kaxi_certificate_inclusion_proofs.v1.json",
        help="Output inclusion-proof JSON",
    )
    parser.add_argument(
        "--required-profiles",
        default=",".join(DEFAULT_REQUIRED_PROFILES),
        help="Comma-separated profiles that must have certificate inclusion proofs",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if verification fails")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    attestation_path = Path(args.attestation)
    out_path = Path(args.out)
    required_profiles = [item.strip() for item in args.required_profiles.split(",") if item.strip()]

    blockers: list[dict[str, Any]] = []
    attestation_record: dict[str, Any] | None = None
    profile_proofs: list[dict[str, Any]] = []
    global_checks: dict[str, Any] = {}
    tamper_check: dict[str, Any] = {
        "status": "not_run",
        "reason": "not_enough_leaves",
    }

    try:
        raw = attestation_path.read_bytes()
        attestation = json.loads(raw.decode("utf-8"))
        attestation_record = {
            "path": str(attestation_path),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
    except Exception as exc:
        blockers.append({"kind": "attestation_read_failed", "detail": str(exc)})
        attestation = {}

    leaves = attestation.get("merkle", {}).get("leaves", [])
    if not isinstance(leaves, list):
        leaves = []
        blockers.append({"kind": "attestation_leaves_not_list"})

    root_expected = str(attestation.get("merkle", {}).get("root", ""))
    leaf_count_expected = attestation.get("merkle", {}).get("leaf_count")
    canonical_leaves = sorted(leaves, key=canonical_leaf_key)
    leaf_order_canonical = leaves == canonical_leaves
    root_recomputed = recompute_root(leaves)
    root_recomputed_canonical = recompute_root(canonical_leaves)

    leaf_hash_mismatches: list[dict[str, Any]] = []
    for idx, row in enumerate(leaves):
        actual = leaf_hash(
            str(row.get("kind", "")),
            str(row.get("profile", "")),
            str(row.get("name", "")),
            str(row.get("sha256", "")),
        )
        if actual != row.get("leaf_hash"):
            leaf_hash_mismatches.append(
                {
                    "index": idx,
                    "profile": row.get("profile"),
                    "kind": row.get("kind"),
                    "expected_leaf_hash": actual,
                    "observed_leaf_hash": row.get("leaf_hash"),
                }
            )

    global_checks = {
        "attestation_schema": attestation.get("schema"),
        "attestation_status": attestation.get("status_summary"),
        "leaf_count_expected": leaf_count_expected,
        "leaf_count_observed": len(leaves),
        "leaf_count_match": leaf_count_expected == len(leaves),
        "leaf_order_canonical": leaf_order_canonical,
        "all_leaf_hashes_recompute": not leaf_hash_mismatches,
        "root_expected": root_expected,
        "root_recomputed": root_recomputed,
        "root_recomputed_canonical": root_recomputed_canonical,
        "root_match": root_expected == root_recomputed,
        "canonical_root_match": root_expected == root_recomputed_canonical,
        "leaf_hash_mismatches": leaf_hash_mismatches,
    }

    if attestation.get("schema") != ATTESTATION_SCHEMA:
        blockers.append({"kind": "attestation_schema_mismatch", "schema": attestation.get("schema")})
    if attestation.get("status_summary") != "pass":
        blockers.append({"kind": "attestation_status_not_pass", "status": attestation.get("status_summary")})
    if not global_checks["leaf_count_match"]:
        blockers.append({"kind": "leaf_count_mismatch"})
    if not leaf_order_canonical:
        blockers.append({"kind": "leaf_order_not_canonical"})
    if leaf_hash_mismatches:
        blockers.append({"kind": "leaf_hash_mismatch", "count": len(leaf_hash_mismatches)})
    if root_expected != root_recomputed:
        blockers.append({"kind": "merkle_root_mismatch"})

    if leaves:
        tampered = [dict(row) for row in leaves]
        tampered[0]["sha256"] = "0" * 64 if tampered[0].get("sha256") != "0" * 64 else "1" * 64
        tampered[0]["leaf_hash"] = leaf_hash(
            str(tampered[0].get("kind", "")),
            str(tampered[0].get("profile", "")),
            str(tampered[0].get("name", "")),
            str(tampered[0].get("sha256", "")),
        )
        tampered_root = recompute_root(tampered)
        tamper_check = {
            "status": "pass" if tampered_root != root_expected else "fail",
            "reason": "single_leaf_digest_mutation_changes_root"
            if tampered_root != root_expected
            else "single_leaf_digest_mutation_did_not_change_root",
            "tampered_leaf_index": 0,
            "tampered_root": tampered_root,
            "original_root": root_expected,
        }
        if tampered_root == root_expected:
            blockers.append({"kind": "tamper_sensitivity_failed"})

    certificates_by_profile = {
        str(row.get("profile", "")): row
        for row in attestation.get("certificates", [])
        if isinstance(row, dict) and row.get("profile")
    }

    leaf_indices_by_profile: dict[str, list[int]] = {}
    for idx, row in enumerate(leaves):
        leaf_indices_by_profile.setdefault(str(row.get("profile", "")), []).append(idx)

    all_proofs_verified = True
    for profile in required_profiles:
        cert = certificates_by_profile.get(profile)
        profile_blockers: list[dict[str, Any]] = []
        if not cert:
            profile_blockers.append({"kind": "profile_certificate_missing"})
        else:
            if cert.get("certificate_status") != "pass":
                profile_blockers.append({"kind": "certificate_status_not_pass", "status": cert.get("certificate_status")})
            if cert.get("failed_checks"):
                profile_blockers.append({"kind": "certificate_failed_checks", "failed_checks": cert.get("failed_checks")})
            runtime = cert.get("runtime") or {}
            if runtime.get("status") != "pass":
                profile_blockers.append({"kind": "runtime_not_pass", "status": runtime.get("status"), "reason": runtime.get("reason")})
            if runtime.get("rung") != profile:
                profile_blockers.append({"kind": "runtime_rung_mismatch", "runtime_rung": runtime.get("rung")})

        indices = leaf_indices_by_profile.get(profile, [])
        observed_kinds = sorted(str(leaves[idx].get("kind", "")) for idx in indices)
        missing_kinds = [kind for kind in REQUIRED_PROFILE_LEAF_KINDS if kind not in observed_kinds]
        if missing_kinds:
            profile_blockers.append({"kind": "required_leaf_kinds_missing", "missing": missing_kinds})

        proof_leaves: list[dict[str, Any]] = []
        profile_leaf_hashes = []
        for idx in indices:
            row = leaves[idx]
            proof = inclusion_proof(leaves, idx)
            verified = verify_inclusion(str(row.get("leaf_hash", "")), proof, root_expected)
            if not verified:
                all_proofs_verified = False
                profile_blockers.append({"kind": "inclusion_proof_failed", "leaf_index": idx})
            profile_leaf_hashes.append(str(row.get("leaf_hash", "")))
            proof_leaves.append(
                {
                    "index": idx,
                    "kind": row.get("kind"),
                    "name": row.get("name"),
                    "sha256": row.get("sha256"),
                    "leaf_hash": row.get("leaf_hash"),
                    "proof": proof,
                    "proof_verified": verified,
                }
            )

        profile_commitment = sha256_text("|".join(sorted(profile_leaf_hashes))) if profile_leaf_hashes else ""
        profile_root = recompute_root([leaves[idx] for idx in indices])
        profile_proofs.append(
            {
                "profile": profile,
                "status": "pass" if not profile_blockers else "fail",
                "blockers": profile_blockers,
                "certificate_status": cert.get("certificate_status") if cert else None,
                "runtime": cert.get("runtime") if cert else None,
                "source": cert.get("source") if cert else None,
                "kaxi": cert.get("kaxi") if cert else None,
                "leaf_count": len(indices),
                "required_leaf_kinds": list(REQUIRED_PROFILE_LEAF_KINDS),
                "observed_leaf_kinds": observed_kinds,
                "profile_commitment_sha256": profile_commitment,
                "profile_local_merkle_root": profile_root,
                "leaves": proof_leaves,
            }
        )
        for blocker in profile_blockers:
            blockers.append({"kind": "profile_inclusion_failed", "profile": profile, "detail": blocker})

    if not all_proofs_verified:
        blockers.append({"kind": "one_or_more_inclusion_proofs_failed"})

    status = "pass" if not blockers else "fail"
    output = {
        "schema": SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "reason": "kaxi_certificate_inclusion_proofs_pass" if status == "pass" else blockers[0]["kind"],
        "source_attestation": attestation_record,
        "required_profiles": required_profiles,
        "global_verification": global_checks,
        "tamper_sensitivity": tamper_check,
        "profile_proofs": profile_proofs,
        "blockers": blockers,
        "boundaries": [
            "verifier_recomputes_leaf_hashes_and_global_merkle_root",
            "verifier_emits_leaf_level_inclusion_proofs_for_each_required_profile",
            "profile_commitments_are_hashes_of_profile_leaf_hash_sets",
            "tamper_check_proves_single_leaf_digest_mutation_changes_the_root",
            "proofs_attest_supported_kaxi_profiles_only",
            "proofs_do_not_claim_arbitrary_sounio_gpu_lowering",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "omega_kaxi_certificate_inclusion_gate: "
        f"status={status} root={global_checks.get('root_recomputed')} "
        f"profiles={len(profile_proofs)} out={out_path}"
    )
    if args.strict and status != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
