#!/usr/bin/env python3
"""Check that the native visual frontend gate covers the original lane plan.

This is intentionally a lightweight evidence checker, not a semantic verifier.
The semantic proof still comes from the Sounio programs in the gate manifest.
This script keeps the plan-to-proof contract explicit so future edits cannot
quietly drop a promised Visual IR, renderer, control, physchem, demo, or CI
surface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True)
class Evidence:
    requirement: str
    kind: str
    target: str
    needles: tuple[str, ...] = ()
    detail: str = ""


REQUIRED: tuple[Evidence, ...] = (
    Evidence(
        "visual-ir-node-kinds",
        "contains",
        "stdlib/viz/ir.sio",
        (
            "VIZ_NODE_SCENE",
            "VIZ_NODE_GROUP",
            "VIZ_NODE_PANEL",
            "VIZ_NODE_CANVAS_LAYER",
            "VIZ_NODE_CHART_LINE",
            "VIZ_NODE_CHART_SCATTER",
            "VIZ_NODE_CHART_BAR",
            "VIZ_NODE_HEATMAP",
            "VIZ_NODE_MESH",
            "VIZ_NODE_MOLECULE",
            "VIZ_NODE_FIELD",
            "VIZ_NODE_LABEL",
            "VIZ_NODE_CONTROL",
        ),
        "Visual IR owns the v1 scene graph vocabulary.",
    ),
    Evidence(
        "visual-ir-style-layout-scene",
        "contains",
        "stdlib/viz/ir.sio",
        ("struct VizStyle", "struct VizLayout", "struct VizScene", "viz_layout_row", "viz_layout_column"),
        "Style, fixed rectangles, and simple row/column layout are Sounio data.",
    ),
    Evidence(
        "visual-ir-fixed-capacity",
        "contains",
        "stdlib/viz/ir.sio",
        ("nodes: [VizNode; 64]", "series: [VizSeriesData; 8]", "molecules: [MoleculeViz; 4]"),
        "Scene storage remains bounded and fixed capacity.",
    ),
    Evidence(
        "canvas-renderer",
        "contains",
        "stdlib/viz/viz_canvas.sio",
        ("viz_canvas_render_scene", "VIZ_NODE_MOLECULE", "VIZ_NODE_MESH", "VIZ_NODE_LATTICE_FIELD"),
        "Native CPU Canvas is the primary renderer.",
    ),
    Evidence(
        "molecule-renderer-mirror",
        "contains",
        "stdlib/viz/ir.sio",
        (
            "molecule_atom_x: [f64; 256]",
            "molecule_bond_a: [i64; 384]",
            "molecule_bond_length: [f64; 384]",
            "viz_scene_sync_molecule_mirror",
        ),
        "Molecule renderer and audit payloads have a fixed-capacity primitive mirror.",
    ),
    Evidence(
        "html-svg-renderer",
        "contains",
        "stdlib/viz/viz_html.sio",
        ("viz_html_emit_scene", "<svg", "data-viz-id", "viz-audit"),
        "Static HTML/SVG export is a second renderer over the same IR.",
    ),
    Evidence(
        "html-svg-passive-export-checker",
        "contains",
        "scripts/ci/check_viz_html_passive_export.sh",
        (
            "VIZ_HTML_PASSIVE_EXPORT_PASS",
            "<[[:space:]]*script",
            "javascript[[:space:]]*:",
            "[[:space:]]on[a-z0-9_-]+[[:space:]]*=",
        ),
        "Static HTML/SVG export is checked as passive markup with no active browser semantics.",
    ),
    Evidence(
        "gate-manifest-script-mode",
        "contains",
        "scripts/ci/check_native_visual_frontend_gate_manifest.py",
        ("\"script\"", "script entry must be executable", "script={counts['script']}"),
        "Gate manifest validation includes executable script proofs.",
    ),
    Evidence(
        "native-controls",
        "contains",
        "stdlib/viz/ir.sio",
        (
            "VIZ_CONTROL_BUTTON",
            "VIZ_CONTROL_SLIDER",
            "VIZ_CONTROL_TOGGLE",
            "VIZ_CONTROL_TABS",
            "VIZ_CONTROL_LEGEND",
            "VIZ_CONTROL_TOOLTIP",
            "VIZ_CONTROL_VIEWPORT",
            "viz_add_plot_viewport",
        ),
        "Buttons, sliders, toggles, tabs, legend, tooltip, and viewports are Visual IR.",
    ),
    Evidence(
        "interaction-state-in-sounio",
        "contains",
        "stdlib/viz/ir.sio",
        ("viz_scene_handle_event", "EV_MOUSE_PRESS", "EV_MOTION", "EV_KEY_PRESS"),
        "Native events reduce into Sounio-owned scene state.",
    ),
    Evidence(
        "scene-authoring-snapshots",
        "contains",
        "stdlib/viz/snapshot.sio",
        ("struct VizSceneSnapshot", "viz_scene_snapshot", "viz_scene_restore_snapshot", "viz_scene_snapshot_matches"),
        "Editable Visual IR state can be captured and restored as Sounio data.",
    ),
    Evidence(
        "scene-authoring-history",
        "contains",
        "stdlib/viz/snapshot.sio",
        ("struct VizSceneHistory", "viz_scene_history_record", "viz_scene_history_undo", "viz_scene_history_redo"),
        "Editable Visual IR node state can be kept in a bounded undo/redo history.",
    ),
    Evidence(
        "scene-authoring-package",
        "contains",
        "stdlib/viz/snapshot.sio",
        ("struct VizScenePackage", "viz_scene_package_capture", "viz_scene_package_restore", "viz_scene_package_emit"),
        "Editable Visual IR state can be packaged, emitted, and restored as Sounio data.",
    ),
    Evidence(
        "scene-project-format",
        "contains",
        "stdlib/viz/snapshot.sio",
        ("struct VizSceneProject", "viz_scene_project_capture", "viz_scene_project_restore", "viz_scene_project_emit"),
        "Visual Project Format v0 bundles package state, replay/export hashes, restore, verify, and emit surfaces.",
    ),
    Evidence(
        "workbench-session-timeline",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchSessionTimeline",
            "viz_workbench_session_timeline_from_session",
            "viz_workbench_replay_to_frame",
            "viz_workbench_session_timeline_mark",
        ),
        "Workbench replay sessions can be inspected as Sounio-owned frame timelines and replayed to selected frames.",
    ),
    Evidence(
        "workbench-session-archive",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchSessionArchive",
            "viz_workbench_session_archive_build",
            "viz_workbench_session_archive_verify",
            "viz_workbench_session_archive_emit",
        ),
        "Workbench replay sessions, timeline frames, visual projects, and A/B diffs can be bundled as a Sounio-owned archive.",
    ),
    Evidence(
        "workbench-visual-notebook",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchNotebook",
            "struct VizWorkbenchNotebookCompare",
            "viz_workbench_notebook_add_archive",
            "viz_workbench_notebook_compare",
            "viz_workbench_notebook_verify",
            "viz_workbench_notebook_emit",
        ),
        "Multiple session archives can be indexed, selected, compared, verified, and exported as a Sounio-owned visual lab notebook.",
    ),
    Evidence(
        "workbench-notebook-browser",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchNotebookBrowser",
            "viz_workbench_notebook_browser_select",
            "viz_workbench_notebook_browser_verify",
            "viz_workbench_notebook_browser_emit",
        ),
        "Notebook archives can be selected and replayed into fresh Workbench scenes with browser hashes and passive export metadata.",
    ),
    Evidence(
        "workbench-notebook-compare-browser",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchNotebookCompareBrowser",
            "viz_workbench_notebook_compare_browser_select",
            "viz_workbench_notebook_compare_browser_verify",
            "viz_workbench_notebook_compare_browser_emit",
        ),
        "Notebook baseline and comparison archives can be replayed side by side with browser hashes and run-to-run frame/audit deltas.",
    ),
    Evidence(
        "workbench-experiment-diff-player",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentDiffPlayer",
            "viz_workbench_experiment_diff_player_select",
            "viz_workbench_experiment_diff_player_verify",
            "viz_workbench_experiment_diff_player_emit",
        ),
        "Notebook baseline and comparison sessions can be scrubbed at arbitrary frames with timeline, frame, and audit deltas.",
    ),
    Evidence(
        "workbench-experiment-diff-overlay",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentDiffOverlay",
            "viz_workbench_experiment_diff_overlay_apply",
            "viz_workbench_experiment_diff_overlay_verify",
            "viz_workbench_experiment_diff_overlay_emit",
        ),
        "Experiment diff player state can be applied to native Visual IR controls as a visible and auditable overlay.",
    ),
    Evidence(
        "workbench-experiment-diff-artifact",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentDiffArtifact",
            "viz_workbench_experiment_diff_artifact_build",
            "viz_workbench_experiment_diff_artifact_verify",
            "viz_workbench_experiment_diff_artifact_emit",
        ),
        "Notebook diff player and overlay state can be packaged with scene audit and rendered frame hashes as a portable experiment artifact.",
    ),
    Evidence(
        "workbench-experiment-diff-library",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentDiffLibrary",
            "viz_workbench_experiment_diff_library_add",
            "viz_workbench_experiment_diff_library_verify",
            "viz_workbench_experiment_diff_library_emit",
        ),
        "Multiple experiment diff artifacts can be indexed, selected, compared, verified, and exported as a fixed-capacity Workbench library.",
    ),
    Evidence(
        "workbench-experiment-diff-library-browser",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentDiffLibraryBrowser",
            "viz_workbench_experiment_diff_library_browser_apply",
            "viz_workbench_experiment_diff_library_browser_verify",
            "viz_workbench_experiment_diff_library_browser_emit",
        ),
        "Experiment diff artifact libraries can be browsed through native Visual IR controls with Canvas-visible selection and passive export metadata.",
    ),
    Evidence(
        "workbench-experiment-compare-mode",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentCompareMode",
            "viz_workbench_experiment_compare_mode_handle_event",
            "viz_workbench_experiment_compare_mode_verify",
            "viz_workbench_experiment_compare_mode_emit",
        ),
        "Experiment artifact libraries can be driven by native Workbench events to update selected, baseline, and comparison artifacts.",
    ),
    Evidence(
        "workbench-experiment-package",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPackage",
            "viz_workbench_experiment_package_build",
            "viz_workbench_experiment_package_verify",
            "viz_workbench_experiment_package_emit",
        ),
        "Notebook, artifact library, browser, compare mode, scene audit, and rendered frame hashes can be packaged as one reproducible Workbench experiment.",
    ),
    Evidence(
        "workbench-experiment-package-restore",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPackageRestore",
            "viz_workbench_experiment_package_restore",
            "viz_workbench_experiment_package_restore_verify",
            "viz_workbench_experiment_package_restore_emit",
        ),
        "Reproducible Workbench experiment packages can be restored into a fresh scene with library, browser, mode, frame, and audit hashes verified.",
    ),
    Evidence(
        "workbench-experiment-package-store",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPackageStore",
            "viz_workbench_experiment_package_store_add",
            "viz_workbench_experiment_package_store_verify",
            "viz_workbench_experiment_package_store_emit",
        ),
        "Multiple reproducible Workbench experiment packages can be saved into fixed slots, selected, compared, verified, and exported.",
    ),
    Evidence(
        "workbench-experiment-package-store-controls",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VIZ_WORKBENCH_TAG_EXPERIMENT_STORE_SAVE",
            "viz_workbench_handle_experiment_package_store_event",
            "viz_workbench_experiment_package_store_mark",
            "viz_workbench_experiment_package_store_restore_active",
        ),
        "Experiment package slots are controlled by native Visual IR events for save, active, baseline, compare, restore, and delta markers.",
    ),
    Evidence(
        "workbench-experiment-timeline-cockpit",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentTimelineCockpit",
            "viz_workbench_experiment_timeline_cockpit_build",
            "viz_workbench_experiment_timeline_cockpit_verify",
            "viz_workbench_experiment_timeline_cockpit_emit",
        ),
        "Experiment package slots, active package, notebook archives, selected frame, timeline hashes, and package/render/audit deltas are bound into one native cockpit state.",
    ),
    Evidence(
        "workbench-experiment-cockpit-playback",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentCockpitPlayback",
            "viz_workbench_handle_experiment_cockpit_playback_event",
            "viz_workbench_experiment_cockpit_playback_verify",
            "viz_workbench_experiment_cockpit_playback_emit",
        ),
        "Experiment cockpit playback controls step active, baseline, and compare timelines with native Visual IR time buttons and verified frame/time/hash state.",
    ),
    Evidence(
        "workbench-experiment-export-packet",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentExportPacket",
            "viz_workbench_experiment_export_packet_build",
            "viz_workbench_experiment_export_packet_capture",
            "viz_workbench_experiment_export_packet_verify",
            "viz_workbench_experiment_export_packet_emit",
        ),
        "Experiment package, restore proof, package store, cockpit, playback, export frame, and scene audit are bound into one shareable native lab packet.",
    ),
    Evidence(
        "workbench-experiment-export-packet-domain-stamps",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "node_stamp_hash",
            "chart_stamp_hash",
            "field_stamp_hash",
            "molecule_stamp_hash",
            "mesh_stamp_hash",
            "payload_stamp_hash",
            "viz_workbench_experiment_export_packet_capture_domain_stamps",
        ),
        "Shareable native lab packets expose compact domain stamps for scene nodes, chart/heatmap payloads, physchem fields, molecule payloads, mesh payloads, and the combined export payload.",
    ),
    Evidence(
        "workbench-experiment-import-replay",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentImportReplayReport",
            "viz_workbench_experiment_import_replay_report_build",
            "viz_workbench_experiment_import_replay_report_verify",
            "viz_workbench_experiment_import_replay_report_emit",
        ),
        "Shareable experiment packets can be imported into a fresh Workbench scene, replayed as Sounio data, and compared against exported frame/audit hashes.",
    ),
    Evidence(
        "workbench-experiment-packet-ledger",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPacketLedger",
            "viz_workbench_experiment_packet_ledger_add",
            "viz_workbench_experiment_packet_ledger_verify",
            "viz_workbench_experiment_packet_ledger_emit",
            "packet_hash_deltas",
            "payload_stamp_deltas",
        ),
        "Shareable native lab packets can be sequenced into a fixed-capacity Sounio-owned ledger with packet/payload/frame/audit deltas and passive HTML/SVG metadata.",
    ),
    Evidence(
        "workbench-experiment-packet-ledger-replay",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPacketLedgerReplayReport",
            "viz_workbench_experiment_packet_ledger_replay_report_build_pair",
            "viz_workbench_experiment_packet_ledger_replay_report_verify",
            "viz_workbench_experiment_packet_ledger_replay_report_emit",
            "final_frame_hash_delta",
            "final_audit_hash_delta",
        ),
        "Experiment packet ledgers can be bound to first/last packets and the final import replay report, proving the sequence closes with zero frame and audit drift.",
    ),
    Evidence(
        "workbench-experiment-packet-ledger-browser",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchExperimentPacketLedgerBrowser",
            "viz_workbench_experiment_packet_ledger_browser_apply",
            "viz_workbench_experiment_packet_ledger_browser_verify",
            "viz_workbench_experiment_packet_ledger_browser_emit",
            "selected_packet_hash",
            "packet_delta_node",
        ),
        "Experiment packet ledgers can be projected back into native Visual IR controls, making selected packet hashes and packet/payload/frame/audit deltas visible in Canvas and passive HTML/SVG exports.",
    ),
    Evidence(
        "workbench-experiment-packet-ledger-browser-events",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "viz_workbench_handle_experiment_packet_ledger_browser_event",
            "EV_KEY_PRESS",
            "KEY_RIGHT",
            "KEY_LEFT",
            "EV_MOUSE_PRESS",
            "app.dirty = true",
        ),
        "Experiment packet-ledger browser selection is navigable through native key and mouse events, updating Sounio-owned browser state and marking the app dirty for redraw.",
    ),
    Evidence(
        "workbench-experiment-flight-recorder",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VizWorkbenchExperimentFlightRecorder",
            "viz_workbench_experiment_flight_recorder_begin",
            "viz_workbench_experiment_flight_recorder_append",
            "viz_workbench_experiment_flight_recorder_verify",
            "event_kinds",
            "frame_hashes",
            "final_audit_delta",
        ),
        "Experiment packet-ledger navigation is captured as a fixed-capacity native flight recorder with event kinds, selected packet indices, browser hashes, frame hashes, and final scene-audit binding.",
    ),
    Evidence(
        "workbench-experiment-flight-recorder-seek",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VizWorkbenchExperimentFlightRecorderSeek",
            "viz_workbench_experiment_flight_recorder_seek_apply",
            "viz_workbench_experiment_flight_recorder_seek_verify",
            "viz_workbench_experiment_flight_recorder_seek_emit",
            "expected_frame_hash",
            "browser_hash_delta",
            "selected_event",
        ),
        "Experiment flight-recorder events can be selected and projected back into native Visual IR with expected browser/frame hashes and passive export metadata.",
    ),
    Evidence(
        "workbench-experiment-flight-recorder-timeline",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VizWorkbenchExperimentFlightRecorderTimeline",
            "viz_workbench_experiment_flight_recorder_timeline_apply",
            "viz_workbench_experiment_flight_recorder_timeline_verify",
            "viz_workbench_handle_experiment_flight_recorder_timeline_event",
            "key_action_count",
            "mouse_action_count",
            "current_expected_frame_hash",
        ),
        "Experiment flight-recorder seek state is navigable through native timeline controls, preserving selected-event state, action counters, and expected frame hashes.",
    ),
    Evidence(
        "workbench-experiment-flight-recorder-evidence-package",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VizWorkbenchExperimentFlightRecorderEvidencePackage",
            "viz_workbench_experiment_flight_recorder_evidence_package_build",
            "viz_workbench_experiment_flight_recorder_evidence_package_verify",
            "viz_workbench_experiment_flight_recorder_evidence_package_emit",
            "rendered_frame_hash",
            "frame_hash_delta",
            "browser_hash_delta",
        ),
        "Experiment flight-recorder timeline state can be sealed into a passive evidence package binding recorder, seek, timeline, rendered frame, audit hash, and zero frame/browser drift.",
    ),
    Evidence(
        "workbench-experiment-import-replay-zero-drift",
        "contains",
        "tests/run-pass/viz_workbench_experiment_diff_player.sio",
        (
            "assert(import_report.replay_frame_hash == export_packet.export_frame_hash)",
            "assert(import_report.replay_scene_audit_hash == export_packet.export_scene_audit_hash)",
            "assert(import_report.frame_hash_delta == 0)",
            "assert(import_report.scene_audit_hash_delta == 0)",
        ),
        "The canonical Workbench experiment import replay proof requires zero Canvas frame drift and zero scene-audit drift.",
    ),
    Evidence(
        "workbench-native-package-controls",
        "contains",
        "stdlib/viz/workbench.sio",
        ("viz_workbench_handle_package_command", "VIZ_WORKBENCH_TAG_PACKAGE_SAVE", "VizScenePackage"),
        "Workbench package save, restore, undo, and redo commands are native Visual IR control state.",
    ),
    Evidence(
        "workbench-native-project-controls",
        "contains",
        "stdlib/viz/workbench.sio",
        ("struct VizWorkbenchProjectStore", "viz_workbench_project_app_handle_event", "VIZ_WORKBENCH_TAG_EXPORT"),
        "Workbench save, load, and export controls operate on a Sounio-owned visual project store.",
    ),
    Evidence(
        "workbench-project-workspace",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchProjectWorkspace",
            "viz_workbench_project_workspace_save",
            "viz_workbench_project_workspace_compare",
        ),
        "Workbench project workspace keeps A/B visual project slots, comparison state, and restore metadata as Sounio data.",
    ),
    Evidence(
        "workbench-project-workspace-controls",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "VIZ_WORKBENCH_TAG_WORKSPACE_SAVE_A",
            "VIZ_WORKBENCH_TAG_WORKSPACE_COMPARE",
            "viz_workbench_workspace_app_handle_event",
        ),
        "Workbench A/B project workspace controls are native Visual IR nodes handled by the Sounio reducer.",
    ),
    Evidence(
        "workbench-project-diff-panel",
        "contains",
        "stdlib/viz/workbench.sio",
        (
            "struct VizWorkbenchProjectDiff",
            "viz_workbench_project_workspace_compare_detail",
            "VIZ_WORKBENCH_TAG_WORKSPACE_DT",
            "VIZ_WORKBENCH_TAG_WORKSPACE_DPROJECT",
        ),
        "Workbench A/B compare produces Sounio-owned semantic diff state and native viewport nodes for key deltas.",
    ),
    Evidence(
        "physchem-core-structs",
        "contains",
        "stdlib/viz/physchem.sio",
        (
            "struct AtomViz",
            "struct BondViz",
            "struct MoleculeViz",
            "struct ScalarField2D",
            "struct VectorField2D",
            "struct Trajectory3D",
        ),
        "Physico-chemistry payloads are native Sounio structs.",
    ),
    Evidence(
        "physchem-extended-surfaces",
        "contains",
        "stdlib/viz/physchem.sio",
        ("struct SpectrumViz", "struct LatticeField2D", "struct ParticleEventView", "variance"),
        "Spectra, lattice/phonon, particle events, and uncertainty arrays are first-class.",
    ),
    Evidence(
        "physchem-units",
        "contains",
        "stdlib/viz/physchem.sio",
        ("use units::lib::{Quantity, UnitDim", "pub length: Quantity", "pub unit: UnitDim"),
        "Physchem unit metadata uses the Sounio units layer.",
    ),
    Evidence(
        "seed-canvas-extension",
        "contains",
        "stdlib/display/canvas_ext.sio",
        ("canvas_blit_surface", "graphics::surface::Surface", "alpha"),
        "Canvas extension bridges real graphics surfaces with alpha.",
    ),
    Evidence(
        "seed-sci-effects",
        "contains",
        "stdlib/viz/sci.sio",
        ("sci_sqrt", "with Mut, Div"),
        "Scientific drawing helper effects are explicit.",
    ),
    Evidence(
        "module-readme-examples",
        "exists",
        "stdlib/viz/mod.sio",
        detail="Viz module surface exists.",
    ),
    Evidence("module-readme", "exists", "stdlib/viz/README.md", detail="Viz README exists."),
    Evidence("module-examples", "exists", "stdlib/viz/EXAMPLES.md", detail="Viz examples doc exists."),
    Evidence(
        "chart-builders",
        "contains",
        "stdlib/viz/ir.sio",
        (
            "viz_add_line_chart",
            "viz_add_scatter_chart",
            "viz_add_bar_chart",
            "viz_add_heatmap",
            "viz_add_uncertainty_band",
            "viz_add_forest_plot",
            "viz_add_waterfall",
        ),
        "Line, scatter, bar, heatmap, band, forest, and waterfall builders exist.",
    ),
    Evidence(
        "lab-demo-rich-scene",
        "contains",
        "examples/viz_lab/main.sio",
        (
            "viz_add_line_chart",
            "viz_add_uncertainty_band",
            "viz_add_heatmap",
            "viz_add_molecule",
            "viz_add_scalar_field",
            "viz_add_mesh",
            "VIZ_CONTROL_SLIDER",
            "VIZ_CONTROL_TABS",
            "VIZ_CONTROL_TOGGLE",
            "viz_app_handle_event",
        ),
        "Headless lab demo composes plot, molecule, field, 3D, and controls.",
    ),
    Evidence(
        "native-window-demo",
        "contains",
        "examples/viz_lab_window/main.sio",
        ("window_open", "viz_window_step", "VIZ_CONTROL_TABS"),
        "Native window demo uses the same Visual IR app path.",
    ),
    Evidence(
        "physchem-demo",
        "contains",
        "examples/viz_physchem_demo/main.sio",
        (
            "molecule_h2o",
            "molecule_rapamycin_coarse",
            "viz_add_vector_field",
            "viz_add_trajectory",
            "viz_add_spectrum",
            "viz_add_lattice_field",
            "viz_add_particle_event",
            "viz_add_mesh",
        ),
        "Physchem demo covers molecule, field, dynamics, spectra, events, and 3D.",
    ),
    Evidence(
        "gpu-deferred-roadmap",
        "contains",
        "docs/design/native_visual_frontend_roadmap.md",
        ("V2: GPU Renderer", "Deferred until general `bin/souc --backend gpu` wiring exists"),
        "GPU renderer stays deferred and documented.",
    ),
    Evidence(
        "gate-headless-run",
        "label",
        "viz_headless",
        detail="Headless proof is in the canonical gate.",
    ),
    Evidence(
        "headless-pixel-assertions",
        "contains",
        "tests/run-pass/viz_headless.sio",
        (
            "axis_px",
            "aa_px",
            "hm_px",
            "band_px",
            "mol_inner_px",
            "mol_bond_px",
            "mol_no_bond_px",
            "mesh_px",
        ),
        "Headless proof includes explicit axis, AA line, heatmap, uncertainty band, molecule atom/bond, and 3D mesh pixel probes.",
    ),
    Evidence(
        "gate-canvas-ext-run",
        "label",
        "viz_canvas_ext_surface",
        detail="Canvas extension pixel/surface proof is in the canonical gate.",
    ),
    Evidence(
        "gate-chart-builders-run",
        "label",
        "viz_chart_builders",
        detail="Chart builder proof is in the canonical gate.",
    ),
    Evidence(
        "gate-interaction-run",
        "label",
        "viz_event_reducer",
        detail="Interaction reducer proof is in the canonical gate.",
    ),
    Evidence(
        "gate-scene-snapshot-run",
        "label",
        "viz_scene_snapshot",
        detail="Scene snapshot/restore proof is in the canonical gate.",
    ),
    Evidence(
        "gate-scene-history-run",
        "label",
        "viz_scene_history",
        detail="Scene undo/redo history proof is in the canonical gate.",
    ),
    Evidence(
        "gate-scene-package-run",
        "label",
        "viz_scene_package",
        detail="Scene package export/import proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-package-controls-run",
        "label",
        "viz_workbench_package_controls",
        detail="Workbench-native package controls proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-project-bundle-run",
        "label",
        "viz_workbench_project_bundle",
        detail="Workbench Visual Project Format v0 proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-project-controls-run",
        "label",
        "viz_workbench_project_controls",
        detail="Workbench project save/load/export controls proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-project-workspace-run",
        "label",
        "viz_workbench_project_workspace",
        detail="Workbench A/B visual project workspace proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-project-workspace-controls-run",
        "label",
        "viz_workbench_project_workspace_controls",
        detail="Workbench native A/B project workspace controls proof is in the canonical gate.",
    ),
    Evidence(
        "gate-html-run",
        "label",
        "viz_html_static",
        detail="Static HTML/SVG proof is in the canonical gate.",
    ),
    Evidence(
        "gate-html-passive-export-script",
        "label",
        "viz_html_passive_export",
        detail="Passive HTML/SVG export proof is in the canonical gate.",
    ),
    Evidence(
        "gate-physchem-run",
        "label",
        "viz_physchem",
        detail="Physchem proof is in the canonical gate.",
    ),
    Evidence(
        "gate-physchem-dynamics-run",
        "label",
        "viz_physchem_dynamics",
        detail="Physchem dynamics proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-run",
        "label",
        "viz_workbench_roundtrip",
        detail="Composed Workbench proof is in the canonical gate.",
    ),
    Evidence(
        "gate-workbench-timeline-run",
        "label",
        "viz_workbench_timeline_playback",
        detail="Workbench timeline playback proof is in the canonical gate.",
    ),
    Evidence(
        "gate-replay-session-run",
        "label",
        "viz_workbench_replay_session",
        detail="Workbench replay session proof is in the canonical gate.",
    ),
    Evidence(
        "gate-replay-html-archive-run",
        "label",
        "viz_workbench_replay_html_archive",
        detail="Replay HTML archive proof is in the canonical gate.",
    ),
    Evidence(
        "gate-session-timeline-run",
        "label",
        "viz_workbench_session_timeline",
        detail="Workbench replay session timeline navigation proof is in the canonical gate.",
    ),
    Evidence(
        "gate-session-archive-run",
        "label",
        "viz_workbench_session_archive",
        detail="Workbench session timeline archive proof is in the canonical gate.",
    ),
    Evidence(
        "gate-notebook-run",
        "label",
        "viz_workbench_notebook",
        detail="Workbench visual lab notebook proof is in the canonical gate.",
    ),
    Evidence(
        "gate-notebook-browser-run",
        "label",
        "viz_workbench_notebook_browser",
        detail="Workbench visual lab notebook browser proof is in the canonical gate.",
    ),
    Evidence(
        "gate-notebook-compare-browser-run",
        "label",
        "viz_workbench_notebook_compare_browser",
        detail="Workbench visual lab notebook compare browser proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-diff-player-run",
        "label",
        "viz_workbench_experiment_diff_player",
        detail="Workbench experiment diff player proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-diff-overlay-run",
        "label",
        "viz_workbench_experiment_diff_overlay",
        detail="Workbench experiment diff overlay proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-diff-artifact-run",
        "label",
        "viz_workbench_experiment_diff_artifact",
        detail="Workbench experiment diff artifact proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-diff-library-run",
        "label",
        "viz_workbench_experiment_diff_library",
        detail="Workbench experiment diff library proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-diff-library-browser-run",
        "label",
        "viz_workbench_experiment_diff_library_browser",
        detail="Workbench experiment diff library browser proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-compare-mode-run",
        "label",
        "viz_workbench_experiment_compare_mode",
        detail="Workbench experiment comparison mode proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-package-run",
        "label",
        "viz_workbench_experiment_package",
        detail="Workbench experiment package proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-package-restore-run",
        "label",
        "viz_workbench_experiment_package_restore",
        detail="Workbench experiment package restore proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-package-store-run",
        "label",
        "viz_workbench_experiment_package_store",
        detail="Workbench experiment package store proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-package-store-controls-run",
        "label",
        "viz_workbench_experiment_package_store_controls",
        detail="Workbench experiment package store controls proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-timeline-cockpit-run",
        "label",
        "viz_workbench_experiment_timeline_cockpit",
        detail="Workbench experiment timeline cockpit proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-cockpit-playback-run",
        "label",
        "viz_workbench_experiment_cockpit_playback",
        detail="Workbench experiment cockpit playback proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-export-packet-run",
        "label",
        "viz_workbench_experiment_export_packet",
        detail="Workbench experiment export packet proof is in the canonical gate.",
    ),
    Evidence(
        "gate-experiment-import-replay-run",
        "label",
        "viz_workbench_experiment_import_replay",
        detail="Workbench experiment import/replay parity proof is in the canonical gate.",
    ),
    Evidence(
        "gate-demo-compiles",
        "labels",
        "",
        ("viz_hello_demo", "viz_lab_demo", "viz_lab_window_demo", "viz_physchem_demo", "viz_workbench_demo"),
        "Demo compile gates are present.",
    ),
    Evidence(
        "gate-demo-runs",
        "labels",
        "",
        ("viz_lab_run", "viz_physchem_demo_run", "viz_workbench_demo_run"),
        "Headless demo run markers are present.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--manifest",
        default="scripts/ci/native_visual_frontend_gate.manifest.tsv",
        help="Native visual frontend gate manifest.",
    )
    return parser.parse_args()


def read_labels(root: Path, manifest_rel: str) -> tuple[set[str], list[str]]:
    manifest = Path(manifest_rel)
    if not manifest.is_absolute():
        manifest = root / manifest
    labels: set[str] = set()
    errors: list[str] = []
    if not manifest.exists():
        return labels, [f"manifest missing: {manifest}"]
    for lineno, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            errors.append(f"{manifest}:{lineno}: expected 4 TSV columns")
            continue
        labels.add(parts[1])
    return labels, errors


def check_evidence(root: Path, labels: set[str], ev: Evidence) -> list[str]:
    if ev.kind == "label":
        return [] if ev.target in labels else [f"{ev.requirement}: missing gate label {ev.target!r}"]
    if ev.kind == "labels":
        missing = [label for label in ev.needles if label not in labels]
        if missing:
            return [f"{ev.requirement}: missing gate labels: {', '.join(missing)}"]
        return []

    path = root / ev.target
    if ev.kind == "exists":
        return [] if path.exists() else [f"{ev.requirement}: missing file {ev.target}"]
    if ev.kind == "contains":
        if not path.exists():
            return [f"{ev.requirement}: missing file {ev.target}"]
        text = path.read_text(encoding="utf-8")
        missing = [needle for needle in ev.needles if needle not in text]
        if missing:
            return [f"{ev.requirement}: {ev.target} missing: {', '.join(missing)}"]
        return []
    return [f"{ev.requirement}: unknown evidence kind {ev.kind!r}"]


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    labels, errors = read_labels(root, args.manifest)

    for ev in REQUIRED:
        errors.extend(check_evidence(root, labels, ev))

    if errors:
        for err in errors:
            print(f"error: native visual frontend plan coverage: {err}", file=sys.stderr)
        return 1

    print(f"native_visual_frontend_plan_coverage: PASS requirements={len(REQUIRED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
