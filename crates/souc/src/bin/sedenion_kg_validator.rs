use serde_json::Value;
use std::collections::HashSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
struct ValidateConfig {
    run_dir: PathBuf,
    sample_count: usize,
    seed: i64,
    score_tolerance: f64,
    rank_tolerance: f64,
    check_full: bool,
}

#[derive(Debug, Clone)]
struct ParsedPrediction {
    head: i64,
    relation: i64,
    tail: i64,
    true_score: f64,
    neg_best_score: f64,
    rank: i64,
    line_no: usize,
}

#[derive(Debug, Clone)]
struct MappingSummary {
    entity_count: Option<usize>,
    relation_count: Option<usize>,
    entity_ids: Option<HashSet<i64>>,
    relation_ids: Option<HashSet<i64>>,
}

fn usage() -> String {
    let mut msg = String::new();
    let _ = writeln!(
        msg,
        "sedenion_kg_validator <run-dir> [--sample-count N] [--seed N] [--score-tol T] [--rank-tol T] [--full]"
    );
    msg
}

fn parse_args() -> Result<ValidateConfig, String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err(usage());
    }

    let mut cfg = ValidateConfig {
        run_dir: PathBuf::from(&args[1]),
        sample_count: 128,
        seed: 0,
        score_tolerance: 1e-4,
        rank_tolerance: 0.0,
        check_full: false,
    };

    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--sample-count" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --sample-count".to_string());
                }
                cfg.sample_count = args[i].parse::<usize>().map_err(|e| format!("{e}"))?;
            }
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --seed".to_string());
                }
                cfg.seed = args[i].parse::<i64>().map_err(|e| format!("{e}"))?;
            }
            "--score-tol" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --score-tol".to_string());
                }
                cfg.score_tolerance = args[i].parse::<f64>().map_err(|e| format!("{e}"))?;
            }
            "--rank-tol" => {
                i += 1;
                if i >= args.len() {
                    return Err("missing value for --rank-tol".to_string());
                }
                cfg.rank_tolerance = args[i].parse::<f64>().map_err(|e| format!("{e}"))?;
            }
            "--full" => {
                cfg.check_full = true;
            }
            "--check-full" => {
                cfg.check_full = true;
            }
            flag => return Err(format!("unknown argument: {flag}")),
        }
        i += 1;
    }

    Ok(cfg)
}

fn read_text(path: PathBuf) -> Result<String, String> {
    fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

fn parse_f64_from_json(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(|v| v.as_f64())
}

fn parse_json_file(path: PathBuf) -> Result<Value, String> {
    let raw = read_text(path)?;
    serde_json::from_str(&raw).map_err(|e| format!("invalid JSON: {e}"))
}

fn parse_predictions(raw: &str) -> Result<Vec<ParsedPrediction>, String> {
    let mut rows = raw
        .lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err("predictions artifact is empty".to_string());
    }

    let header = rows.remove(0);
    let expected = [
        "head",
        "relation",
        "tail",
        "true_score",
        "neg_best_score",
        "rank",
    ];
    let columns = header.split(',').map(|v| v.trim()).collect::<Vec<_>>();
    if columns.len() != expected.len() {
        return Err(format!(
            "predictions header has {} columns, expected {}",
            columns.len(),
            expected.len()
        ));
    }

    for (got, want) in columns.iter().zip(expected.iter()) {
        if got != want {
            return Err(format!("predictions header mismatch: got '{got}', expected '{want}'"));
        }
    }

    let mut parsed = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter().enumerate() {
        let fields = row.split(',').map(|field| field.trim()).collect::<Vec<_>>();
        if fields.len() != expected.len() {
            return Err(format!(
                "prediction row {} has {} columns, expected {}",
                idx + 2,
                fields.len(),
                expected.len()
            ));
        }
        let head = fields[0]
            .parse::<i64>()
            .map_err(|e| format!("line {}: head parse failed: {e}", idx + 2))?;
        let relation = fields[1]
            .parse::<i64>()
            .map_err(|e| format!("line {}: relation parse failed: {e}", idx + 2))?;
        let tail = fields[2]
            .parse::<i64>()
            .map_err(|e| format!("line {}: tail parse failed: {e}", idx + 2))?;
        let true_score = fields[3]
            .parse::<f64>()
            .map_err(|e| format!("line {}: true_score parse failed: {e}", idx + 2))?;
        let neg_best_score = fields[4]
            .parse::<f64>()
            .map_err(|e| format!("line {}: neg_best_score parse failed: {e}", idx + 2))?;
        let rank = fields[5]
            .parse::<i64>()
            .map_err(|e| format!("line {}: rank parse failed: {e}", idx + 2))?;

        if !true_score.is_finite() {
            return Err(format!("line {}: true_score is not finite", idx + 2));
        }
        if !neg_best_score.is_finite() {
            return Err(format!("line {}: neg_best_score is not finite", idx + 2));
        }
        if rank < 0 {
            return Err(format!("line {}: rank is negative", idx + 2));
        }

        parsed.push(ParsedPrediction {
            head,
            relation,
            tail,
            true_score,
            neg_best_score,
            rank,
            line_no: idx + 2,
        });
    }
    Ok(parsed)
}

fn parse_mappings(raw: &str) -> Result<MappingSummary, String> {
    let json = serde_json::from_str::<Value>(raw).map_err(|e| format!("invalid mappings JSON: {e}"))?;

    let (entity_count, entity_ids) = match json.get("entities") {
        Some(raw_entities) => {
            let arr = raw_entities
                .as_array()
                .ok_or_else(|| "mappings.entities is not an array".to_string())?;
            let mut ids = HashSet::new();
            for id in arr {
                let value = id
                    .as_i64()
                    .ok_or_else(|| "mappings.entities contains a non-integer entry".to_string())?;
                ids.insert(value);
            }
            let count = ids.len();
            (Some(count), Some(ids))
        }
        None => (None, None),
    };

    let (relation_count, relation_ids) = match json.get("relations") {
        Some(raw_relations) => {
            let arr = raw_relations
                .as_array()
                .ok_or_else(|| "mappings.relations is not an array".to_string())?;
            let mut ids = HashSet::new();
            for id in arr {
                let value = id
                    .as_i64()
                    .ok_or_else(|| "mappings.relations contains a non-integer entry".to_string())?;
                ids.insert(value);
            }
            let count = ids.len();
            (Some(count), Some(ids))
        }
        None => (None, None),
    };

    Ok(MappingSummary {
        entity_count,
        relation_count,
        entity_ids,
        relation_ids,
    })
}

fn deterministic_sample_indices(total: usize, k: usize, seed: i64) -> Vec<usize> {
    if total == 0 || k == 0 {
        return Vec::new();
    }
    let take = if k > total { total } else { k };
    let mut idxs: Vec<usize> = (0..total).collect();
    let mut state = seed as u64 ^ 0x9e3779b97f4a7c15u64;
    for i in (1..total).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state % (i as u64 + 1)) as usize;
        idxs.swap(i, j);
    }
    idxs.truncate(take);
    idxs
}

fn validate_metric(key: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("metric '{key}' is not finite"));
    }
    if key == "mrr" || key.starts_with("hits_at_") {
        if !(0.0..=1.0).contains(&value) {
            return Err(format!("metric '{key}' outside [0,1]: {value}"));
        }
    }
    if key == "mean_rank" && value < 0.0 {
        return Err(format!("metric 'mean_rank' is negative: {value}"));
    }
    Ok(())
}

fn validate_sampled_rows(
    rows: &[ParsedPrediction],
    indices: &[usize],
    cfg: &ValidateConfig,
    mappings: &MappingSummary,
) -> Result<(), String> {
    for &idx in indices {
        let row = &rows[idx];
        if row.rank == 1 && row.neg_best_score - row.true_score > cfg.score_tolerance {
            return Err(format!(
                "line {}: rank=1 but neg_best_score ({}) exceeds true_score ({}) by > score_tolerance {}",
                row.line_no, row.neg_best_score, row.true_score, cfg.score_tolerance
            ));
        }
        if row.rank > 1 && row.true_score - row.neg_best_score > cfg.score_tolerance {
            return Err(format!(
                "line {}: rank={} but true_score ({}) exceeds neg_best_score ({}) by > score_tolerance {}",
                row.line_no, row.rank, row.true_score, row.neg_best_score, cfg.score_tolerance
            ));
        }

        if let Some(entity_ids) = &mappings.entity_ids {
            if !entity_ids.contains(&row.head) {
                return Err(format!(
                    "line {}: head id {} is not present in mappings.entities",
                    row.line_no, row.head
                ));
            }
            if !entity_ids.contains(&row.tail) {
                return Err(format!(
                    "line {}: tail id {} is not present in mappings.entities",
                    row.line_no, row.tail
                ));
            }
        } else if let Some(limit) = mappings.entity_count {
            let max = limit as i64;
            if row.head < 0 || row.head >= max {
                return Err(format!(
                    "line {}: head id {} outside mapped entity range [0,{})",
                    row.line_no, row.head, max
                ));
            }
            if row.tail < 0 || row.tail >= max {
                return Err(format!(
                    "line {}: tail id {} outside mapped entity range [0,{})",
                    row.line_no, row.tail, max
                ));
            }
        }

        if let Some(relation_ids) = &mappings.relation_ids {
            if !relation_ids.contains(&row.relation) {
                return Err(format!(
                    "line {}: relation id {} is not present in mappings.relations",
                    row.line_no, row.relation
                ));
            }
        } else if let Some(limit) = mappings.relation_count {
            let max = limit as i64;
            if row.relation < 0 || row.relation >= max {
                return Err(format!(
                    "line {}: relation id {} outside mapped relation range [0,{})",
                    row.line_no, row.relation, max
                ));
            }
        }
    }
    Ok(())
}

fn validate_artifacts(cfg: &ValidateConfig) -> Result<String, String> {
    if !cfg.run_dir.exists() || !cfg.run_dir.is_dir() {
        return Err(format!("run dir not found: {}", cfg.run_dir.display()));
    }

    let mappings_path = cfg.run_dir.join("mappings");
    let predictions_path = cfg.run_dir.join("predictions");
    let metrics_path = cfg.run_dir.join("metrics.json");
    let manifest_path = cfg.run_dir.join("manifest.json");

    if !mappings_path.exists() {
        return Err(format!("missing mappings artifact: {}", mappings_path.display()));
    }
    if !predictions_path.exists() {
        return Err(format!(
            "missing predictions artifact: {}",
            predictions_path.display()
        ));
    }
    if !manifest_path.exists() {
        return Err(format!(
            "missing run manifest artifact: {}",
            manifest_path.display()
        ));
    }
    if !metrics_path.exists() {
        return Err(format!(
            "missing metrics artifact: {}",
            metrics_path.display()
        ));
    }

    let mappings_raw = read_text(mappings_path)?;
    let mapping_summary = parse_mappings(&mappings_raw)?;

    let predictions_raw = read_text(predictions_path)?;
    let predictions = parse_predictions(&predictions_raw)?;
    if cfg.sample_count > 0 && predictions.len() < cfg.sample_count {
        return Err(format!(
            "predictions rows {} below requested sample_count {}",
            predictions.len(),
            cfg.sample_count
        ));
    }

    let metrics_json = parse_json_file(metrics_path)?;
    let manifest_json = parse_json_file(manifest_path)?;

    let metric_checks = [
        "mean_rank",
        "mrr",
        "hits_at_1",
        "hits_at_3",
        "hits_at_10",
    ];
    if cfg.score_tolerance < 0.0 || cfg.rank_tolerance < 0.0 {
        return Err("score_tolerance and rank_tolerance must be non-negative".to_string());
    }
    for key in metric_checks {
        let value = parse_f64_from_json(&metrics_json, key)
            .ok_or_else(|| format!("metrics.json missing key: {key}"))?;
        validate_metric(key, value)?;
    }

    let total_steps = manifest_json
        .get("total_steps")
        .and_then(Value::as_i64)
        .ok_or_else(|| "manifest.json missing total_steps".to_string())?;
    let total_epochs = manifest_json
        .get("total_epochs")
        .and_then(Value::as_i64)
        .ok_or_else(|| "manifest.json missing total_epochs".to_string())?;
    let run_name = manifest_json
        .get("run_name")
        .and_then(Value::as_str)
        .unwrap_or("unknown");

    if total_steps < 0 || total_epochs < 0 {
        return Err("manifest.json total_steps or total_epochs is negative".to_string());
    }
    let sample_count = if cfg.check_full {
        predictions.len()
    } else if cfg.sample_count == 0 {
        1
    } else {
        cfg.sample_count
    };
    let sample_rows = deterministic_sample_indices(predictions.len(), sample_count, cfg.seed);
    validate_sampled_rows(&predictions, &sample_rows, cfg, &mapping_summary)?;

    let mut report = String::new();
    let _ = writeln!(report, "{{");
    let _ = writeln!(report, "  \"status\": \"ok\",");
    let _ = writeln!(report, "  \"run_dir\": \"{}\",", cfg.run_dir.display());
    let _ = writeln!(report, "  \"run_name\": \"{run_name}\",");
    let _ = writeln!(report, "  \"total_steps\": {total_steps},");
    let _ = writeln!(report, "  \"total_epochs\": {total_epochs},");
    let _ = writeln!(report, "  \"prediction_rows\": {},", predictions.len());
    let _ = writeln!(report, "  \"sampled_rows\": {},", sample_rows.len());
    if let Some(entity_count) = mapping_summary.entity_count {
        let _ = writeln!(report, "  \"mapped_entity_count\": {entity_count},");
    }
    if let Some(relation_count) = mapping_summary.relation_count {
        let _ = writeln!(report, "  \"mapped_relation_count\": {relation_count},");
    }
    let _ = writeln!(
        report,
        "  \"sample_count_requested\": {},",
        cfg.sample_count
    );
    let _ = writeln!(report, "  \"seed\": {},", cfg.seed);
    let _ = writeln!(report, "  \"check_full\": {},", cfg.check_full);
    let _ = writeln!(report, "  \"rank_tolerance\": {},", cfg.rank_tolerance);
    let _ = writeln!(report, "  \"score_tolerance\": {} ", cfg.score_tolerance);
    let _ = writeln!(report, "}}");
    Ok(report)
}

fn main() {
    match parse_args() {
        Ok(cfg) => match validate_artifacts(&cfg) {
            Ok(report) => {
                println!("{report}");
            }
            Err(err) => {
                eprintln!("validator failed: {err}");
                std::process::exit(1);
            }
        },
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}
