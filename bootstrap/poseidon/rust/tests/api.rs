mod common;

use poseidon_wrapper::{Module, PoseidonError, VmConfig};

#[test]
fn from_bytes_runs_add_fixture() {
    let fixture = common::fixture_by_id("add");
    let module = Module::from_bytes(&common::read_fixture_bytes(&fixture)).expect("module");
    let result = module.run(&VmConfig::default()).expect("run result");
    assert_eq!(result.exit_code, 42);
    assert!(result.steps_executed > 0);
}

#[test]
fn from_file_runs_return_fixture() {
    let fixture = common::fixture_by_id("return_const");
    let module = Module::from_file(common::fixture_path(&fixture)).expect("module");
    let result = module.run(&VmConfig::default()).expect("run result");
    assert_eq!(result.exit_code, 42);
}

#[test]
fn missing_main_is_reported() {
    let fixture = common::fixture_by_id("missing_main");
    let module = Module::from_bytes(&common::read_fixture_bytes(&fixture)).expect("module");
    let err = module.run(&VmConfig::default()).expect_err("missing main");
    assert_eq!(err, PoseidonError::NoMainFunction);
}

#[test]
fn loader_invalid_fixtures_report_expected_status() {
    for fixture in common::fixtures_by_kind("loader_invalid") {
        let err = Module::from_bytes(&common::read_fixture_bytes(&fixture)).expect_err("loader-invalid fixture should fail");
        let expected = match fixture.expected_status.as_str() {
            "invalid_bytecode" => PoseidonError::InvalidBytecode,
            "unsupported_version" => PoseidonError::UnsupportedVersion,
            other => panic!("unexpected loader-invalid status {other}"),
        };
        assert_eq!(err, expected, "fixture {}", fixture.id);
    }
}
