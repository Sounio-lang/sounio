//! ODEP CLI: reads an `UnlearnClaim` in JSON from stdin, writes the
//! envelope JSON to stdout, and exits 0 on success / non-zero on
//! constraint violation.

use odep_prover::{prove, UnlearnClaim};
use std::io::{self, Read, Write};

fn main() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let claim: UnlearnClaim = serde_json::from_str(&input)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let regulation = std::env::var("ODEP_REGULATION").unwrap_or_else(|_| "GDPR-Art17".into());
    match prove(&claim, &regulation) {
        Ok(env) => {
            let out = serde_json::to_string_pretty(&env)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            writeln!(io::stdout(), "{out}")?;
            Ok(())
        }
        Err(e) => {
            writeln!(io::stderr(), "ODEP rejected: {e:?}")?;
            std::process::exit(1);
        }
    }
}
