use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::{Config, Context, Editor, Result as RustyResult};

const VERSION: &str = "0.2.0";

struct ReplHelper;

impl Completer for ReplHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> RustyResult<(usize, Vec<Pair>)> {
        let commands = vec![
            ":help", ":h", ":quit", ":q", ":reset", ":show", ":time", ":load", ":save",
        ];

        let mut completions = Vec::new();

        // If the line starts with ':', complete commands
        if line.starts_with(':') {
            let prefix = &line[..pos];
            for cmd in &commands {
                if cmd.starts_with(prefix) {
                    completions.push(Pair {
                        display: cmd.to_string(),
                        replacement: cmd.to_string(),
                    });
                }
            }
            return Ok((0, completions));
        }

        // If we're in a :load or :save context, try file completion
        // This is a simple approximation
        if line.starts_with(":load ") || line.starts_with(":save ") {
            let prefix = &line[6..pos];
            if let Ok(entries) = std::fs::read_dir(".") {
                for entry in entries.flatten() {
                    if let Ok(name) = entry.file_name().into_string() {
                        if name.starts_with(prefix) {
                            completions.push(Pair {
                                display: name.clone(),
                                replacement: name,
                            });
                        }
                    }
                }
            }
            return Ok((6, completions));
        }

        Ok((pos, completions))
    }
}

impl rustyline::Helper for ReplHelper {}
impl rustyline::hint::Hinter for ReplHelper {
    type Hint = String;
}
impl rustyline::validate::Validator for ReplHelper {}
impl rustyline::highlight::Highlighter for ReplHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> std::borrow::Cow<'b, str> {
        let _ = default;
        std::borrow::Cow::Borrowed(prompt)
    }
}

struct Repl {
    compiler_path: PathBuf,
    session: Session,
    tmp_counter: u64,
    show_timing: bool,
    history_file: PathBuf,
    session_file: PathBuf,
}

struct Session {
    declarations: Vec<String>,
}

impl Session {
    fn new() -> Self {
        Session {
            declarations: Vec::new(),
        }
    }

    fn source(&self) -> String {
        self.declarations.join("\n")
    }

    fn reset(&mut self) {
        self.declarations.clear();
    }

    fn push(&mut self, decl: String) {
        self.declarations.push(decl);
    }

    fn is_empty(&self) -> bool {
        self.declarations.is_empty()
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, self.source())
    }

    fn load(&mut self, path: &Path) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        self.declarations = content.lines().map(|s| s.to_string()).collect();
        Ok(())
    }
}

#[derive(Debug)]
enum ReplError {
    Compile(String),
    Io(std::io::Error),
    Quit,
}

impl From<std::io::Error> for ReplError {
    fn from(e: std::io::Error) -> Self {
        ReplError::Io(e)
    }
}

fn filter_compiler_output(output: &str) -> String {
    let mut result = Vec::new();
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("error:")
            || trimmed.starts_with("warning:")
            || trimmed.starts_with("typecheck: failed")
            || (trimmed.starts_with("E") && trimmed.len() > 3 && trimmed.as_bytes()[3] == b' ')
        {
            result.push(trimmed.to_string());
        }
    }
    if result.is_empty() {
        let trimmed = output.trim();
        if trimmed.len() > 500 {
            format!("{}\n... (truncated)", &trimmed[..500])
        } else {
            trimmed.to_string()
        }
    } else {
        result.join("\n")
    }
}

impl Repl {
    fn new(compiler_path: PathBuf) -> Self {
        let data_dir = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"))
            .join(".local/share/sounio-repl");
        let _ = std::fs::create_dir_all(&data_dir);
        let history_file = data_dir.join("history");
        let session_file = data_dir.join("session.sio");

        let mut repl = Repl {
            compiler_path,
            session: Session::new(),
            tmp_counter: 0,
            show_timing: false,
            history_file,
            session_file,
        };

        if repl.session_file.exists() {
            if let Err(e) = repl.session.load(&repl.session_file) {
                eprintln!("warn: failed to restore session: {}", e);
            }
        }

        repl
    }

    fn tmp_path(&mut self, suffix: &str) -> PathBuf {
        let path = PathBuf::from(format!("/tmp/sounio-repl-{}-{}", self.tmp_counter, suffix));
        self.tmp_counter += 1;
        path
    }

    fn is_declaration(input: &str) -> bool {
        let trimmed = input.trim();
        trimmed.starts_with("fn ")
            || trimmed.starts_with("let ")
            || trimmed.starts_with("var ")
            || trimmed.starts_with("use ")
            || trimmed.starts_with("struct ")
            || trimmed.starts_with("pub ")
    }

    fn compile_source_str(&mut self, src: &str) -> Result<(), ReplError> {
        let src_path = self.tmp_path("check.sio");
        let out_path = self.tmp_path("check.out");
        std::fs::write(&src_path, src)?;
        let output = Command::new(&self.compiler_path)
            .arg(&src_path)
            .arg(&out_path)
            .output()?;
        let _ = std::fs::remove_file(&src_path);
        let _ = std::fs::remove_file(&out_path);
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(ReplError::Compile(filter_compiler_output(&stdout)));
        }
        Ok(())
    }

    fn compile_and_run(&mut self, src: &str) -> Result<String, ReplError> {
        let src_path = self.tmp_path("src.sio");
        let out_path = self.tmp_path("out");
        std::fs::write(&src_path, src)?;
        let output = Command::new(&self.compiler_path)
            .arg(&src_path)
            .arg(&out_path)
            .output()?;
        let _ = std::fs::remove_file(&src_path);
        if !output.status.success() {
            let _ = std::fs::remove_file(&out_path);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(ReplError::Compile(filter_compiler_output(&stdout)));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&out_path)?.permissions();
            perms.set_mode(perms.mode() | 0o755);
            std::fs::set_permissions(&out_path, perms)?;
        }

        let run_output = Command::new(&out_path).output()?;
        let _ = std::fs::remove_file(&out_path);
        if !run_output.status.success() {
            let stdout = String::from_utf8_lossy(&run_output.stdout);
            if !stdout.is_empty() {
                return Ok(stdout.trim().to_string());
            }
            let stderr = String::from_utf8_lossy(&run_output.stderr);
            return Err(ReplError::Compile(format!(
                "Program exited with code {:?}\n{}",
                run_output.status.code(),
                stderr.trim()
            )));
        }
        let stdout = String::from_utf8_lossy(&run_output.stdout);
        Ok(stdout.trim().to_string())
    }

    fn eval_declaration(&mut self, decl: &str) -> Result<String, ReplError> {
        let src = format!(
            "{}\n{}\nfn main() -> i64 {{ 0 }}\n",
            self.session.source(),
            decl
        );
        self.compile_source_str(&src)?;
        self.session.push(decl.to_string());
        let _ = self.session.save(&self.session_file);
        Ok("OK".to_string())
    }

    fn eval_expression(&mut self, expr: &str) -> Result<String, ReplError> {
        let src_int = format!(
            "{}\nfn main() -> i64 with IO, Mut, Panic, Div, Alloc {{\n    print_int({});\n    print_char(10);\n    0\n}}\n",
            self.session.source(),
            expr
        );
        match self.compile_and_run(&src_int) {
            Ok(result) => return Ok(result),
            Err(ReplError::Compile(msg)) => {
                if msg.contains("type mismatch")
                    || msg.contains("typecheck: failed")
                    || msg.contains("E200")
                {
                    let src_f64 = format!(
                        "{}\nfn main() -> i64 with IO, Mut, Panic, Div, Alloc {{\n    print_f64({});\n    print_char(10);\n    0\n}}\n",
                        self.session.source(),
                        expr
                    );
                    if let Ok(result) = self.compile_and_run(&src_f64) {
                        return Ok(result);
                    }
                }
                return Err(ReplError::Compile(msg));
            }
            Err(e) => return Err(e),
        }
    }

    fn run_command(&mut self, cmd: &str) -> Result<String, ReplError> {
        let parts: Vec<&str> = cmd.trim().split_whitespace().collect();
        if parts.is_empty() {
            return Ok(String::new());
        }
        match parts[0] {
            ":q" | ":quit" => Err(ReplError::Quit),
            ":reset" => {
                self.session.reset();
                let _ = self.session.save(&self.session_file);
                Ok("Session reset.".to_string())
            }
            ":show" => {
                if self.session.is_empty() {
                    Ok("(empty session)".to_string())
                } else {
                    Ok(self.session.source())
                }
            }
            ":time" => {
                self.show_timing = !self.show_timing;
                Ok(format!(
                    "Timing {}.",
                    if self.show_timing { "enabled" } else { "disabled" }
                ))
            }
            ":load" => {
                if parts.len() < 2 {
                    return Ok("Usage: :load <file.sio>".to_string());
                }
                let path = parts[1];
                let content = std::fs::read_to_string(path)?;
                let src = format!(
                    "{}\n{}\nfn main() -> i64 {{ 0 }}\n",
                    self.session.source(),
                    content
                );
                self.compile_source_str(&src)?;
                self.session.push(content);
                let _ = self.session.save(&self.session_file);
                Ok(format!("Loaded {}.", path))
            }
            ":save" => {
                let path = if parts.len() >= 2 {
                    PathBuf::from(parts[1])
                } else {
                    PathBuf::from("repl_session.sio")
                };
                self.session.save(&path)?;
                Ok(format!("Session saved to {}.", path.display()))
            }
            ":help" | ":h" => {
                Ok(r#"Sounio REPL Commands:
  :quit, :q       Exit the REPL
  :reset          Clear the session
  :show           Show current session declarations
  :load <file>    Load a .sio file into the session
  :save [file]    Save session to a file (default: repl_session.sio)
  :time           Toggle compilation/evaluation timing
  :help, :h       Show this help

Enter declarations (fn, let, var, struct, use) or expressions.
Expressions are automatically wrapped in main() and evaluated.
"#.to_string())
            }
            _ => Ok(format!("Unknown command: {}", parts[0])),
        }
    }

    fn is_balanced(input: &str) -> bool {
        let mut braces = 0i32;
        let mut parens = 0i32;
        let mut brackets = 0i32;
        let mut in_string = false;
        let mut escape = false;
        for ch in input.chars() {
            if escape {
                escape = false;
                continue;
            }
            if ch == '\\' && in_string {
                escape = true;
                continue;
            }
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            match ch {
                '{' => braces += 1,
                '}' => braces -= 1,
                '(' => parens += 1,
                ')' => parens -= 1,
                '[' => brackets += 1,
                ']' => brackets -= 1,
                _ => {}
            }
        }
        braces == 0 && parens == 0 && brackets == 0
    }

    fn run_repl(&mut self) -> RustyResult<()> {
        println!("Sounio REPL v{}", VERSION);
        println!("Type :help for commands, :quit to exit.");
        if !self.session.is_empty() {
            println!(
                "({} declarations restored from previous session)",
                self.session.declarations.len()
            );
        }
        println!();

        let config = Config::builder().tab_stop(4).build();
        let helper = ReplHelper;
        let mut rl: Editor<ReplHelper, rustyline::history::DefaultHistory> =
            Editor::with_config(config)?;
        rl.set_helper(Some(helper));
        let _ = rl.load_history(&self.history_file);

        let mut multi_line_buffer = String::new();
        let mut in_multi_line = false;

        loop {
            let prompt = if in_multi_line { "....> " } else { "sounio> " };
            let readline = rl.readline(prompt);

            match readline {
                Ok(line) => {
                    let trimmed = line.trim_end();
                    let explicit_continue = trimmed.ends_with('\\');
                    let line_to_add = if explicit_continue {
                        &trimmed[..trimmed.len() - 1]
                    } else {
                        trimmed
                    };

                    if in_multi_line {
                        multi_line_buffer.push('\n');
                        multi_line_buffer.push_str(line_to_add);
                    } else {
                        multi_line_buffer.clear();
                        multi_line_buffer.push_str(line_to_add);
                    }

                    if explicit_continue {
                        in_multi_line = true;
                        continue;
                    }

                    if !Repl::is_balanced(&multi_line_buffer) {
                        in_multi_line = true;
                        continue;
                    }

                    in_multi_line = false;
                    let input = multi_line_buffer.trim().to_string();
                    if input.is_empty() {
                        continue;
                    }

                    let _ = rl.add_history_entry(&input);

                    let start = Instant::now();
                    let result = if input.starts_with(':') {
                        self.run_command(&input)
                    } else if Self::is_declaration(&input) {
                        self.eval_declaration(&input)
                    } else {
                        self.eval_expression(&input)
                    };
                    let elapsed = start.elapsed();

                    match result {
                        Ok(output) => {
                            if !output.is_empty() {
                                println!("{}", output);
                            }
                        }
                        Err(ReplError::Compile(msg)) => {
                            eprintln!("Error: {}", msg.trim());
                        }
                        Err(ReplError::Io(e)) => {
                            eprintln!("IO error: {}", e);
                        }
                        Err(ReplError::Quit) => {
                            println!("Bye!");
                            break;
                        }
                    }

                    if self.show_timing {
                        println!("[{:?}]", elapsed);
                    }
                }
                Err(ReadlineError::Interrupted) => {
                    println!("^C");
                    if in_multi_line {
                        in_multi_line = false;
                        multi_line_buffer.clear();
                        continue;
                    }
                    break;
                }
                Err(ReadlineError::Eof) => {
                    println!();
                    break;
                }
                Err(err) => {
                    eprintln!("Error: {:?}", err);
                    break;
                }
            }
        }

        let _ = rl.save_history(&self.history_file);
        let _ = self.session.save(&self.session_file);
        Ok(())
    }
}

fn find_compiler() -> PathBuf {
    if let Ok(path) = std::env::var("SOUNIO_COMPILER") {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let repo_bin = dir.join("../../../../bin/souc-linux-x86_64");
            if repo_bin.exists() {
                return repo_bin.canonicalize().unwrap_or(repo_bin);
            }
        }
    }

    if let Ok(output) = Command::new("which").arg("souc").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                let wrapper = PathBuf::from(&path);
                if let Some(dir) = wrapper.parent() {
                    let native = dir.join("souc-linux-x86_64");
                    if native.exists() {
                        return native;
                    }
                }
                return wrapper;
            }
        }
    }

    for path in &[
        "/workspace/sounio/bin/souc-linux-x86_64",
        "./bin/souc-linux-x86_64",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return p;
        }
    }

    PathBuf::from("souc-linux-x86_64")
}

fn main() {
    let compiler_path = find_compiler();
    if !compiler_path.exists() {
        eprintln!("Error: Cannot find souc compiler binary.");
        eprintln!("Set SOUNIO_COMPILER or ensure souc is in PATH.");
        std::process::exit(1);
    }

    let mut repl = Repl::new(compiler_path);
    if let Err(e) = repl.run_repl() {
        eprintln!("REPL error: {}", e);
        std::process::exit(1);
    }
}
