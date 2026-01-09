---
title: I/O and Files
description: File operations, directory handling, and standard streams
---

# I/O and Files

The `io` module provides file system operations, process control, environment access, and standard stream I/O. All I/O operations require the `IO` effect.

## Effect Requirements

All I/O functions in this module require the `IO` effect annotation:

```sio
fn process_file(path: &str) -> Result<Data, IoError> with IO {
    let content = read_file(path)?
    // ...
}
```

## Error Type

### IoError

```sio
pub enum IoError {
    NotFound { path: String },
    PermissionDenied { path: String },
    ReadError { path: String, message: String },
    WriteError { path: String, message: String },
    InvalidPath { path: String },
    AlreadyExists { path: String },
    DirectoryNotEmpty { path: String },
    Interrupted,
    Other { message: String },
}
```

#### Constructor Methods

```sio
pub fn not_found(path: String) -> IoError
pub fn permission_denied(path: String) -> IoError
pub fn read_error(path: String, message: String) -> IoError
pub fn write_error(path: String, message: String) -> IoError
pub fn other(message: String) -> IoError
```

#### message

```sio
pub fn message(self) -> String
```

Returns a human-readable error message.

**Example:**

```sio
match read_file("missing.txt") {
    Ok(content) => process(content),
    Err(err) => eprintln("Error: " ++ err.message()),
}
```

## File Operations

### read_file

```sio
pub fn read_file(path: &str) -> Result<String, IoError> with IO
```

Reads the entire contents of a file as a string.

**Parameters:**
- `path` - Path to the file to read

**Returns:** `Ok(content)` on success, `Err(IoError)` on failure.

**Example:**

```sio
fn load_config() -> Result<Config, IoError> with IO {
    let content = read_file("config.toml")?
    parse_config(content)
}
```

### write_file

```sio
pub fn write_file(path: &str, content: &str) -> Result<(), IoError> with IO
```

Writes string content to a file. Creates the file if it doesn't exist, overwrites if it does.

**Parameters:**
- `path` - Path to the file to write
- `content` - Content to write

**Example:**

```sio
fn save_data(data: &str) -> Result<(), IoError> with IO {
    write_file("output.txt", data)
}
```

### append_file

```sio
pub fn append_file(path: &str, content: &str) -> Result<(), IoError> with IO
```

Appends string content to a file. Creates the file if it doesn't exist.

**Parameters:**
- `path` - Path to the file
- `content` - Content to append

**Example:**

```sio
fn log_message(msg: &str) -> Result<(), IoError> with IO {
    let timestamp = get_timestamp()
    append_file("app.log", "[" ++ timestamp ++ "] " ++ msg ++ "\n")
}
```

### file_exists

```sio
pub fn file_exists(path: &str) -> bool with IO
```

Checks if a file exists at the given path.

**Example:**

```sio
if file_exists("config.toml") {
    load_config()
} else {
    use_defaults()
}
```

### remove_file

```sio
pub fn remove_file(path: &str) -> Result<(), IoError> with IO
```

Deletes a file.

**Parameters:**
- `path` - Path to the file to delete

**Example:**

```sio
fn cleanup_temp() -> Result<(), IoError> with IO {
    remove_file("/tmp/myapp.tmp")
}
```

### copy_file

```sio
pub fn copy_file(from: &str, to: &str) -> Result<(), IoError> with IO
```

Copies a file from source to destination.

**Parameters:**
- `from` - Source file path
- `to` - Destination file path

**Example:**

```sio
fn backup_config() -> Result<(), IoError> with IO {
    copy_file("config.toml", "config.toml.bak")
}
```

### rename

```sio
pub fn rename(from: &str, to: &str) -> Result<(), IoError> with IO
```

Renames or moves a file.

**Parameters:**
- `from` - Current file path
- `to` - New file path

**Example:**

```sio
fn process_and_archive(path: &str) -> Result<(), IoError> with IO {
    process_file(path)?
    rename(path, "archive/" ++ path)
}
```

## Directory Operations

### DirEntry

```sio
pub struct DirEntry {
    pub name: String,     // Name of the file or directory
    pub path: String,     // Full path
    pub is_dir: bool,     // Whether this is a directory
    pub is_file: bool,    // Whether this is a file
    pub size: u64,        // File size in bytes (0 for directories)
}
```

### create_dir

```sio
pub fn create_dir(path: &str) -> Result<(), IoError> with IO
```

Creates a directory.

**Parameters:**
- `path` - Path of directory to create

**Example:**

```sio
fn init_project(name: &str) -> Result<(), IoError> with IO {
    create_dir(name)?
    create_dir(name ++ "/src")?
    create_dir(name ++ "/tests")
}
```

### create_dir_all

```sio
pub fn create_dir_all(path: &str) -> Result<(), IoError> with IO
```

Creates a directory and all parent directories if they don't exist.

**Example:**

```sio
fn ensure_output_dir() -> Result<(), IoError> with IO {
    create_dir_all("build/output/data")
}
```

### remove_dir

```sio
pub fn remove_dir(path: &str) -> Result<(), IoError> with IO
```

Removes an empty directory.

**Parameters:**
- `path` - Path of directory to remove

**Errors:** Returns `DirectoryNotEmpty` if the directory is not empty.

### remove_dir_all

```sio
pub fn remove_dir_all(path: &str) -> Result<(), IoError> with IO
```

Removes a directory and all its contents recursively.

**Example:**

```sio
fn clean_build() -> Result<(), IoError> with IO {
    if is_dir("build") {
        remove_dir_all("build")
    } else {
        Ok(())
    }
}
```

### is_dir

```sio
pub fn is_dir(path: &str) -> bool with IO
```

Checks if a path is a directory.

**Example:**

```sio
if is_dir(path) {
    process_directory(path)
} else {
    process_file(path)
}
```

### read_dir

```sio
pub fn read_dir(path: &str) -> Result<Vec<DirEntry>, IoError> with IO
```

Reads the contents of a directory.

**Parameters:**
- `path` - Path of directory to read

**Returns:** Vector of directory entries.

**Example:**

```sio
fn list_files(dir: &str) -> Result<Vec<String>, IoError> with IO {
    let entries = read_dir(dir)?
    var files: Vec<String> = Vec::new()

    for entry in entries {
        if entry.is_file {
            files.push(entry.name)
        }
    }

    Ok(files)
}
```

## File Metadata

### Metadata

```sio
pub struct Metadata {
    pub size: u64,        // File size in bytes
    pub is_file: bool,    // Whether this is a file
    pub is_dir: bool,     // Whether this is a directory
    pub is_symlink: bool, // Whether this is a symbolic link
    pub modified: i64,    // Last modification time (Unix timestamp)
    pub created: i64,     // Creation time (Unix timestamp)
}
```

### metadata

```sio
pub fn metadata(path: &str) -> Result<Metadata, IoError> with IO
```

Gets file metadata.

**Example:**

```sio
fn file_info(path: &str) -> Result<(), IoError> with IO {
    let meta = metadata(path)?

    println("Size: " ++ meta.size.to_string() ++ " bytes")
    println("Modified: " ++ format_timestamp(meta.modified))

    if meta.is_symlink {
        println("(symbolic link)")
    }

    Ok(())
}
```

## Path Utilities

The `path` submodule provides path manipulation functions.

### path::join

```sio
pub fn join(base: &str, component: &str) -> String
```

Joins two path components.

**Example:**

```sio
let full = path::join("dir", "file.txt")  // "dir/file.txt"
let nested = path::join("/home/user", "documents")  // "/home/user/documents"
```

### path::file_name

```sio
pub fn file_name(p: &str) -> Option<String>
```

Extracts the file name from a path.

**Example:**

```sio
path::file_name("/home/user/file.txt")  // Some("file.txt")
path::file_name("/home/user/")          // None
```

### path::parent

```sio
pub fn parent(p: &str) -> Option<String>
```

Gets the parent directory from a path.

**Example:**

```sio
path::parent("/home/user/file.txt")  // Some("/home/user")
path::parent("/")                     // None
```

### path::extension

```sio
pub fn extension(p: &str) -> Option<String>
```

Gets the file extension.

**Example:**

```sio
path::extension("file.txt")      // Some("txt")
path::extension("archive.tar.gz") // Some("gz")
path::extension(".gitignore")    // None
```

## Environment

The `env` submodule provides environment access.

### env::args

```sio
pub fn args() -> Vec<String> with IO
```

Gets command line arguments. The first element is the program name.

**Example:**

```sio
fn main() with IO {
    let args = env::args()

    if args.len() < 2 {
        eprintln("Usage: " ++ args[0] ++ " <input>")
        exit(1)
    }

    let input = args[1]
    process(input)
}
```

### env::var

```sio
pub fn var(key: &str) -> Option<String> with IO
```

Gets an environment variable.

**Example:**

```sio
fn get_home_dir() -> String with IO {
    env::var("HOME").unwrap_or("/tmp")
}
```

### env::set_var

```sio
pub fn set_var(key: &str, value: &str) with IO
```

Sets an environment variable.

**Example:**

```sio
env::set_var("MY_APP_DEBUG", "1")
```

### env::current_dir

```sio
pub fn current_dir() -> Result<String, IoError> with IO
```

Gets the current working directory.

**Example:**

```sio
fn show_cwd() -> Result<(), IoError> with IO {
    let cwd = env::current_dir()?
    println("Current directory: " ++ cwd)
    Ok(())
}
```

## Standard Streams

### print

```sio
pub fn print(s: &str) with IO
```

Prints to standard output without a newline.

**Example:**

```sio
print("Loading...")
// ... do work ...
println("done!")
```

### println

```sio
pub fn println(s: &str) with IO
```

Prints to standard output with a newline.

**Example:**

```sio
println("Hello, world!")
```

### eprint

```sio
pub fn eprint(s: &str) with IO
```

Prints to standard error without a newline.

### eprintln

```sio
pub fn eprintln(s: &str) with IO
```

Prints to standard error with a newline.

**Example:**

```sio
fn log_error(msg: &str) with IO {
    eprintln("[ERROR] " ++ msg)
}
```

### read_line

```sio
pub fn read_line() -> Result<String, IoError> with IO
```

Reads a line from standard input.

**Example:**

```sio
fn prompt(message: &str) -> Result<String, IoError> with IO {
    print(message)
    read_line()
}

fn main() with IO {
    let name = prompt("Enter your name: ")?
    println("Hello, " ++ name)
}
```

## Process Control

### exit

```sio
pub fn exit(code: i32) -> ! with IO
```

Exits the process with the given exit code. This function never returns.

**Parameters:**
- `code` - Exit code (0 for success, non-zero for error)

**Example:**

```sio
fn main() with IO {
    match run_app() {
        Ok(_) => exit(0),
        Err(e) => {
            eprintln("Fatal error: " ++ e.message())
            exit(1)
        }
    }
}
```

## Common Patterns

### Reading and Processing Files

```sio
fn process_lines(path: &str) -> Result<i32, IoError> with IO {
    let content = read_file(path)?
    var count = 0

    for line in content.lines() {
        if !line.is_empty() {
            count = count + 1
        }
    }

    Ok(count)
}
```

### Recursive Directory Traversal

```sio
fn find_all_files(dir: &str, extension: &str) -> Result<Vec<String>, IoError> with IO, Alloc {
    var results: Vec<String> = Vec::new()

    let entries = read_dir(dir)?
    for entry in entries {
        if entry.is_dir {
            let sub_results = find_all_files(&entry.path, extension)?
            for r in sub_results {
                results.push(r)
            }
        } else if entry.name.ends_with(extension) {
            results.push(entry.path)
        }
    }

    Ok(results)
}
```

### Safe File Operations with Cleanup

```sio
fn with_temp_file<F, R>(prefix: &str, f: F) -> Result<R, IoError>
where F: FnOnce(&str) -> Result<R, IoError>
with IO {
    let temp_path = "/tmp/" ++ prefix ++ "_" ++ random_id()

    let result = f(&temp_path)

    // Always try to clean up
    let _ = remove_file(&temp_path)

    result
}
```

### Configuration File Loading

```sio
fn load_config_with_fallback() -> Config with IO {
    // Try user config first
    if let Some(home) = env::var("HOME") {
        let user_config = path::join(&home, ".myapp/config.toml")
        if file_exists(&user_config) {
            if let Ok(content) = read_file(&user_config) {
                if let Ok(config) = parse_config(&content) {
                    return config
                }
            }
        }
    }

    // Fall back to system config
    if file_exists("/etc/myapp/config.toml") {
        if let Ok(content) = read_file("/etc/myapp/config.toml") {
            if let Ok(config) = parse_config(&content) {
                return config
            }
        }
    }

    // Use defaults
    Config::default()
}
```

## See Also

- [Result<T, E>](../core/result.md) - Error handling
- [Vec<T>](../collections/vec.md) - For storing file contents
- [Iterator](../iter.md) - For processing file data
