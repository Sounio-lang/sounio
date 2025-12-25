//@ compile-fail
//@ error-pattern: effect not declared

fn read_file(path: string) -> string with IO {
    // implementation
    ""
}

// Missing IO effect declaration
fn main() {
    let content = read_file("test.txt")
}
