# Generated API Documentation

This directory contains auto-generated API documentation for the Sounio standard library.

## How it's generated

The documentation is automatically generated in CI by:
1. Running `souniodoc generate ../stdlib --output ../target/doc` in the compiler directory
2. Copying the generated HTML to `hugo/static/api/`
3. Deploying with the Hugo site

## Local development

To generate the API docs locally:

```bash
cd compiler
cargo run --bin souniodoc -- generate ../stdlib --output ../hugo/static/api
```

Then serve the Hugo site:

```bash
cd ../hugo
hugo server
```

The API docs will be available at: http://localhost:1313/api/

## Do not commit

The generated files in this directory should NOT be committed to git. They are generated fresh in CI for each deployment.
