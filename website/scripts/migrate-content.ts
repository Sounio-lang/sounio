#!/usr/bin/env npx ts-node
/**
 * Migration script: Hugo content to Astro
 * Converts Hugo markdown files and i18n TOML to Astro format
 */

import * as fs from 'fs';
import * as path from 'path';

const SUPPORTED_LANGUAGES = ['en', 'pt', 'el', 'zh', 'ja', 'es'];

function extractLanguage(filename: string): string {
  const match = filename.match(/\.([a-z]{2})\.md$/);
  return match && SUPPORTED_LANGUAGES.includes(match[1]) ? match[1] : 'en';
}

function parseHugoFrontmatter(content: string): { frontmatter: Record<string, unknown>; body: string } {
  // Match TOML frontmatter between +++ delimiters
  const tomlRegex = /^\+\+\+\s*\n([\s\S]*?)\n\+\+\+\s*\n?/;
  // Match YAML frontmatter between --- delimiters
  const yamlRegex = /^---\s*\n([\s\S]*?)\n---\s*\n?/;

  let frontmatter: Record<string, unknown> = {};
  let body = content;

  const tomlMatch = content.match(tomlRegex);
  if (tomlMatch) {
    frontmatter = parseToml(tomlMatch[1]);
    body = content.replace(tomlRegex, '').trim();
  } else {
    const yamlMatch = content.match(yamlRegex);
    if (yamlMatch) {
      frontmatter = parseYaml(yamlMatch[1]);
      body = content.replace(yamlRegex, '').trim();
    }
  }

  return { frontmatter, body };
}

function parseToml(tomlStr: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  const lines = tomlStr.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const eqIndex = trimmed.indexOf('=');
    if (eqIndex === -1) continue;

    const key = trimmed.slice(0, eqIndex).trim();
    let value = trimmed.slice(eqIndex + 1).trim();

    // Parse value
    if (value.startsWith('"') && value.endsWith('"')) {
      result[key] = value.slice(1, -1);
    } else if (value === 'true') {
      result[key] = true;
    } else if (value === 'false') {
      result[key] = false;
    } else if (!isNaN(Number(value))) {
      result[key] = Number(value);
    } else {
      result[key] = value;
    }
  }

  return result;
}

function parseYaml(yamlStr: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  const lines = yamlStr.split('\n');

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    const colonIndex = trimmed.indexOf(':');
    if (colonIndex === -1) continue;

    const key = trimmed.slice(0, colonIndex).trim();
    let value = trimmed.slice(colonIndex + 1).trim();

    // Parse value
    if (value.startsWith('"') && value.endsWith('"')) {
      result[key] = value.slice(1, -1);
    } else if (value.startsWith("'") && value.endsWith("'")) {
      result[key] = value.slice(1, -1);
    } else if (value === 'true') {
      result[key] = true;
    } else if (value === 'false') {
      result[key] = false;
    } else if (!isNaN(Number(value)) && value !== '') {
      result[key] = Number(value);
    } else {
      result[key] = value;
    }
  }

  return result;
}

function toYamlString(obj: Record<string, unknown>): string {
  const lines: string[] = [];
  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      lines.push(`${key}: "${value.replace(/"/g, '\\"')}"`);
    } else if (typeof value === 'boolean' || typeof value === 'number') {
      lines.push(`${key}: ${value}`);
    } else if (Array.isArray(value)) {
      lines.push(`${key}:`);
      for (const item of value) {
        lines.push(`  - "${item}"`);
      }
    }
  }
  return lines.join('\n');
}

function convertShortcodes(body: string): string {
  // Convert Hugo shortcodes to MDX components
  let result = body;

  // {{< highlight sio >}} ... {{< /highlight >}} -> ```sio ... ```
  result = result.replace(/\{\{<\s*highlight\s+(\w+)\s*>\}\}([\s\S]*?)\{\{<\s*\/highlight\s*>\}\}/g, '```$1$2```');

  // {{< note >}} ... {{< /note >}} -> <Note>...</Note>
  result = result.replace(/\{\{<\s*note\s*>\}\}([\s\S]*?)\{\{<\s*\/note\s*>\}\}/g, '<Note>$1</Note>');

  // {{< warning >}} ... {{< /warning >}} -> <Warning>...</Warning>
  result = result.replace(/\{\{<\s*warning\s*>\}\}([\s\S]*?)\{\{<\s*\/warning\s*>\}\}/g, '<Warning>$1</Warning>');

  // {{< ref "path" >}} -> [](/path)
  result = result.replace(/\{\{<\s*ref\s+"([^"]+)"\s*>\}\}/g, '/$1');

  return result;
}

function migrateFile(inputPath: string, outputBase: string): void {
  const content = fs.readFileSync(inputPath, 'utf8');
  const { frontmatter, body } = parseHugoFrontmatter(content);

  const filename = path.basename(inputPath);
  const lang = extractLanguage(filename);
  const baseName = filename.replace(/\.md$/, '').replace(/\.[a-z]{2}$/, '').replace(/^_/, '');

  // Add language to frontmatter
  const astroFrontmatter = {
    ...frontmatter,
    lang,
  };

  // Convert shortcodes
  const processedBody = convertShortcodes(body);

  // Build output content
  const yamlFm = toYamlString(astroFrontmatter);
  const outputContent = `---\n${yamlFm}\n---\n\n${processedBody}`;

  // Determine output path
  const relDir = path.relative('hugo/content', path.dirname(inputPath));
  const outputDir = path.join(outputBase, lang, relDir);

  fs.mkdirSync(outputDir, { recursive: true });

  const outputPath = path.join(outputDir, `${baseName}.mdx`);
  fs.writeFileSync(outputPath, outputContent);
  console.log(`Migrated: ${inputPath} -> ${outputPath}`);
}

function migrateI18n(inputPath: string, outputBase: string): void {
  const content = fs.readFileSync(inputPath, 'utf8');
  const data = parseToml(content);

  const lang = path.basename(inputPath, '.toml');
  const outputPath = path.join(outputBase, `${lang}.json`);

  fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
  console.log(`Migrated i18n: ${inputPath} -> ${outputPath}`);
}

function walkDir(dir: string, callback: (filePath: string) => void): void {
  if (!fs.existsSync(dir)) {
    console.warn(`Directory not found: ${dir}`);
    return;
  }

  const files = fs.readdirSync(dir);
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      walkDir(fullPath, callback);
    } else if (file.endsWith('.md')) {
      callback(fullPath);
    }
  }
}

function main(): void {
  const projectRoot = path.resolve(__dirname, '..');
  const hugoContent = path.join(projectRoot, '..', 'hugo', 'content');
  const hugoI18n = path.join(projectRoot, '..', 'hugo', 'i18n');
  const outputContent = path.join(projectRoot, 'src', 'content', 'docs');
  const outputI18n = path.join(projectRoot, 'src', 'i18n');

  console.log('Starting migration...');
  console.log(`Hugo content: ${hugoContent}`);
  console.log(`Output: ${outputContent}`);

  // Migrate i18n files
  if (fs.existsSync(hugoI18n)) {
    const files = fs.readdirSync(hugoI18n).filter(f => f.endsWith('.toml'));
    for (const file of files) {
      migrateI18n(path.join(hugoI18n, file), outputI18n);
    }
  }

  // Migrate content files
  walkDir(hugoContent, (filePath) => {
    migrateFile(filePath, outputContent);
  });

  console.log('Migration completed!');
}

main();
