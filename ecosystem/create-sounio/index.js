#!/usr/bin/env node
'use strict'

const fs = require('fs')
const path = require('path')

const name = process.argv[2]

if (!name) {
  console.error('Usage: create-sounio <project-name>')
  process.exit(1)
}

if (fs.existsSync(name)) {
  console.error(`Error: directory '${name}' already exists`)
  process.exit(1)
}

const tmpl = path.join(__dirname, 'templates')
const dest = path.join(process.cwd(), name)

fs.mkdirSync(dest)

// Copy templates (including dotfiles)
for (const f of fs.readdirSync(tmpl)) {
  fs.copyFileSync(path.join(tmpl, f), path.join(dest, f))
}

console.log(`\nCreated Sounio project: ${name}/`)
console.log('\nNext steps:')
console.log(`  cd ${name}`)
console.log('  souc run main.sio')
console.log('\nDocs: https://pages.souniolang.org')
