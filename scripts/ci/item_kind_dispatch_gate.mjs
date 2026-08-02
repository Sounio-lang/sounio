import {readFileSync} from 'node:fs';

const ast = readFileSync('self-hosted/parser/ast.sio', 'utf8');
const checker = readFileSync('self-hosted/check/check.sio', 'utf8');
const enumBlock = ast.match(/pub enum ItemKind\s*\{([\s\S]*?)\n\}/);
const dispatchBlock = checker.match(
  /fn check_item\(self, item: Item\)[\s\S]*?match item\.kind\s*\{([\s\S]*?)\n\s*\}\n\s*\}/,
);

if (!enumBlock || !dispatchBlock) {
  throw new Error('unable to locate ItemKind declaration or Checker.check_item');
}

const variants = enumBlock[1]
  .split('\n')
  .map((line) => line.replace(/\/\/.*$/, '').replace(/,$/, '').trim())
  .filter((line) => /^Item[A-Za-z0-9_]+$/.test(line));
const dispatched = [
  ...dispatchBlock[1].matchAll(/ItemKind::(Item[A-Za-z0-9_]+)/g),
].map((match) => match[1]);
const declared = new Set(variants);
const seen = new Set(dispatched);
const missing = variants.filter((variant) => !seen.has(variant));
const unknown = dispatched.filter((variant) => !declared.has(variant));
const duplicates = dispatched.filter(
  (variant, index) => dispatched.indexOf(variant) !== index,
);
const hasDefensiveArm = /^\s*_\s*=>/m.test(dispatchBlock[1]);
const verified =
  variants.length > 0 &&
  missing.length === 0 &&
  unknown.length === 0 &&
  duplicates.length === 0 &&
  hasDefensiveArm;

process.stdout.write(`${JSON.stringify({
  schema: 'sounio.item-kind-dispatch-coverage.v1',
  declaredCount: variants.length,
  dispatchedCount: dispatched.length,
  missing,
  unknown,
  duplicates,
  hasDefensiveArm,
  verified,
}, null, 2)}\n`);

if (!verified) process.exitCode = 1;
