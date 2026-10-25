// Visible text of generated HTML. Script and style elements drop out with
// their contents; every other tag becomes a space. This is a scan, not a
// regular expression that tries to match a tag: those miss end tags such as
// `</script >` and can be put back together by a single replacement.

function isWordChar(ch) {
  const code = ch.charCodeAt(0);
  return (code >= 48 && code <= 57) || (code >= 65 && code <= 90) || (code >= 97 && code <= 122) || code === 95;
}

function isNameChar(ch) {
  return isWordChar(ch) || ch === '-' || ch === ':';
}

function isSpace(ch) {
  return ch === ' ' || ch === '\n' || ch === '\r' || ch === '\t' || ch === '\f';
}

function tagName(body) {
  const trimmed = body.trim();
  const closing = trimmed.startsWith('/');
  const source = closing ? trimmed.slice(1).trimStart() : trimmed;
  let length = 0;
  while (length < source.length && isNameChar(source[length])) length += 1;
  return { closing, name: source.slice(0, length).toLowerCase() };
}

function closingTagEnd(html, lower, from, name) {
  const marker = `</${name}`;
  let search = from;
  while (search < html.length) {
    const found = lower.indexOf(marker, search);
    if (found < 0) return -1;
    let cursor = found + marker.length;
    if (cursor < html.length && isNameChar(html[cursor])) {
      search = found + 1;
      continue;
    }
    while (cursor < html.length && isSpace(html[cursor])) cursor += 1;
    if (html[cursor] === '>') return cursor + 1;
    const gt = html.indexOf('>', cursor);
    if (gt < 0 || html.slice(cursor, gt).includes('<')) {
      search = found + 1;
      continue;
    }
    return gt + 1;
  }
  return -1;
}

export function htmlVisibleText(html, { collapseWhitespace = false } = {}) {
  let out = '';
  let index = 0;
  const lower = html.toLowerCase();
  while (index < html.length) {
    const lt = html.indexOf('<', index);
    if (lt < 0) {
      out += html.slice(index);
      break;
    }
    out += html.slice(index, lt);
    const gt = html.indexOf('>', lt + 1);
    if (gt < 0) {
      out += html.slice(lt);
      break;
    }
    const { closing, name } = tagName(html.slice(lt + 1, gt));
    if (!closing && (name === 'script' || name === 'style')) {
      const end = closingTagEnd(html, lower, gt + 1, name);
      if (end >= 0) {
        index = end;
        continue;
      }
    }
    out += ' ';
    index = gt + 1;
  }
  return collapseWhitespace ? out.replace(/\s+/g, ' ') : out;
}
