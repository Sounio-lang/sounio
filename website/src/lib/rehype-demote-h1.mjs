/**
 * Docs, tutorials, insights and science pages already render a page H1.
 * Markdown ATX H1s in the body become a second H1. Demote those to H2.
 * Fenced code is not a heading node, so shell comments stay comments.
 */
export function rehypeDemoteH1() {
  return (tree) => {
    demote(tree);
  };
}

function demote(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'element' && node.tagName === 'h1') node.tagName = 'h2';
  if (Array.isArray(node.children)) {
    for (const child of node.children) demote(child);
  }
}
