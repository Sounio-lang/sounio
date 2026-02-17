function walk(node, visitor) {
  if (!node || typeof node !== 'object') return;
  visitor(node);
  if (Array.isArray(node.children)) {
    for (const child of node.children) {
      walk(child, visitor);
    }
  }
}

export default function remarkSioAsRust() {
  return (tree) => {
    walk(tree, (node) => {
      if (node.type !== 'code') return;
      if (typeof node.lang !== 'string') return;
      if (node.lang.toLowerCase() !== 'sio') return;

      node.lang = 'rust';
      node.meta = node.meta ? `${node.meta} original-language=sio` : 'original-language=sio';
    });
  };
}
