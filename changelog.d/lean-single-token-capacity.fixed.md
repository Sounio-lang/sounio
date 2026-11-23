The lean_single seed now holds 4,194,304 tokens, up from 2,097,152, and refuses
a source with more tokens: "error: source too large: the token buffer ... filled
before the end of the source". Before, it parsed the truncated token stream, and
the result was a parse error at a line with nothing wrong. The Madaros source
tree was within about 2,300 tokens of the old limit.
