"""The declared RelocationTable size and every bound check must be the same number."""
import re
import sys

DECLARATION = re.compile(r"entries:\s*\[Relocation;\s*(\d+)\s*\]")
# The table's storage may instead live in parallel BSS columns
# (`pub var NC_RELOC_<FIELD>: [i64; N] = [0; N]`), which keeps the by-value
# RelocationTable header a few words. Then EVERY column is a declaration: each
# one's length and its zero-initialiser must agree with the accessor, because
# add_* writes all of them at the same index.
COLUMN = re.compile(r"pub\s+var\s+(NC_RELOC_\w+):\s*\[i64;\s*(\d+)\s*\]\s*=\s*\[0;\s*(\d+)\s*\]")
ACCESSOR = re.compile(r"fn reloc_table_capacity\(\)\s*->\s*i64\s*\{\s*(\d+)\s*\}")
# A guard written as a bare number instead of the accessor is the defect itself.
LITERAL_GUARD = re.compile(r"if\s+t\.count\s*<\s*(\d+)")
ACCESSOR_GUARD = re.compile(r"if\s+t\.count\s*<\s*reloc_table_capacity\(\)")
# Raising the guard to the declared capacity only moves the cliff. What makes
# falling off it survivable is that the drop is RECORDED and something refuses
# to emit. All three parts are checked, because any one of them alone is the
# silent-drop bug wearing a fix.
OVERFLOW_FIELD = re.compile(r"pub\s+overflow:\s*bool")
OVERFLOW_SET = re.compile(r"\}\s*else\s*\{\s*\n\s*t\.overflow\s*=\s*true")
OVERFLOW_INIT = re.compile(r"out\.overflow\s*=\s*false")


def main(path: str) -> int:
    text = open(path).read()

    declared = DECLARATION.search(text)
    columns = COLUMN.findall(text)
    if declared and columns:
        print("both `entries: [Relocation; N]` and NC_RELOC_* columns declared -- ambiguous storage")
        return 1
    if not declared and not columns:
        print("no `entries: [Relocation; N]` declaration and no NC_RELOC_* columns found -- the pattern moved")
        return 1
    accessor = ACCESSOR.search(text)
    if not accessor:
        print("no reloc_table_capacity() accessor found")
        return 1

    n_accessor = int(accessor.group(1))
    if declared:
        n_declared = int(declared.group(1))
    else:
        sizes = set()
        for name, n_len, n_init in columns:
            print(f"  column {name:<20} [i64; {n_len}] = [0; {n_init}]")
            sizes.add(int(n_len))
            sizes.add(int(n_init))
        if len(sizes) != 1:
            print(f"  DIVERGES: the NC_RELOC_* columns disagree on their length {sorted(sizes)}")
            return 1
        n_declared = sizes.pop()
    literal_guards = LITERAL_GUARD.findall(text)
    accessor_guards = len(ACCESSOR_GUARD.findall(text))

    problems = 0
    print(f"  declared entries    {n_declared}")
    print(f"  accessor returns    {n_accessor}")
    print(f"  guards via accessor {accessor_guards}")
    print(f"  guards via literal  {len(literal_guards)} {literal_guards if literal_guards else ''}")

    if n_declared != n_accessor:
        print(f"  DIVERGES: the array holds {n_declared} but the bound check allows {n_accessor}")
        problems += 1
    if literal_guards:
        # This is how the original defect was written: a number typed into the
        # guard, four times, while the declaration said something else.
        print("  DIVERGES: a bound check uses a literal instead of reloc_table_capacity()")
        problems += 1
    field = len(OVERFLOW_FIELD.findall(text))
    sets = len(OVERFLOW_SET.findall(text))
    init = len(OVERFLOW_INIT.findall(text))
    print(f"  overflow field      {field}")
    print(f"  guards recording it {sets} of {accessor_guards}")
    print(f"  initialised         {init}")
    if field != 1:
        print("  DIVERGES: no `pub overflow: bool` on the table -- a full table drops silently")
        problems += 1
    if init != 1:
        print("  DIVERGES: overflow is never initialised, so its value on a fresh table is undefined")
        problems += 1
    if accessor_guards > 0 and sets != accessor_guards:
        print(f"  DIVERGES: {accessor_guards - sets} guard(s) drop an entry without recording it")
        problems += 1

    if accessor_guards == 0:
        print("  DIVERGES: nothing calls the accessor, so the bound is unenforced")
        problems += 1

    print(f"  problems={problems}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
