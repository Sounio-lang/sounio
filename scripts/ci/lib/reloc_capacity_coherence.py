"""The declared RelocationTable size and every bound check must be the same number."""
import re
import sys

DECLARATION = re.compile(r"entries:\s*\[Relocation;\s*(\d+)\s*\]")
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
    if not declared:
        print("no `entries: [Relocation; N]` declaration found -- the pattern moved")
        return 1
    accessor = ACCESSOR.search(text)
    if not accessor:
        print("no reloc_table_capacity() accessor found")
        return 1

    n_declared = int(declared.group(1))
    n_accessor = int(accessor.group(1))
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
    # The file now carries a SECOND overflow (ExternRelocTable, dynlink fail-
    # closed). Its field is `pub overflow: bool` too, and its fresh-table init
    # `out.overflow = false` is textually identical, so count file-wide and the
    # field==1 / init==1 checks broke. Scope the FIELD to the RelocationTable
    # struct and the INIT to reloc_table_new(); the SET pattern (t.overflow=true
    # vs (*table).overflow=true) already distinguishes the two tables.
    struct_block = ""
    m = re.search(r"pub struct RelocationTable \{(.*?)\n\}", text, re.DOTALL)
    if m:
        struct_block = m.group(0)
    init_block = ""
    m2 = re.search(r"fn reloc_table_new\(\) -> RelocationTable with Mut \{(.*?)\n\}", text, re.DOTALL)
    if m2:
        init_block = m2.group(0)
    field = len(OVERFLOW_FIELD.findall(struct_block))
    sets = len(OVERFLOW_SET.findall(text))
    init = len(OVERFLOW_INIT.findall(init_block))
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
