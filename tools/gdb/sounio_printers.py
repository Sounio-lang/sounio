#!/usr/bin/env python3
"""
Sounio GDB Pretty Printers

This module provides GDB pretty printers for Sounio's core types:
- Knowledge<T>: Epistemic values with confidence, provenance, and ontology
- Quantity<T, U>: Numeric values with units of measure
- Effect types: IO, Mut, Alloc, Async, etc.

Usage:
    (gdb) source sounio_printers.py
    (gdb) print my_knowledge_value
    Knowledge<f64> { value: 3.14159, confidence: 0.95, source: Measurement, ontology: PATO:mass }

Or add to ~/.gdbinit:
    python
    import sys
    sys.path.insert(0, '/path/to/sounio/tools/gdb')
    import sounio_printers
    sounio_printers.register_printers(None)
    end
"""

import gdb
import re
from typing import Optional, Iterator, Tuple


# =============================================================================
# Sounio DWARF attribute constants
# =============================================================================

DW_AT_SOUNIO_EFFECTS = 0x3000
DW_AT_SOUNIO_UNIT = 0x3001
DW_AT_SOUNIO_EPSILON = 0x3002
DW_AT_SOUNIO_PROVENANCE = 0x3003
DW_AT_SOUNIO_ONTOLOGY = 0x3004
DW_AT_SOUNIO_LINEAR = 0x3005


# =============================================================================
# Helper functions
# =============================================================================

def get_basic_type(val: gdb.Value) -> str:
    """Get the basic type name, stripping qualifiers."""
    t = val.type
    if t.code == gdb.TYPE_CODE_PTR:
        t = t.target()
    return t.unqualified().strip_typedefs().tag or str(t)


def is_sounio_type(typename: str, prefix: str) -> bool:
    """Check if a type name matches a Sounio type pattern."""
    return typename.startswith(prefix) or f"::{prefix}" in typename


def extract_template_arg(typename: str) -> Optional[str]:
    """Extract the first template argument from a type name."""
    match = re.search(r'<([^<>]+)>', typename)
    if match:
        return match.group(1).split(',')[0].strip()
    return None


def format_confidence(conf: float) -> str:
    """Format confidence as a percentage or special value."""
    if conf >= 0.999:
        return "certain"
    elif conf >= 0.95:
        return f"{conf:.1%}"
    elif conf >= 0.5:
        return f"{conf:.0%}"
    elif conf <= 0.01:
        return "uncertain"
    else:
        return f"{conf:.1%}"


def format_unit_dimension(dim: list) -> str:
    """Format a dimension vector as a unit string."""
    # dim = [M, L, T, I, Theta, N, J]
    units = ['kg', 'm', 's', 'A', 'K', 'mol', 'cd']
    parts = []

    for i, (exp, unit) in enumerate(zip(dim, units)):
        if exp == 0:
            continue
        elif exp == 1:
            parts.append(unit)
        elif exp > 0:
            parts.append(f"{unit}^{exp}")
        else:
            parts.append(f"{unit}^({exp})")

    if not parts:
        return "dimensionless"

    return " ".join(parts)


def gdb_evaluate_expression(expr_string: str) -> gdb.Value:
    """Safely evaluate a GDB expression string."""
    # Use gdb.parse_and_eval which is the standard GDB Python API
    # This evaluates expressions in the debugger context, not Python
    return gdb.parse_and_eval(expr_string)


# =============================================================================
# Knowledge<T> Pretty Printer
# =============================================================================

class KnowledgePrinter:
    """
    Pretty printer for Sounio Knowledge<T> types.

    Knowledge is the core epistemic type that tracks:
    - The wrapped value
    - Confidence level (0.0 to 1.0)
    - Source/provenance
    - Ontology binding
    """

    def __init__(self, val: gdb.Value):
        self.val = val
        self.typename = str(val.type)

    def to_string(self) -> str:
        try:
            # Try to access the value field
            value = self.val['value']

            # Try to get confidence
            confidence = self._get_confidence()

            # Try to get source
            source = self._get_source()

            # Try to get ontology
            ontology = self._get_ontology()

            # Build the output
            content_type = extract_template_arg(self.typename) or "?"
            parts = [f"Knowledge<{content_type}>"]
            parts.append("{")
            parts.append(f"value: {value}")

            if confidence is not None:
                parts.append(f", confidence: {format_confidence(confidence)}")

            if source:
                parts.append(f", source: {source}")

            if ontology:
                parts.append(f", ontology: {ontology}")

            parts.append("}")

            return " ".join(parts)

        except gdb.error as e:
            return f"Knowledge<...> {{ <error: {e}> }}"

    def _get_confidence(self) -> Optional[float]:
        """Extract confidence value."""
        try:
            # Try common field names
            for field_name in ['confidence', 'epsilon', 'conf', 'epistemic']:
                try:
                    conf_field = self.val[field_name]
                    # Handle nested structures
                    if conf_field.type.code == gdb.TYPE_CODE_STRUCT:
                        conf_field = conf_field['value']
                    return float(conf_field)
                except (gdb.error, KeyError):
                    continue
            return None
        except:
            return None

    def _get_source(self) -> Optional[str]:
        """Extract source/provenance."""
        try:
            for field_name in ['source', 'provenance', 'origin']:
                try:
                    source = self.val[field_name]
                    # Handle enum types
                    if source.type.code == gdb.TYPE_CODE_ENUM:
                        return str(source)
                    elif source.type.code == gdb.TYPE_CODE_STRUCT:
                        # Nested provenance struct
                        return str(source['kind']) if 'kind' in source.type else None
                    return str(source)
                except (gdb.error, KeyError):
                    continue
            return None
        except:
            return None

    def _get_ontology(self) -> Optional[str]:
        """Extract ontology binding."""
        try:
            for field_name in ['ontology', 'domain', 'binding']:
                try:
                    ont = self.val[field_name]
                    if ont.type.code == gdb.TYPE_CODE_PTR:
                        if int(ont) == 0:
                            return None
                        return ont.string()
                    return str(ont)
                except (gdb.error, KeyError):
                    continue
            return None
        except:
            return None

    def children(self) -> Iterator[Tuple[str, gdb.Value]]:
        """Yield child elements for detailed inspection."""
        try:
            for field in self.val.type.fields():
                yield (field.name, self.val[field.name])
        except:
            pass

    def display_hint(self) -> str:
        return 'map'


# =============================================================================
# Quantity<T, U> Pretty Printer
# =============================================================================

class QuantityPrinter:
    """
    Pretty printer for Sounio Quantity<T, U> types.

    Quantity represents a numeric value with compile-time unit checking:
    - The numeric value
    - Unit symbol (e.g., "mg", "L", "s")
    - Dimension vector for verification
    """

    def __init__(self, val: gdb.Value):
        self.val = val
        self.typename = str(val.type)

    def to_string(self) -> str:
        try:
            # Get the numeric value
            value = self.val['value']

            # Try to get unit symbol
            unit = self._get_unit()

            # Format output
            if unit:
                return f"{value} {unit}"
            else:
                # Fall back to dimension vector if available
                dim = self._get_dimension()
                if dim:
                    return f"{value} [{format_unit_dimension(dim)}]"
                return str(value)

        except gdb.error as e:
            return f"Quantity {{ <error: {e}> }}"

    def _get_unit(self) -> Optional[str]:
        """Extract unit symbol from type or DWARF info."""
        try:
            # Try to get from SYMBOL constant
            for field_name in ['_unit', 'unit', 'symbol']:
                try:
                    unit = self.val[field_name]
                    if unit.type.code == gdb.TYPE_CODE_PTR:
                        return unit.string()
                    return str(unit)
                except (gdb.error, KeyError):
                    continue

            # Try to extract from type name
            # e.g., "Quantity<f64, Milligram>" -> "mg"
            match = re.search(r'Quantity<[^,]+,\s*(\w+)>', self.typename)
            if match:
                unit_type = match.group(1)
                return self._unit_type_to_symbol(unit_type)

            return None
        except:
            return None

    def _unit_type_to_symbol(self, unit_type: str) -> str:
        """Convert unit type name to symbol."""
        unit_map = {
            # Mass
            'Kilogram': 'kg', 'Gram': 'g', 'Milligram': 'mg',
            'Microgram': 'ug', 'Nanogram': 'ng',
            # Length
            'Meter': 'm', 'Centimeter': 'cm', 'Millimeter': 'mm',
            'Micrometer': 'um', 'Nanometer': 'nm',
            # Time
            'Second': 's', 'Millisecond': 'ms', 'Microsecond': 'us',
            'Minute': 'min', 'Hour': 'h', 'Day': 'd',
            # Volume
            'Liter': 'L', 'Milliliter': 'mL', 'Microliter': 'uL',
            # Amount
            'Mole': 'mol', 'Millimole': 'mmol', 'Micromole': 'umol',
            # Concentration
            'MolarConcentration': 'M', 'MilliMolar': 'mM',
            # Other
            'Dimensionless': '', 'Percent': '%',
        }
        return unit_map.get(unit_type, unit_type)

    def _get_dimension(self) -> Optional[list]:
        """Extract dimension vector."""
        try:
            for field_name in ['dimension', 'dim', '_dim']:
                try:
                    dim = self.val[field_name]
                    # Convert to list
                    return [int(dim[i]) for i in range(7)]
                except (gdb.error, KeyError):
                    continue
            return None
        except:
            return None


# =============================================================================
# Effect Type Pretty Printer
# =============================================================================

class EffectPrinter:
    """
    Pretty printer for Sounio Effect types.

    Effects track what side effects a function can perform:
    - IO: File/network I/O
    - Mut: Mutable state
    - Alloc: Memory allocation
    - Panic: May panic
    - Async: Asynchronous operation
    - GPU: GPU computation
    - Prob: Probabilistic computation
    - Div: May diverge
    """

    EFFECT_SYMBOLS = {
        'IO': 'IO',
        'Mut': 'Mut',
        'Alloc': 'Alloc',
        'Panic': 'Panic',
        'Async': 'Async',
        'GPU': 'GPU',
        'Prob': 'Prob',
        'Div': 'Div',
        'Pure': 'Pure',
    }

    def __init__(self, val: gdb.Value):
        self.val = val
        self.typename = str(val.type)

    def to_string(self) -> str:
        try:
            effects = self._get_effects()
            if not effects:
                return "Pure"
            return " + ".join(effects)
        except gdb.error as e:
            return f"Effect {{ <error: {e}> }}"

    def _get_effects(self) -> list:
        """Extract effect list from the type."""
        effects = []

        # Try to read effect flags
        for effect_name, symbol in self.EFFECT_SYMBOLS.items():
            try:
                flag = self.val[effect_name.lower()]
                if bool(flag):
                    effects.append(symbol)
            except (gdb.error, KeyError):
                pass

        # Try bitfield representation
        try:
            bits = int(self.val['bits'])
            effect_list = ['IO', 'Mut', 'Alloc', 'Panic', 'Async', 'GPU', 'Prob', 'Div']
            for i, effect in enumerate(effect_list):
                if bits & (1 << i):
                    effects.append(effect)
        except (gdb.error, KeyError):
            pass

        return effects


# =============================================================================
# Linear Type Pretty Printer
# =============================================================================

class LinearTypePrinter:
    """
    Pretty printer for Sounio linear/affine types.

    Linear types must be used exactly once:
    - Linear: Must be consumed exactly once
    - Affine: May be dropped, cannot be copied
    - Normal: Standard copyable type
    """

    def __init__(self, val: gdb.Value):
        self.val = val
        self.typename = str(val.type)

    def to_string(self) -> str:
        linearity = self._get_linearity()
        inner = self._get_inner_value()

        if linearity == 'Linear':
            return f"linear {inner}"
        elif linearity == 'Affine':
            return f"affine {inner}"
        else:
            return str(inner)

    def _get_linearity(self) -> str:
        try:
            for field_name in ['linearity', 'kind', 'mode']:
                try:
                    lin = self.val[field_name]
                    return str(lin)
                except (gdb.error, KeyError):
                    continue

            # Check type name
            if 'Linear' in self.typename or 'linear' in self.typename:
                return 'Linear'
            elif 'Affine' in self.typename or 'affine' in self.typename:
                return 'Affine'

            return 'Normal'
        except:
            return 'Normal'

    def _get_inner_value(self) -> gdb.Value:
        try:
            for field_name in ['value', 'inner', 'data', '_0']:
                try:
                    return self.val[field_name]
                except (gdb.error, KeyError):
                    continue
            return self.val
        except:
            return self.val


# =============================================================================
# Reference Type Pretty Printer
# =============================================================================

class ReferencePrinter:
    """
    Pretty printer for Sounio reference types.

    Sounio uses:
    - &T for shared references
    - &!T for exclusive/mutable references (NOT &mut!)
    """

    def __init__(self, val: gdb.Value):
        self.val = val
        self.typename = str(val.type)

    def to_string(self) -> str:
        is_exclusive = self._is_exclusive()
        target = self._get_target()

        if is_exclusive:
            return f"&!{target}"
        else:
            return f"&{target}"

    def _is_exclusive(self) -> bool:
        try:
            # Check for exclusive marker
            for field_name in ['exclusive', 'is_exclusive', 'mutable']:
                try:
                    return bool(self.val[field_name])
                except (gdb.error, KeyError):
                    continue

            # Check type name
            return 'Exclusive' in self.typename or 'Mut' in self.typename
        except:
            return False

    def _get_target(self) -> str:
        try:
            if self.val.type.code == gdb.TYPE_CODE_PTR:
                target = self.val.dereference()
                return str(target)

            for field_name in ['ptr', 'target', 'inner']:
                try:
                    ptr = self.val[field_name]
                    return str(ptr.dereference())
                except (gdb.error, KeyError):
                    continue

            return "..."
        except:
            return "..."


# =============================================================================
# Printer Lookup and Registration
# =============================================================================

class SounioPrinterLookup:
    """Lookup function for Sounio pretty printers."""

    def __init__(self):
        self.name = "sounio"
        self.enabled = True
        self.printers = []

    def __call__(self, val: gdb.Value):
        typename = str(val.type.unqualified().strip_typedefs())

        # Knowledge types
        if is_sounio_type(typename, 'Knowledge'):
            return KnowledgePrinter(val)

        # Quantity types
        if is_sounio_type(typename, 'Quantity'):
            return QuantityPrinter(val)

        # Effect types
        if is_sounio_type(typename, 'Effect') or is_sounio_type(typename, 'Effects'):
            return EffectPrinter(val)

        # Linear types
        if 'Linear<' in typename or 'Affine<' in typename:
            return LinearTypePrinter(val)

        # Reference types with exclusive marker
        if 'ExclusiveRef' in typename or 'MutRef' in typename:
            return ReferencePrinter(val)

        return None


def register_printers(objfile):
    """
    Register Sounio pretty printers with GDB.

    Args:
        objfile: The object file to register printers for, or None for global.
    """
    lookup = SounioPrinterLookup()

    if objfile is None:
        # Register globally
        gdb.pretty_printers.append(lookup)
    else:
        # Register for specific object file
        objfile.pretty_printers.append(lookup)

    print("Sounio pretty printers registered.")


# =============================================================================
# GDB Commands
# =============================================================================

class SounioInfoCommand(gdb.Command):
    """
    Display Sounio-specific information about a value.

    Usage: sounio-info <expression>

    Shows detailed epistemic, unit, and effect information for Sounio types.
    """

    def __init__(self):
        super(SounioInfoCommand, self).__init__(
            "sounio-info",
            gdb.COMMAND_DATA,
            gdb.COMPLETE_EXPRESSION
        )

    def invoke(self, arg: str, from_tty: bool):
        try:
            val = gdb_evaluate_expression(arg)
            typename = str(val.type)

            print(f"Type: {typename}")

            if is_sounio_type(typename, 'Knowledge'):
                self._print_knowledge_info(val)
            elif is_sounio_type(typename, 'Quantity'):
                self._print_quantity_info(val)
            else:
                print(f"Value: {val}")

        except gdb.error as e:
            print(f"Error: {e}")

    def _print_knowledge_info(self, val: gdb.Value):
        printer = KnowledgePrinter(val)
        print(f"Summary: {printer.to_string()}")
        print("\nDetails:")
        for name, child in printer.children():
            print(f"  {name}: {child}")

    def _print_quantity_info(self, val: gdb.Value):
        printer = QuantityPrinter(val)
        print(f"Value: {printer.to_string()}")
        dim = printer._get_dimension()
        if dim:
            print(f"Dimension: {dim}")
            print(f"SI units: {format_unit_dimension(dim)}")


class SounioConfidenceCommand(gdb.Command):
    """
    Show confidence levels for Knowledge values in scope.

    Usage: sounio-confidence

    Lists all Knowledge values in the current frame with their confidence levels.
    """

    def __init__(self):
        super(SounioConfidenceCommand, self).__init__(
            "sounio-confidence",
            gdb.COMMAND_DATA
        )

    def invoke(self, arg: str, from_tty: bool):
        try:
            frame = gdb.selected_frame()
            block = frame.block()

            knowledge_vars = []

            while block:
                for sym in block:
                    if sym.is_variable:
                        try:
                            val = sym.value(frame)
                            typename = str(val.type)
                            if is_sounio_type(typename, 'Knowledge'):
                                printer = KnowledgePrinter(val)
                                conf = printer._get_confidence()
                                knowledge_vars.append((sym.name, conf, val))
                        except:
                            pass
                block = block.superblock

            if not knowledge_vars:
                print("No Knowledge values found in current scope.")
                return

            print("Knowledge values and confidence levels:")
            print("-" * 50)
            for name, conf, val in sorted(knowledge_vars, key=lambda x: -(x[1] or 0)):
                conf_str = format_confidence(conf) if conf is not None else "unknown"
                print(f"  {name}: {conf_str}")

        except gdb.error as e:
            print(f"Error: {e}")


# =============================================================================
# Auto-registration
# =============================================================================

# Register printers when module is loaded
register_printers(None)

# Register custom commands
SounioInfoCommand()
SounioConfidenceCommand()

print("Sounio GDB support loaded. Commands: sounio-info, sounio-confidence")
