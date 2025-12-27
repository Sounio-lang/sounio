#!/usr/bin/env python3
"""
Sounio LLDB Pretty Printers and Type Formatters

This module provides LLDB type formatters for Sounio's core types:
- Knowledge<T>: Epistemic values with confidence, provenance, and ontology
- Quantity<T, U>: Numeric values with units of measure
- Effect types: IO, Mut, Alloc, Async, etc.

Usage:
    (lldb) command script import sounio_lldb.py
    (lldb) type summary add -F sounio_lldb.knowledge_summary -x "^Knowledge<.+>$"

Or add to ~/.lldbinit:
    command script import /path/to/sounio/tools/debug/sounio_lldb.py
"""

import lldb
import re
from typing import Optional


# =============================================================================
# Helper Functions
# =============================================================================

def get_template_arg(typename: str) -> Optional[str]:
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
    units = ['kg', 'm', 's', 'A', 'K', 'mol', 'cd']
    parts = []

    for exp, unit in zip(dim, units):
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


def get_child_by_name(valobj: lldb.SBValue, name: str) -> Optional[lldb.SBValue]:
    """Safely get a child value by name."""
    child = valobj.GetChildMemberWithName(name)
    if child.IsValid():
        return child
    return None


# =============================================================================
# Knowledge<T> Summary Provider
# =============================================================================

def knowledge_summary(valobj: lldb.SBValue, internal_dict: dict) -> str:
    """
    Summary function for Knowledge<T> types.

    Displays: Knowledge<type> { value: X, confidence: Y%, source: Z }
    """
    try:
        typename = valobj.GetTypeName()
        content_type = get_template_arg(typename) or "?"

        # Get value
        value_child = get_child_by_name(valobj, "value")
        value_str = str(value_child.GetValue()) if value_child else "?"

        # Get confidence
        confidence = None
        for field_name in ['confidence', 'epsilon', 'conf']:
            conf_child = get_child_by_name(valobj, field_name)
            if conf_child:
                try:
                    # Handle nested value field
                    inner = get_child_by_name(conf_child, "value")
                    if inner:
                        confidence = float(inner.GetValue())
                    else:
                        confidence = float(conf_child.GetValue())
                    break
                except (ValueError, TypeError):
                    continue

        # Get source/provenance
        source = None
        for field_name in ['source', 'provenance', 'origin']:
            source_child = get_child_by_name(valobj, field_name)
            if source_child:
                source = source_child.GetSummary() or str(source_child.GetValue())
                if source:
                    source = source.strip('"')
                    break

        # Get ontology
        ontology = None
        for field_name in ['ontology', 'domain', 'binding']:
            ont_child = get_child_by_name(valobj, field_name)
            if ont_child:
                ontology = ont_child.GetSummary() or str(ont_child.GetValue())
                if ontology:
                    ontology = ontology.strip('"')
                    break

        # Build output
        parts = [f"Knowledge<{content_type}> {{"]
        parts.append(f"value: {value_str}")

        if confidence is not None:
            parts.append(f", confidence: {format_confidence(confidence)}")

        if source:
            parts.append(f", source: {source}")

        if ontology:
            parts.append(f", ontology: {ontology}")

        parts.append("}")

        return " ".join(parts)

    except Exception as e:
        return f"Knowledge<...> {{ <error: {e}> }}"


# =============================================================================
# Quantity<T, U> Summary Provider
# =============================================================================

def quantity_summary(valobj: lldb.SBValue, internal_dict: dict) -> str:
    """
    Summary function for Quantity<T, U> types.

    Displays: 500.0 mg
    """
    try:
        # Get numeric value
        value_child = get_child_by_name(valobj, "value")
        if not value_child:
            return "Quantity { ? }"

        value = value_child.GetValue()

        # Try to get unit from type name
        typename = valobj.GetTypeName()
        match = re.search(r'Quantity<[^,]+,\s*(\w+)>', typename)

        if match:
            unit_type = match.group(1)
            unit = unit_type_to_symbol(unit_type)
            return f"{value} {unit}"

        # Try to get unit from field
        for field_name in ['_unit', 'unit', 'symbol']:
            unit_child = get_child_by_name(valobj, field_name)
            if unit_child:
                unit = unit_child.GetSummary() or str(unit_child.GetValue())
                if unit:
                    return f"{value} {unit.strip('\"')}"

        return str(value)

    except Exception as e:
        return f"Quantity {{ <error: {e}> }}"


def unit_type_to_symbol(unit_type: str) -> str:
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


# =============================================================================
# Effect Type Summary Provider
# =============================================================================

def effect_summary(valobj: lldb.SBValue, internal_dict: dict) -> str:
    """
    Summary function for Effect types.

    Displays: IO + Mut + Alloc
    """
    try:
        effect_names = ['IO', 'Mut', 'Alloc', 'Panic', 'Async', 'GPU', 'Prob', 'Div']
        effects = []

        # Try to read effect flags
        for effect in effect_names:
            flag = get_child_by_name(valobj, effect.lower())
            if flag and flag.GetValueAsUnsigned() != 0:
                effects.append(effect)

        # Try bitfield representation
        if not effects:
            bits_child = get_child_by_name(valobj, "bits")
            if bits_child:
                bits = bits_child.GetValueAsUnsigned()
                for i, effect in enumerate(effect_names):
                    if bits & (1 << i):
                        effects.append(effect)

        if not effects:
            return "Pure"

        return " + ".join(effects)

    except Exception as e:
        return f"Effect {{ <error: {e}> }}"


# =============================================================================
# Linear Type Summary Provider
# =============================================================================

def linear_summary(valobj: lldb.SBValue, internal_dict: dict) -> str:
    """
    Summary function for Linear/Affine types.

    Displays: linear <value> or affine <value>
    """
    try:
        typename = valobj.GetTypeName()

        # Determine linearity
        if 'Linear<' in typename or 'linear' in typename.lower():
            prefix = "linear"
        elif 'Affine<' in typename or 'affine' in typename.lower():
            prefix = "affine"
        else:
            prefix = ""

        # Get inner value
        for field_name in ['value', 'inner', 'data', '_0']:
            inner = get_child_by_name(valobj, field_name)
            if inner:
                inner_str = inner.GetSummary() or str(inner.GetValue())
                if prefix:
                    return f"{prefix} {inner_str}"
                return inner_str

        return valobj.GetSummary() or str(valobj.GetValue())

    except Exception as e:
        return f"<error: {e}>"


# =============================================================================
# Reference Type Summary Provider
# =============================================================================

def reference_summary(valobj: lldb.SBValue, internal_dict: dict) -> str:
    """
    Summary function for Sounio reference types.

    Displays: &value or &!value (exclusive)
    """
    try:
        typename = valobj.GetTypeName()

        # Check if exclusive/mutable
        is_exclusive = 'Exclusive' in typename or 'Mut' in typename

        for field_name in ['exclusive', 'is_exclusive', 'mutable']:
            flag = get_child_by_name(valobj, field_name)
            if flag and flag.GetValueAsUnsigned() != 0:
                is_exclusive = True
                break

        # Get target value
        for field_name in ['ptr', 'target', 'inner']:
            target = get_child_by_name(valobj, field_name)
            if target:
                deref = target.Dereference()
                target_str = deref.GetSummary() or str(deref.GetValue())
                if is_exclusive:
                    return f"&!{target_str}"
                return f"&{target_str}"

        return "&..."

    except Exception as e:
        return f"<error: {e}>"


# =============================================================================
# LLDB Commands
# =============================================================================

def sounio_info(debugger: lldb.SBDebugger, command: str, result: lldb.SBCommandReturnObject, internal_dict: dict):
    """
    Display Sounio-specific information about a value.

    Usage: sounio-info <expression>
    """
    target = debugger.GetSelectedTarget()
    frame = target.GetProcess().GetSelectedThread().GetSelectedFrame()

    valobj = frame.EvaluateExpression(command)
    if not valobj.IsValid():
        result.AppendMessage(f"Error: could not evaluate '{command}'")
        return

    typename = valobj.GetTypeName()
    result.AppendMessage(f"Type: {typename}")

    if 'Knowledge' in typename:
        result.AppendMessage(f"Summary: {knowledge_summary(valobj, {})}")
        result.AppendMessage("\nFields:")
        for i in range(valobj.GetNumChildren()):
            child = valobj.GetChildAtIndex(i)
            result.AppendMessage(f"  {child.GetName()}: {child.GetValue()}")
    elif 'Quantity' in typename:
        result.AppendMessage(f"Value: {quantity_summary(valobj, {})}")
    else:
        result.AppendMessage(f"Value: {valobj.GetValue()}")


def sounio_confidence(debugger: lldb.SBDebugger, command: str, result: lldb.SBCommandReturnObject, internal_dict: dict):
    """
    Show confidence levels for Knowledge values in scope.

    Usage: sounio-confidence
    """
    target = debugger.GetSelectedTarget()
    frame = target.GetProcess().GetSelectedThread().GetSelectedFrame()

    knowledge_vars = []

    for var in frame.GetVariables(True, True, True, True):
        typename = var.GetTypeName()
        if 'Knowledge' in typename:
            conf = None
            for field_name in ['confidence', 'epsilon', 'conf']:
                conf_child = get_child_by_name(var, field_name)
                if conf_child:
                    try:
                        inner = get_child_by_name(conf_child, "value")
                        if inner:
                            conf = float(inner.GetValue())
                        else:
                            conf = float(conf_child.GetValue())
                        break
                    except:
                        continue
            knowledge_vars.append((var.GetName(), conf))

    if not knowledge_vars:
        result.AppendMessage("No Knowledge values found in current scope.")
        return

    result.AppendMessage("Knowledge values and confidence levels:")
    result.AppendMessage("-" * 50)

    for name, conf in sorted(knowledge_vars, key=lambda x: -(x[1] or 0)):
        conf_str = format_confidence(conf) if conf is not None else "unknown"
        result.AppendMessage(f"  {name}: {conf_str}")


# =============================================================================
# Registration
# =============================================================================

def __lldb_init_module(debugger: lldb.SBDebugger, internal_dict: dict):
    """
    Initialize the Sounio LLDB module.

    This is called automatically when the module is loaded.
    """
    # Register type summaries
    debugger.HandleCommand('type summary add -F sounio_lldb.knowledge_summary -x "^Knowledge<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.quantity_summary -x "^Quantity<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.effect_summary -x "^Effect(s)?<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.linear_summary -x "^Linear<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.linear_summary -x "^Affine<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.reference_summary -x "^ExclusiveRef<.+>$"')
    debugger.HandleCommand('type summary add -F sounio_lldb.reference_summary -x "^MutRef<.+>$"')

    # Register commands
    debugger.HandleCommand('command script add -f sounio_lldb.sounio_info sounio-info')
    debugger.HandleCommand('command script add -f sounio_lldb.sounio_confidence sounio-confidence')

    print("Sounio LLDB support loaded. Commands: sounio-info, sounio-confidence")
