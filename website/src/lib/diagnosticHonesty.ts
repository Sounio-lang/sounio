/**
 * Honesty kernel for compiler diagnostics on the site.
 *
 * #1794 measured that 33 user-facing diagnostics had content length > 127
 * and were silently cut to 127 characters on the pre-#1784 Madaros print
 * path. Largest: 333 characters at check.sio help for code 219, cut by 206.
 *
 * A DiagnosticCodeBlock must render the whole literal. This module is the
 * counterpart of that census: it measures, it can mark the old cap, and it
 * refuses to return a shortened string as if it were the diagnostic.
 */

/** Pre-#1784 Madaros print cap (content length). Documented in the #1794 census. */
export const PRE_FIX_PRINT_CAP = 127;

/**
 * E219 primary message — `self-hosted/check/check.sio` `code == 219` table.
 * Length 70; would have printed in full even before the fix.
 */
export const E219_MESSAGE =
  'call to an `extern "C"` function the native backend does not implement';

/**
 * E219 help literal, including the rustc-style prefix and trailing newline,
 * exactly as printed by `print_error_help(219)`. Census length: 333.
 */
export const E219_HELP =
  '   |\n   = help: only these body-less names are implemented: print, print_int, print_char, print_f64, get_arg, get_arg_count, str_len, str_char_at, str_eq, str_slice, starts_with, str_concat, str_from_bytes, read_file, write_file, file_size, sqrt, exp, log, sin, cos, assert, heap_alloc, heap_free, f64_to_bits, bits_to_f64, syscall6\n';

/**
 * E219 note literal from `print_error_note(219)`. Length 124 — under the cap.
 */
export const E219_NOTE =
  '   = note: there is no dynamic linker in this backend; an unimplemented extern compiles to an empty stub whose calls read 0\n';

export type TruncationSplit = {
  length: number;
  wouldTruncate: boolean;
  kept: string;
  dropped: string;
  droppedCount: number;
};

export function measureLiteral(content: string): number {
  return content.length;
}

export function splitAtPrintCap(
  content: string,
  cap: number = PRE_FIX_PRINT_CAP,
): TruncationSplit {
  const length = content.length;
  if (length <= cap) {
    return {
      length,
      wouldTruncate: false,
      kept: content,
      dropped: '',
      droppedCount: 0,
    };
  }
  return {
    length,
    wouldTruncate: true,
    kept: content.slice(0, cap),
    dropped: content.slice(cap),
    droppedCount: length - cap,
  };
}

/** Never return a silently shortened diagnostic. The dropped tail stays attached. */
export function fullDiagnosticText(content: string): string {
  return content;
}
