# Nsight 2025.3 Help-Reports Compatibility Design

## Scope

Fix only the `Detect supported reports` stage and its fallback behavior. Keep
SQLite export/cache semantics, report collection, workflow capture, and Torch
Profiler behavior unchanged. Development uses fake `nsys` executables and
fixture SQLite only; no real GPU or Nsight invocation is required.

## Root Cause

`detect_supported_reports()` currently treats every nonzero
`nsys stats --help-reports` exit code as fatal before inspecting the help body.
It also redirects stdout to `.help-reports.txt` while streaming stderr only to
the progress log. Nsight Systems 2025.3.1 can print a valid help body and return
exit code 1, including when that body is written to stderr. Therefore the
current control flow discards valid capability evidence and never reaches
report collection.

## Help Capture and Validation

The help subprocess will merge stdout and stderr into one temporary file while
preserving the complete command, progress status, and child exit code.

Report parsing will recognize concrete report tokens and strip optional grammar
suffixes. For example:

```text
nvtx_sum[:nvtx-name][:base|:mangled]
```

contributes the base report name `nvtx_sum`, while concrete names such as
`cuda_gpu_kern_sum:base` remain concrete variants.

Exit-code handling is content-aware:

- exit code 0 plus a parseable core report list succeeds;
- exit code 1 succeeds with a warning when the merged text contains
  `The following built-in reports are available:`, `cuda_gpu_kern_sum`, and
  `cuda_api_sum`;
- empty/unparseable output or output without the core kernel report fails help
  detection, regardless of exit code.

Warnings are written to stderr/progress.log and propagated into parser metadata
when the facade handles a degraded help result.

## Fallback Probe

If help detection fails, `parse_nsys.py` records a warning and calls the normal
report collector in probe mode. Probe mode attempts every selected report
directly against the already resolved SQLite:

```text
nsys stats --report <report> --format csv <input.sqlite>
```

The generated CSV is the probe result, so reports are not executed twice.
Failure of `cuda_gpu_kern_sum` remains fatal. Unsupported or failed optional
reports remain visible warnings.

The fallback occurs after `resolve_sqlite()`. A valid adjacent SQLite whose
mtime is not older than the `.nsys-rep` is reused exactly as before; neither
help failure nor probe mode invokes `nsys export`.

## Regression Tests

Fake-NSys tests cover:

- exit 1 plus valid help on stdout succeeds with warning;
- exit 1 plus empty merged output fails help detection;
- valid help written to stderr succeeds;
- optional grammar suffixes normalize to the base report name;
- failed help falls back to direct report probes;
- a current `qwen-tp4-full.sqlite` is reused and the recorded command log has no
  `nsys export` invocation;
- fallback core failure is nonzero and optional failure remains a warning.
