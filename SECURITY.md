# Security

To report a vulnerability, email `rahadbhuiya2021@gmail.com`. Do not open public issues for security bugs.

Reports should include:
- Description and affected components.
- Reproduction steps or proof of concept.
- Target hardware and driver version (e.g., RTX 3060, Driver 617.14).

## Scope

Tessera runs user-space code that loads PTX or SPIR-V through vendor drivers (`nvcuda.dll` / `libcuda.so`).

- Host allocations and memory transfers are bounds-checked before driver dispatch.
- Matrix dimensions are guarded via boundary predication to prevent out-of-bounds global memory access.
- Kernel names and identifiers are validated against alphanumeric and underscore patterns before emission.
