# Upstream Runtime Snapshot

This inference runtime was ported from the dirty working tree at
`D:\styledecouple_dinov3`. It does not claim to match the clean Git commit by
itself. The commit, exact binary diff digest, source hashes, and adapted
destination hashes below jointly identify the reviewed snapshot.
`vendored_sha256` uses LF-canonicalized bytes so Git's Windows checkout policy
does not change the recorded source identity.

The vendored files keep the corrected model behavior. Local adaptations are
limited to package-relative imports, an inlined SHA256 helper, formatting, and
local naming/docstring cleanup. The resolver also adapts to both known
`torch.hub._get_cache_or_reload` signatures without executing `hubconf.py`.
The upstream license is
`D:\styledecouple_dinov3\LICENSE` (SHA256
`6f1e622c82a380075843bb084a7ec3b1f1d12a4a02526d75e78b0924a860aa75`).

```json
{
  "source_repository": "D:\\styledecouple_dinov3",
  "source_head": "77a9428e4e67d1ebe80f861076c878ce8c1c4f65",
  "source_diff_sha256": "1fb8108ac7b23c3dacac26d971d492c05f9a785b96c9476fa5029c27f0769c84",
  "source_diff_bytes": 89215,
  "source_sha256": {
    "src/data/transforms.py": "1c4121648a869d65bdfce982e571422290d733d8593a19b437ca5b2fa7b82180",
    "src/models/feature_extractors.py": "01fc1afd543149dcc76b68339e0373a0a5bfbda05243c8998cb65e477a7fd91e",
    "src/models/projector.py": "5947933330acc332755002d21129bcbce086aea3c9e6b78e41b190fbddeb6bd0",
    "src/models/style_decoupler.py": "785623ff601964298fb040e02b9a0a26482f8720f05ccc6e055b73d3efe130bd"
  },
  "vendored_sha256": {
    "feature_extractors.py": "661cc4d66dce00822ac8df299b4dd1e91de81e1e985b1dba1cb0efe534ae4a4b",
    "projector.py": "f923c22df45fd780f63b406055d68d9b9cf9ea65c3a91c1861297b0665d1282c",
    "style_decoupler.py": "9e882ff6bf8b5fe345673669caf153250b88300e922c28887b8b3d9dd4634a05",
    "transforms.py": "b19bb3bc3b07c9d111e4e08947c2485b48308986d7ec80ca7f619a26f5155eae"
  }
}
```

## Refresh Procedure

1. Inspect `git status --short` and `git rev-parse HEAD` in the source
   repository. Do not describe a dirty tree as a clean revision.
2. Capture `git diff --binary HEAD` with `subprocess.run(...,
   stdout=subprocess.PIPE).stdout`; record its byte length and SHA256 without
   text or newline conversion.
3. Hash the four source files as bytes, port the inference-only changes, and run
   the strict-loader and transform regression tests.
4. Run Ruff formatting once, replace CRLF with LF in the four final vendored
   files before hashing their bytes, update this JSON block, and rerun the
   vendored-hash test.
