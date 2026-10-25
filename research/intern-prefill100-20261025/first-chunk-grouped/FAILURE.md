# Rejected dynamic group rebuild

Run first-chunk-grouped-r1 recovered with rc1. Original grouped and hybrid ungrouped phases completed; hybrid grouped failed during its first system preparation. SDK: `batch prefill status=-9`, network belongs to one group even after vip_destroy_group. No successful hybrid grouped request, no speed result, no close-zero claim for failed worker. Do not use this binary or launch fulli16-firstfix.

Replacement experiment suffixgroup-firstfix keeps group membership immutable for the lifetime; onlyblocks2..11 join suffixgroup, block0 and original/candidateblock1 are standalone. Three submissions/batch; actual latency requires new pairedmeasurement. No vendor private-memory edits.
