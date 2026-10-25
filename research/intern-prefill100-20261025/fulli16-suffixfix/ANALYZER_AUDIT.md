# Offline analyzer coverage audit — final patched revision

**All previously listed source-coverage fixes are implemented.** This is read-only source review, not a completed runtime-result audit. The analyzer and running benchmark were not modified by this auditor; no board operation was performed.

Verified coverage:

- Exact unique variant×case totals:72 fixed32 performance,144 natural (old24+new24 pervariant),24 diagnostic requests with all8 expected cells.
- Source/config hashes, recoveredfixtures equality to frozeninputs, tokenizerSHA; independent check confirms systemmetadata attachment does not change the frozen inputdocument.
- All29actualgraph records matchconfig and graph_verified events; additional originalL2 event and loadedbinary/settings/actualready24layers,C1280,no true8,singlepass are bound.
- Actual diagnosticF32 size/SHA/finite andargmax==emittedtokens; diagnosticexcludedfromtiming.
- Perrequest prefill+scalar beforestatistics,32fixedfeedback and31decodeintervals; matcheddecode ratios now recompute fromintervalduration.
- Actual81first/suffixmarkers and3close driver0; old/new qualitysets separate.
- Pervariant andperformancephase resourcebuckets with same_bucket intervals;24matchedTTFT/decode andA-A drift records.
- Measuredgrid100ms is separate from whole_objective_verified=False;boundary20,continuousrange andlongstability remainexplicitlyunverified.
- Analyzer/sharedhelper/contract SHA are included in finalanalysis provenance.

Interpretation limits remain: threeauthoredcases/cell andonepass do not measure dependabletail latency orlongstability;fixedfeedback cost is notnaturalquality. Globalresourcesinclude systemrebuild/load/hash/diagnostic. Perphaseaudio maxima remaincumulative,notlocalpeaks. Semanticcompatibility andRAG conflicts requireindependentmanualreview afterrecovery.

Final reviewed analyzer SHA256: `3de283c3c0ef3e7e6a707accdc6330058f37e5cff85fbe48778cac5874cd0ac7`
Analyzer size: 12235 bytes
Metric helper SHA256: `5855a91b06697d48ade2afc161245a3218fa316491d678d51bf76d70e6a4560e`

No current-run runtime result has been audited yet.
