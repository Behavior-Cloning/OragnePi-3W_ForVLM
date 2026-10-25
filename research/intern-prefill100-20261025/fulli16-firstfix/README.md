# First-chunk-preserved I16 full-model qualification

Prepared, not launched. Same static64/128(body59/123+5), body64/128/192/256+8,new72/136/200/264; output32 fixed feedback. Original grouped / I16 grouped with original L2 only pos0 / original repeat. Only cold system preparation destroys/rebuilds group; original first chunk preserved, all dynamic calls grouped. Original system first16 graph has explicit hash and strict actual port check. No dynamic caching.

72 performance,144 natural (old24+new24 per variant),24 diagnostic requests of4full-vocab steps. Newset source frozen before actual model outputs, artificial exact-length background composition; no general accuracy claim. Same synthetic streaming speech and residentRAG as priorfulltest; not vision/microphone/motors. ExistingRAG safety defects remain; mock only.

Do not launch until first-chunk-grouped diagnostic returns0 and recoveredactuallogits confirm grouped/ungrouped hybrid parity. All board operations throughcontrollerdual locks. PriorI16failedqualityevidence preserved. Capacity1280 deliberately retained for causal comparison; C512/M32 follows separately. Goal100ms remainsunachieved.
