# Dense arithmetic budget for the 100ms objective

Actual 24-layer exported dense matrices, 168 projections, 357826560 MAC per new token. Counts only dense mathematical operations, not measured engine work or a wall-time bound. No board access.

System total64/128=body59/123+template5, only static KV reused. Fixed length does not change dense work per newly computed token; attention work can change.

|Dynamic body|New incl template|Dense GMAC|Required dense-only effective TOPS within100ms|
|---:|---:|---:|---:|
|64|72|25.764|0.515|
|128|136|48.664|0.973|
|192|200|71.565|1.431|
|256|264|94.466|1.889|

Multiply and add count separately. Attention, normalization, rotary, activations, vocabulary head, transfers and setup need additional time/work. Do not compare this directly with a vendor TOPS number without matching datatype and sustained utilization. This does not establish either feasibility or impossibility of the target.
