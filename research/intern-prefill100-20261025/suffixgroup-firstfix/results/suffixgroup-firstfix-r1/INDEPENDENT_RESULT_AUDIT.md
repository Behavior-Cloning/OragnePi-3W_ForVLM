# Independent suffix-group first-chunk diagnostic audit

**Bounded diagnostic gate passed; broad qualification remains pending.** Controller rc0/meta complete. Independently checked48 actual F32 files, all finite, actualSHA and argmax/token matched. Input/new/reused contracts and source/build hashes pass.

- 24 results:12 fixed four-step feedback and12 natural max32/EOS. Three close events all driver0. Original:first0/suffix0; hybrid-ungrouped:first8/suffix0; hybrid-suffix:first8/suffix8.
- All8 suffix-group output logits are bitwise identical to hybrid-ungrouped. All24 outputs reproduce the corresponding prior ungrouped diagnostic logits.
- All16 fixed predictions and four natural replies in the suffix variant match original. Residual logits versus original remain about2.9% relativeL2; matching tokens does not mean bitwise original.

|Presentation|Fixed logits relL2 vs original|Natural reply|
|---|---:|---|
|s64-d64-0|0.02891674|点头回应|
|s128-d256-0|0.02920787|点头回应|
|s64-d128-base-back|0.02893957|后退两厘米|
|s128-d128-base-back|0.02992508|后退两厘米|

The two backward presentations restore 后退两厘米 but omit 缓慢, so the established rubric remains compatible_partial. These are four development presentations, not a general holdout: two cost inputs and two static variants of one backward scenario.

All timings are diagnostic/ineligible. No speech/RAG concurrency or motors. Permanent suffix grouping requires three submissions per batch, so new performance measurements are required. This audit supports proceeding to the separately reviewed full matrix; it does not promote the model.
