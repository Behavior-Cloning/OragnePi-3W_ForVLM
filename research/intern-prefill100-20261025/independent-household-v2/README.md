# New household scenes, frozen before evaluating the range16 full model

12 new scenarios × static64/128 =24 presentations. Dynamic body exactly64/128/192/256; template8 counted separately. Tokenizer and original fixed system IDs are unchanged. Freeze SHA in semantic-inputs.json provenance.

Texts are manually specified cores plus complete neutral background sentences selected deterministically to reach exact token counts. This is artificial composition, not recorded spontaneous speech. No truncation, no model-output-based text/gold selection. Not a claim of training-data-disjoint data. No physical motors; generic hardware capability assumptions must be checked before real execution.

Do not recalibrate on this set before first frozen-candidate evaluation. Judge high-level instruction compatibility and complete constraint fulfillment separately; token equality or a substring is not an accuracy metric.
