# First minimal probe: invalid graph schema

All eight exports stopped while parsing layer IDs because the constructed IDs lacked numeric suffixes. None reached SDK VerifyGraph; these are harness preparation errors, not evidence that any dtype is unsupported. Retained unchanged. Corrected and controlled matrix lives in ../down-minimal-v2.
