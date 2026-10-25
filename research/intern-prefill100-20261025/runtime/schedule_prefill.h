/* Pure scheduling contract; logical coverage never includes padded rows. */
#ifndef OPI_SCHEDULE_PREFILL_H
#define OPI_SCHEDULE_PREFILL_H
enum {OPI_SCALAR=0,OPI_BASE16=1,OPI_BULK=2,OPI_TAIL8=3};
typedef struct {int kind,rows,valid;} PrefillChoice;
static PrefillChoice choose_prefill(int pos,int prompt,int prefix,int capacity,int bulk,int tail8,int tail_min,int bulk_pad_final) {
    PrefillChoice x={OPI_SCALAR,1,1};
    if(pos<0||pos>=prompt||prefix<1||prefix>prompt||prompt>capacity)return x;
    int remaining=prompt-pos;
    /* System KV always uses exactly the original M16 partition. */
    if(pos<prefix){if(prefix-pos>=16){x.kind=OPI_BASE16;x.rows=x.valid=16;}return x;}
    if(bulk&&remaining>=bulk){x.kind=OPI_BULK;x.rows=x.valid=bulk;return x;}
    if(bulk_pad_final&&bulk&&remaining>16&&remaining<bulk&&pos+bulk<=capacity){x.kind=OPI_BULK;x.rows=bulk;x.valid=remaining;return x;}
    if(remaining>=16){x.kind=OPI_BASE16;x.rows=x.valid=16;return x;}
    if(tail8&&remaining==8){x.kind=OPI_TAIL8;x.rows=x.valid=8;return x;}
    if(tail_min&&remaining>=tail_min&&pos+16<=capacity){x.kind=OPI_BASE16;x.rows=16;x.valid=remaining;}
    return x;
}
#endif
