/* Compact graph history retains EVERY valid token; storage capacity is separate.
 * This is raw KV transfer only, no quantization or attention approximation. */
static int history_contract(const Tensor *dst,const Tensor *src,int past,int storage,int key) {
    Tensor expected=*src;
    if(past<1||past>storage||src->count!=(unsigned)(KV*storage*HD)||src->p.num_of_dims!=4)return 0;
    int axis=key&&KEY_CACHE_TRANSPOSED?0:1;
    if(src->p.sizes[axis]!=(unsigned)storage)return 0;
    expected.count=KV*past*HD;expected.p.sizes[axis]=past;
    return same_tensor_contract(dst,&expected);
}
static void pack_history(Tensor *dst,const Tensor *src,int past,int storage,int keep,int key) {
    if(keep<0||keep>past||!history_contract(dst,src,past,storage,key)){
        fprintf(stderr,"compact history contract/overflow\n");exit(2);
    }
    size_t width=0;
    switch(dst->p.data_format){
    case VIP_BUFFER_FORMAT_INT8:width=1;break;
    case VIP_BUFFER_FORMAT_INT16:case VIP_BUFFER_FORMAT_FP16:width=2;break;
    case VIP_BUFFER_FORMAT_FP32:width=4;break;
    default:exit(2);
    }
    memset(dst->data,0,dst->count*width);
    if(key&&KEY_CACHE_TRANSPOSED){
        for(int h=0;h<KV;h++)for(int d=0;d<HD;d++)
            memcpy((char*)dst->data+((h*HD+d)*past)*width,
                   (const char*)src->data+((h*HD+d)*storage)*width,keep*width);
    }else for(int h=0;h<KV;h++)
        memcpy((char*)dst->data+h*past*HD*width,
               (const char*)src->data+h*storage*HD*width,keep*HD*width);
}

static int compact_chunk_eligible(int rows,int past,int pos,int reused,int total) {
    return rows>0 && reused>0 && pos>=reused && pos<=past && total-pos>=rows;
}

/* Prefer the smaller graph; never remove any live history. */
static int compact_bank_selection(int rows128,int rows256,int pos,int reused,int total) {
    if(compact_chunk_eligible(rows128,128,pos,reused,total))return 1;
    if(compact_chunk_eligible(rows256,256,pos,reused,total))return 2;
    return 0;
}
