/* Batched causal prefill shares decode KV only after exact IO checks. */
typedef struct { Layer *layers; int tokens,blocks,capacity,previous_position; } BatchPrefill;
static int same_tensor_contract(const Tensor *a,const Tensor *b) {
    if(a->count!=b->count||a->p.num_of_dims!=b->p.num_of_dims||
       a->p.data_format!=b->p.data_format||a->p.quant_format!=b->p.quant_format||
       memcmp(a->p.sizes,b->p.sizes,a->p.num_of_dims*sizeof(a->p.sizes[0])))return 0;
    if(a->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return a->p.quant_data.dfp.fixed_point_pos==b->p.quant_data.dfp.fixed_point_pos;
    return a->p.quant_format==VIP_BUFFER_QUANTIZE_NONE;
}
static BatchPrefill load_prefill(const char *path,int tokens,int blocks,int capacity,Layer *decode) {
    BatchPrefill b={0};if(!tokens)return b;
    b.tokens=tokens;b.blocks=blocks;b.capacity=capacity;b.layers=calloc(blocks,sizeof(Layer));
    if(!b.layers)exit(2);
    unsigned ins[TENSOR_MAX_INPUTS]={(unsigned)(tokens*DIM)},outs[TENSOR_MAX_OUTPUTS]={(unsigned)(tokens*DIM)};
    unsigned repeat=DIM/(KV*HD);
    for(int k=0;k<FUSED_LAYERS;k++){
        ins[IN_K(k)]=ins[IN_V(k)]=KV*capacity*HD;
        ins[IN_MASK(k)]=repeat*tokens*(capacity+tokens);
        ins[IN_COS(k)]=ins[IN_SIN(k)]=tokens*HD;
        outs[OUT_K(k)]=outs[OUT_V(k)]=KV*tokens*HD;
    }
    for(int i=0;i<blocks;i++){
        char filename[1024];snprintf(filename,sizeof(filename),"%s/prefill-layer%d.nb",path,i*FUSED_LAYERS);
        Layer *l=&b.layers[i];load_net(l,filename,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS,ins,outs);
        /* Future history slots and the within-batch causal triangle are static.
         * Each subsequent call only opens newly available history columns. */
        for(int k=0;k<FUSED_LAYERS;k++)for(unsigned r=0;r<repeat;r++)
            for(int t=0;t<tokens;t++)for(int j=0;j<capacity+tokens;j++)
                put(&l->in[IN_MASK(k)],(r*tokens+t)*(capacity+tokens)+j,
                    (j>=capacity&&j<=capacity+t)?0:-10000);
        for(int k=0;k<FUSED_LAYERS;k++)for(int kind=0;kind<2;kind++){
            unsigned port=kind?IN_V(k):IN_K(k);
            if(!same_tensor_contract(&l->in[port],&decode[i].in[port])){
                fprintf(stderr,"prefill/decode KV mismatch block=%d port=%u\n",i,port);exit(2);
            }
            check(vip_set_input(l->net,port,decode[i].in[port].b),"share prefill KV");
        }
        if(i){
            if(!same_tensor_contract(&l->in[0],&b.layers[i-1].out[0])){
                fprintf(stderr,"prefill hidden mismatch\n");exit(2);
            }
            check(vip_set_input(l->net,0,b.layers[i-1].out[0].b),"link prefill hidden");
        }
    }
    return b;
}
static void run_prefill(BatchPrefill *b,Layer *decode,const float *embedding,
                        const int32_t *prompt,int pos,float *last_hidden,double *api_seconds) {
    const int n=b->tokens,c=b->capacity,repeat=DIM/(KV*HD);
    if(pos<b->previous_position||pos+n>c){fprintf(stderr,"prefill position/capacity overflow\n");exit(2);}
    for(int t=0;t<n;t++)for(int j=0;j<DIM;j++)
        put(&b->layers[0].in[0],t*DIM+j,embedding[(size_t)prompt[pos+t]*DIM+j]);
    float cosines[n*HD],sines[n*HD];
    for(int t=0;t<n;t++)for(int j=0;j<HD;j++){
        float angle=(pos+t)/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
        cosines[t*HD+j]=cosf(angle);sines[t*HD+j]=sinf(angle);
    }
    for(int i=0;i<b->blocks;i++){
        Layer *l=&b->layers[i];
        for(int k=0;k<FUSED_LAYERS;k++){
            for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=b->previous_position;j<pos;j++)
                put(&l->in[IN_MASK(k)],((r*n+t)*(c+n)+j),0);
            for(int j=0;j<n*HD;j++){
                put(&l->in[IN_COS(k)],j,cosines[j]);put(&l->in[IN_SIN(k)],j,sines[j]);
            }
            check(vip_flush_buffer(decode[i].in[IN_K(k)].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush shared K");
            check(vip_flush_buffer(decode[i].in[IN_V(k)].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush shared V");
        }
        if(!i)check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush batch embedding");
        for(unsigned j=1+2*FUSED_LAYERS;j<TENSOR_MAX_INPUTS;j++)
            check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush batch position");
        double stamp=now();check(vip_run_network(l->net),"batch prefill");*api_seconds+=now()-stamp;
        for(unsigned j=1;j<TENSOR_MAX_OUTPUTS;j++)
            check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read batch KV");
        for(int k=0;k<FUSED_LAYERS;k++){
            for(int h=0;h<KV;h++){
                copy_values(&decode[i].in[IN_K(k)],(h*c+pos)*HD,&l->out[OUT_K(k)],h*n*HD,n*HD);
                copy_values(&decode[i].in[IN_V(k)],(h*c+pos)*HD,&l->out[OUT_V(k)],h*n*HD,n*HD);
            }
            for(int t=0;t<pos+n;t++)put(&decode[i].in[IN_MASK(k)],t,0);
        }
    }
    Tensor *out=&b->layers[b->blocks-1].out[0];
    b->previous_position=pos;
    check(vip_flush_buffer(out->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read batch hidden");
    for(int j=0;j<DIM;j++){
        last_hidden[j]=get(out,(n-1)*DIM+j);
        if(!isfinite(last_hidden[j])){fprintf(stderr,"nonfinite batch hidden\n");exit(3);}
    }
}
static void free_prefill(BatchPrefill *b) {
    /* Aliases belong to decode/preceding output. Free only owned port buffers. */
    for(int i=0;i<b->blocks;i++)free_net(&b->layers[i],TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);
    free(b->layers);
}
