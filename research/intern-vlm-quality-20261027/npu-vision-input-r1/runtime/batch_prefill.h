#include "visual_input.h"
/* Batched causal prefill shares decode KV only after exact IO checks. */
typedef struct { Layer *layers,*scalar; vip_group group; int grouped; int tokens,blocks,capacity,past_capacity,previous_position,host_hidden,scalar_block; } BatchPrefill;
static void capture_prefill_io(BatchPrefill *b,Layer *decode,int pos,int block,int output) {
    const char *prefix=getenv("OPI_CAPTURE_PREFIX");
    if(!prefix||!getenv("OPI_CAPTURE_POSITION")||!getenv("OPI_CAPTURE_BLOCK")||
       atoi(getenv("OPI_CAPTURE_POSITION"))!=pos||atoi(getenv("OPI_CAPTURE_BLOCK"))!=block)return;
    char path[1024];snprintf(path,sizeof(path),"%s-%s.f32",prefix,output?"output":"input");
    FILE *file=fopen(path,"wb");if(!file)exit(2);
    for(unsigned port=0;port<(output?TENSOR_MAX_OUTPUTS:TENSOR_MAX_INPUTS);port++){
        Tensor *t=output?&b->layers[block].out[port]:&b->layers[block].in[port];
        if(!output){
            if(!port&&block&&!b->host_hidden)t=&b->layers[block-1].out[0];
            else if(b->past_capacity==b->capacity&&port>=1&&port<1+2*FUSED_LAYERS)t=&decode[block].in[port];
        }
        if(output||(!port&&block&&!b->host_hidden))
            check(vip_flush_buffer(t->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"capture device output");
        for(unsigned j=0;j<t->count;j++){
            float value=get(t,j);if(fwrite(&value,sizeof(value),1,file)!=1)exit(2);
        }
    }
    fclose(file);
}
static int same_tensor_contract(const Tensor *a,const Tensor *b) {
    if(a->count!=b->count||a->p.num_of_dims!=b->p.num_of_dims||
       a->p.data_format!=b->p.data_format||a->p.quant_format!=b->p.quant_format||
       memcmp(a->p.sizes,b->p.sizes,a->p.num_of_dims*sizeof(a->p.sizes[0])))return 0;
    if(a->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return a->p.quant_data.dfp.fixed_point_pos==b->p.quant_data.dfp.fixed_point_pos;
    return a->p.quant_format==VIP_BUFFER_QUANTIZE_NONE;
}
#include "compact_history.h"
static BatchPrefill load_prefill_history(const char *path,int tokens,int blocks,int capacity,int storage,Layer *decode) {
    BatchPrefill b={0};if(!tokens)return b;
    b.tokens=tokens;b.blocks=blocks;b.capacity=storage;b.past_capacity=capacity;b.layers=calloc(blocks,sizeof(Layer));
    b.host_hidden=getenv("OPI_PREFILL_HOST_HIDDEN")!=NULL;
    b.scalar_block=getenv("OPI_PREFILL_SCALAR_BLOCK")?atoi(getenv("OPI_PREFILL_SCALAR_BLOCK")):-1;
    if(b.scalar_block<-1||b.scalar_block>=blocks)exit(2);
    if(capacity!=storage&&(b.scalar_block>=0||getenv("OPI_FIRST_PREFILL_L2")||b.host_hidden))exit(2);
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
            if(capacity!=storage){
                if(!history_contract(&l->in[port],&decode[i].in[port],capacity,storage,!kind))exit(2);
                continue;
            }
            if(!same_tensor_contract(&l->in[port],&decode[i].in[port])){
                fprintf(stderr,"prefill/decode KV mismatch block=%d port=%u\n",i,port);exit(2);
            }
            check(vip_set_input(l->net,port,decode[i].in[port].b),"share prefill KV");
        }
        if(i&&!b.host_hidden){
            if(!same_tensor_contract(&l->in[0],&b.layers[i-1].out[0])){
                fprintf(stderr,"prefill hidden mismatch\n");exit(2);
            }
            check(vip_set_input(l->net,0,b.layers[i-1].out[0].b),"link prefill hidden");
        }
    }
    if(b.scalar_block>=0){
        b.scalar=calloc(1,sizeof(Layer));if(!b.scalar)exit(2);
        unsigned scalar_in[TENSOR_MAX_INPUTS]={DIM},scalar_out[TENSOR_MAX_OUTPUTS]={DIM};
        for(int k=0;k<FUSED_LAYERS;k++){
            scalar_in[IN_K(k)]=scalar_in[IN_V(k)]=KV*capacity*HD;
            scalar_in[IN_MASK(k)]=capacity+1;scalar_in[IN_COS(k)]=scalar_in[IN_SIN(k)]=HD;
            scalar_out[OUT_K(k)]=scalar_out[OUT_V(k)]=KV*HD;
        }
        char filename[1024];snprintf(filename,sizeof(filename),"%s/layer%d.nb",path,b.scalar_block*FUSED_LAYERS);
        /* A network owned by a VIP group cannot also execute individually. */
        load_net(b.scalar,filename,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS,scalar_in,scalar_out);
        for(int k=0;k<FUSED_LAYERS;k++)for(int kind=0;kind<2;kind++){
            unsigned port=kind?IN_V(k):IN_K(k);
            if(!same_tensor_contract(&b.scalar->in[port],&decode[b.scalar_block].in[port]))exit(2);
            check(vip_set_input(b.scalar->net,port,decode[b.scalar_block].in[port].b),"share scalar prefill KV");
        }
    }
    const char *initial=getenv("OPI_FIRST_PREFILL_L2");
    if(initial&&tokens==8){
        initial=getenv("OPI_FIRST_PREFILL8_L2");
        if(!initial||!*initial){fprintf(stderr,"missing original M8 first-chunk graph\n");exit(2);}
    }
    if(initial){
        if(!*initial||blocks<3||b.scalar||b.host_hidden){
            fprintf(stderr,"invalid first-chunk diagnostic configuration\n");exit(2);
        }
        /* Reuse the scalar pointer only as ownership storage; scalar_block
         * stays -1. The substitute has the exact same M-row tensor contract. */
        b.scalar=calloc(1,sizeof(Layer));if(!b.scalar)exit(2);
        load_net(b.scalar,initial,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS,ins,outs);
        for(unsigned j=0;j<TENSOR_MAX_INPUTS;j++){
            Tensor *target=&b.layers[1].in[j];
            if(!j)target=&b.layers[0].out[0];
            else if(j<1+2*FUSED_LAYERS)target=&decode[1].in[j];
            if(!same_tensor_contract(&b.scalar->in[j],target))exit(2);
            check(vip_set_input(b.scalar->net,j,target->b),"bind initial input");
        }
        for(unsigned j=0;j<TENSOR_MAX_OUTPUTS;j++){
            if(!same_tensor_contract(&b.scalar->out[j],&b.layers[1].out[j]))exit(2);
            check(vip_set_output(b.scalar->net,j,b.layers[1].out[j].b),"bind initial output");
        }
    }
    b.grouped=getenv("OPI_GROUP_PREFILL")!=NULL;
    if(b.grouped){
        if(b.host_hidden||b.scalar_block>=0)exit(2);
        unsigned timeout=30000;
        int begin=getenv("OPI_FIRST_PREFILL_L2")?2:0;
        check(vip_create_group(blocks-begin,&b.group),"create prefill group");
        check(vip_set_group(b.group,VIP_GROUP_PROP_SET_TIME_OUT,&timeout),"prefill group timeout");
        for(int i=begin;i<blocks;i++)check(vip_add_network(b.group,b.layers[i].net),"add prefill group");
    }
    return b;
}
static BatchPrefill load_prefill(const char *path,int tokens,int blocks,int capacity,Layer *decode) {
    return load_prefill_history(path,tokens,blocks,capacity,capacity,decode);
}
/* Use the proven decode graph for one sensitive block, keeping all compute on
 * the NPU and the other blocks batched. CPU only transfers hidden/KV values. */
static void run_scalar_prefill_block(BatchPrefill *b,Layer *decode,int block,int pos,
                                    const float *cosines,const float *sines,double *api_seconds) {
    Layer *l=b->scalar,*batch=&b->layers[block],*kv=&decode[block];
    Tensor *input=(block&&!b->host_hidden)?&b->layers[block-1].out[0]:&batch->in[0];
    if(block&&!b->host_hidden)check(vip_flush_buffer(input->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read scalar prefill hidden");
    for(int k=0;k<FUSED_LAYERS;k++)for(int j=0;j<=b->capacity;j++)
        put(&l->in[IN_MASK(k)],j,(j<pos||j==b->capacity)?0:-10000);
    for(int t=0;t<b->tokens;t++){
        copy_values(&l->in[0],0,input,t*DIM,DIM);
        for(int k=0;k<FUSED_LAYERS;k++){
            if(pos+t)put(&l->in[IN_MASK(k)],pos+t-1,0);
            for(int j=0;j<HD;j++){
                put(&l->in[IN_COS(k)],j,cosines[t*HD+j]);
                put(&l->in[IN_SIN(k)],j,sines[t*HD+j]);
            }
        }
        for(unsigned port=0;port<TENSOR_MAX_INPUTS;port++){
            Tensor *input_port=port>=1&&port<1+2*FUSED_LAYERS?&kv->in[port]:&l->in[port];
            check(vip_flush_buffer(input_port->b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush scalar prefill input");
        }
        double stamp=now();check(vip_run_network(l->net),"scalar prefill block");*api_seconds+=now()-stamp;
        for(unsigned port=0;port<TENSOR_MAX_OUTPUTS;port++)
            check(vip_flush_buffer(l->out[port].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read scalar prefill block");
        copy_values(&batch->out[0],t*DIM,&l->out[0],0,DIM);
        for(int k=0;k<FUSED_LAYERS;k++){
            append_key(&kv->in[IN_K(k)],&l->out[OUT_K(k)],b->capacity,pos+t,1);
            for(int h=0;h<KV;h++)copy_values(&kv->in[IN_V(k)],(h*b->capacity+pos+t)*HD,&l->out[OUT_V(k)],h*HD,HD);
        }
    }
    for(int k=0;k<FUSED_LAYERS;k++)for(int t=0;t<pos+b->tokens;t++)put(&kv->in[IN_MASK(k)],t,0);
    check(vip_flush_buffer(batch->out[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"publish scalar prefill hidden");
}
static void run_prefill_rows(BatchPrefill *b,Layer *decode,const float *embedding,
                        const int32_t *prompt,int pos,float *last_hidden,double *api_seconds,int valid) {
    const int n=b->tokens,c=b->past_capacity,storage=b->capacity,repeat=DIM/(KV*HD);
    if(pos<b->previous_position||pos+n>storage||pos>c||valid<1||valid>n||(valid!=n&&b->scalar_block>=0)){
        fprintf(stderr,"prefill position/capacity/valid-row overflow\n");exit(2);
    }
    /* Earlier real queries cannot attend to padded future rows. Keep the
     * causal mask; publish only real rows and their last hidden. */
    for(int t=0;t<n;t++)for(int j=0;j<DIM;j++)
        put(&b->layers[0].in[0],t*DIM+j,opi_input_value(embedding,prompt[pos+(t<valid?t:valid-1)],pos+(t<valid?t:valid-1),j,DIM));
    float cosines[n*HD],sines[n*HD];
    for(int t=0;t<n;t++)for(int j=0;j<HD;j++){
        float angle=(pos+t)/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
        cosines[t*HD+j]=cosf(angle);sines[t*HD+j]=sinf(angle);
    }
    if(b->grouped){
        /* Prepare all independent input ports first; linked hidden tensors stay
         * on device. KV publication occurs only after the whole group finishes. */
        for(int i=0;i<b->blocks;i++){
            Layer *l=&b->layers[i];
            for(int k=0;k<FUSED_LAYERS;k++){
                for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=b->previous_position;j<pos;j++)
                    put(&l->in[IN_MASK(k)],(r*n+t)*(c+n)+j,0);
                for(int j=0;j<n*HD;j++){
                    put(&l->in[IN_COS(k)],j,cosines[j]);put(&l->in[IN_SIN(k)],j,sines[j]);
                }
                for(int kind=0;kind<2;kind++){
                    unsigned port=kind?IN_V(k):IN_K(k);
                    Tensor *input=&decode[i].in[port];
                    if(c!=storage){input=&l->in[port];pack_history(input,&decode[i].in[port],c,storage,pos,!kind);}
                    check(vip_flush_buffer(input->b,VIP_BUFFER_OPER_TYPE_FLUSH),"prefill history");
                }
            }
            if(!i)check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"group hidden");
            for(unsigned j=1+2*FUSED_LAYERS;j<TENSOR_MAX_INPUTS;j++)
                check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"group position");
        }
        double stamp=now();
        if(getenv("OPI_FIRST_PREFILL_L2")){
            /* Permanent ownership: blocks0 and1 never join any group.
             * The suffix group owns blocks2..end for the full lifetime. */
            check(vip_run_network(b->layers[0].net),"prefill first block");
            check(vip_run_network(pos==0?b->scalar->net:b->layers[1].net),"prefill sensitive block");
            check(vip_run_group(b->group,b->blocks-2),"prefill suffix group");
            if(pos==0){
                fprintf(stderr,"FIRST_CHUNK_ORIGINAL block=1 pos=0 rows=%d valid=%d\n",n,valid);
                fprintf(stderr,"FIRST_CHUNK_SUFFIX_GROUP blocks=%d\n",b->blocks-2);
            }
        }else check(vip_run_group(b->group,b->blocks),"prefill group");
        *api_seconds+=now()-stamp;
        for(int i=0;i<b->blocks;i++){
            Layer *l=&b->layers[i];
            for(unsigned j=1;j<TENSOR_MAX_OUTPUTS;j++)
                check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"group read KV");
            for(int k=0;k<FUSED_LAYERS;k++){
                append_key_rows(&decode[i].in[IN_K(k)],&l->out[OUT_K(k)],storage,pos,valid,n);
                for(int h=0;h<KV;h++)copy_values(&decode[i].in[IN_V(k)],(h*storage+pos)*HD,&l->out[OUT_V(k)],h*n*HD,valid*HD);
                for(int t=0;t<pos+valid;t++)put(&decode[i].in[IN_MASK(k)],t,0);
            }
        }
        goto read_last_hidden;
    }
    for(int i=0;i<b->blocks;i++){
        Layer *l=&b->layers[i];
        if(i&&b->host_hidden)copy_values(&l->in[0],0,&b->layers[i-1].out[0],0,n*DIM);
        if(i==b->scalar_block){
            run_scalar_prefill_block(b,decode,i,pos,cosines,sines,api_seconds);
            continue;
        }
        for(int k=0;k<FUSED_LAYERS;k++){
            for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=b->previous_position;j<pos;j++)
                put(&l->in[IN_MASK(k)],((r*n+t)*(c+n)+j),0);
            for(int j=0;j<n*HD;j++){
                put(&l->in[IN_COS(k)],j,cosines[j]);put(&l->in[IN_SIN(k)],j,sines[j]);
            }
            for(int kind=0;kind<2;kind++){
                unsigned port=kind?IN_V(k):IN_K(k);
                Tensor *input=&decode[i].in[port];
                if(c!=storage){input=&l->in[port];pack_history(input,&decode[i].in[port],c,storage,pos,!kind);}
                check(vip_flush_buffer(input->b,VIP_BUFFER_OPER_TYPE_FLUSH),"prefill history");
            }
        }
        if(!i||b->host_hidden)check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush batch hidden");
        for(unsigned j=1+2*FUSED_LAYERS;j<TENSOR_MAX_INPUTS;j++)
            check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush batch position");
        capture_prefill_io(b,decode,pos,i,0);
        if(getenv("OPI_PREFILL_POISON_OUTPUT")){
            for(unsigned j=0;j<l->out[0].count;j++)put(&l->out[0],j,-123.0f);
            check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"publish prefill output sentinel");
        }
        if(getenv("OPI_PREFILL_FLUSH_OUTPUT"))
            check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush prefill hidden before device write");
        if(getenv("OPI_PREFILL_CLEAR_OUTPUT")){
            Tensor *out=&l->out[0];
            if(out->p.data_format!=VIP_BUFFER_FORMAT_FP16||out->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE)exit(2);
            memset(out->data,0,out->count*sizeof(__fp16));
            check(vip_flush_buffer(out->b,VIP_BUFFER_OPER_TYPE_FLUSH),"publish cleared prefill hidden");
        }
        if(getenv("OPI_PREFILL_REBIND_OUTPUTS"))
            for(unsigned o=0;o<TENSOR_MAX_OUTPUTS;o++)check(vip_set_output(l->net,o,l->out[o].b),"rebind batch output");
        int initial_used=pos==0&&i==1&&getenv("OPI_FIRST_PREFILL_L2");
        double stamp=now();check(vip_run_network(initial_used?b->scalar->net:l->net),"batch prefill");*api_seconds+=now()-stamp;
        if(initial_used)fprintf(stderr,"FIRST_CHUNK_ORIGINAL block=1 pos=0 rows=%d valid=%d\n",n,valid);
        capture_prefill_io(b,decode,pos,i,1);
        audit_outputs(l,TENSOR_MAX_OUTPUTS,"prefill",pos,i);
        for(unsigned j=b->host_hidden?0:1;j<TENSOR_MAX_OUTPUTS;j++)
            check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read batch KV");
        for(int k=0;k<FUSED_LAYERS;k++){
            append_key_rows(&decode[i].in[IN_K(k)],&l->out[OUT_K(k)],storage,pos,valid,n);
            for(int h=0;h<KV;h++){
                copy_values(&decode[i].in[IN_V(k)],(h*storage+pos)*HD,&l->out[OUT_V(k)],h*n*HD,valid*HD);
            }
            for(int t=0;t<pos+valid;t++)put(&decode[i].in[IN_MASK(k)],t,0);
        }
    }
read_last_hidden: ;
    Tensor *out=&b->layers[b->blocks-1].out[0];
    b->previous_position=pos;
    check(vip_flush_buffer(out->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read batch hidden");
    for(int j=0;j<DIM;j++){
        last_hidden[j]=get(out,(valid-1)*DIM+j);
        if(!isfinite(last_hidden[j])){fprintf(stderr,"nonfinite batch hidden\n");exit(3);}
    }
}
static void run_prefill(BatchPrefill *b,Layer *decode,const float *embedding,
                        const int32_t *prompt,int pos,float *last_hidden,double *api_seconds) {
    run_prefill_rows(b,decode,embedding,prompt,pos,last_hidden,api_seconds,b->tokens);
}
static void free_prefill(BatchPrefill *b) {
    if(b->grouped)check(vip_destroy_group(b->group),"free prefill group");
    /* Aliases belong to decode/preceding output. Free only owned port buffers. */
    for(int i=0;i<b->blocks;i++)free_net(&b->layers[i],TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);
    free(b->layers);
    if(b->scalar){free_net(b->scalar,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS);free(b->scalar);}
}
