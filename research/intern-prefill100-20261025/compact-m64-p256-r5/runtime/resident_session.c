/* Experimental one-slot, resident full-depth NPU session.
 * The existing benchmark entry stays unchanged. Use a private pipe for JSON
 * events so SDK diagnostics cannot corrupt the protocol. Binary requests are
 * eight little-endian uint32 words followed by token IDs and optional feedback:
 * magic, request_id, prompt_count, max_new, feedback_count, reset, flags, reserved.
 * flags bit 0 dumps complete logits for correctness tests, never speed reports.
 */
#define OPI_FULL_MODEL_ENTRY full_model_example_main
#include "full_model.c"
#include "greedy_single_pass.h"
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error Resident protocol requires a little-endian host
#endif

typedef struct {
    int single_pass_greedy;
    Layer *decoder, head[HEAD_SHARDS];
    BatchPrefill batch, tail8, compact, compact256;
    vip_group decoder_group, head_group;
    int layers, blocks, capacity, vocab, chunk, valid;
    int32_t *history;
    const float *embedding;
    float *hidden;
} Session;
#include "mask_templates.h"
#include "session_release.h"

static void reset_session(Session *s) {
    int templated=getenv("OPI_MASK_TEMPLATE")!=NULL;
    s->valid=0;s->batch.previous_position=0;s->tail8.previous_position=0;s->compact.previous_position=0;s->compact256.previous_position=0;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Layer *l=&s->decoder[i];
        if(templated)restore_mask_template(&l->in[IN_MASK(k)],s->capacity,0,1,0);
        else{for(int j=0;j<s->capacity;j++)put(&l->in[IN_MASK(k)],j,-10000);
             put(&l->in[IN_MASK(k)],s->capacity,0);}
        for(unsigned j=0;j<l->in[IN_K(k)].count;j++){
            put(&l->in[IN_K(k)],j,0);put(&l->in[IN_V(k)],j,0);
        }
        BatchPrefill *banks[4]={&s->batch,&s->tail8,&s->compact,&s->compact256};
        for(int bank=0;bank<4;bank++)if(banks[bank]->tokens){
            Tensor *m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=banks[bank]->past_capacity,repeat=DIM/(KV*HD);
            if(templated)restore_mask_template(m,c,n,repeat,0);
            else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
                put(m,(r*n+t)*(c+n)+j,(j>=c&&j<=c+t)?0:-10000);
        }
    }
}

/* Causal prefix entries are independent of a discarded dynamic suffix.
 * Close every stale mask column before reusing the prefix; batch masks only
 * open columns monotonically during generation, so rewinding the cursor alone
 * is insufficient. Stale finite K/V values remain unreachable behind masks. */
static void truncate_session(Session *s,int keep) {
    int templated=getenv("OPI_MASK_TEMPLATE")!=NULL;
    s->valid=keep;s->batch.previous_position=keep;s->tail8.previous_position=keep;s->compact.previous_position=keep<128?keep:128;s->compact256.previous_position=keep<256?keep:256;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Tensor *m=&s->decoder[i].in[IN_MASK(k)];
        if(templated)restore_mask_template(m,s->capacity,0,1,keep);
        else{for(int j=0;j<s->capacity;j++)put(m,j,j<keep?0:-10000);
             put(m,s->capacity,0);}
        BatchPrefill *banks[4]={&s->batch,&s->tail8,&s->compact,&s->compact256};
        for(int bank=0;bank<4;bank++)if(banks[bank]->tokens){
            m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=banks[bank]->past_capacity,repeat=DIM/(KV*HD);
            if(templated)restore_mask_template(m,c,n,repeat,keep<c?keep:c);
            else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
                put(m,(r*n+t)*(c+n)+j,(j<(keep<c?keep:c)||(j>=c&&j<=c+t))?0:-10000);
        }
    }
}

static void session_scalar(Session *s,int token,int pos,double *api) {
    Layer *d=s->decoder;
    for(int j=0;j<DIM;j++)put(&d[0].in[0],j,s->embedding[(size_t)token*DIM+j]);
    float cosine[HD],sine[HD];
    for(int j=0;j<HD;j++){
        float angle=pos/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
        cosine[j]=cosf(angle);sine[j]=sinf(angle);
    }
    for(int i=0;i<s->blocks;i++){
        Layer *l=&d[i];
        for(int k=0;k<FUSED_LAYERS;k++){
            if(pos)put(&l->in[IN_MASK(k)],pos-1,0);
            for(int j=0;j<HD;j++){
                put(&l->in[IN_COS(k)],j,cosine[j]);put(&l->in[IN_SIN(k)],j,sine[j]);
            }
        }
        for(unsigned j=i?1:0;j<TENSOR_MAX_INPUTS;j++)
            check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"session flush");
    }
    double start=now();
    check(vip_run_group(s->decoder_group,s->blocks),"session decoder");*api+=now()-start;
    for(int i=0;i<s->blocks;i++){
        Layer *l=&d[i];
        for(unsigned j=i==s->blocks-1?0:1;j<TENSOR_MAX_OUTPUTS;j++)
            check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"session read");
        for(int k=0;k<FUSED_LAYERS;k++){
            append_key(&l->in[IN_K(k)],&l->out[OUT_K(k)],s->capacity,pos,1);
            for(int h=0;h<KV;h++)copy_values(&l->in[IN_V(k)],(h*s->capacity+pos)*HD,&l->out[OUT_V(k)],h*HD,HD);
        }
    }
    for(int j=0;j<DIM;j++){
        s->hidden[j]=get(&d[s->blocks-1].out[0],j);
        if(!isfinite(s->hidden[j])){fprintf(stderr,"nonfinite session hidden\n");exit(3);}
    }
}

static int session_head(Session *s,double *api,FILE *dump) {
    for(int i=0;i<HEAD_SHARDS;i++){
        for(int j=0;j<DIM;j++)put(&s->head[i].in[0],j,s->hidden[j]);
        check(vip_flush_buffer(s->head[i].in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"session head flush");
    }
    double start=now();check(vip_run_group(s->head_group,HEAD_SHARDS),"session head");*api+=now()-start;
    int best=-1,reference=-1;float value=-INFINITY,refvalue=-INFINITY;
    for(int i=0;i<HEAD_SHARDS;i++){
        Tensor *out=&s->head[i].out[0];
        check(vip_flush_buffer(out->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"session head read");
        unsigned maximum=0;
        if(out->p.data_format==VIP_BUFFER_FORMAT_INT16&&
           out->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT&&
           out->p.quant_data.dfp.fixed_point_pos>=-110&&out->p.quant_data.dfp.fixed_point_pos<=110)
            maximum=s->single_pass_greedy?greedy_i16_single_pass((const int16_t*)out->data,out->count):greedy_i16((const int16_t*)out->data,out->count);
        else{
            float current=-INFINITY;
            for(unsigned j=0;j<out->count;j++){
                float x=get(out,j);if(!isfinite(x))exit(3);
                if(x>current){current=x;maximum=j;}
            }
        }
        float x=get(out,maximum);if(!isfinite(x))exit(3);
        if(x>value){value=x;best=i*s->chunk+(int)maximum;}
        if(dump)for(unsigned j=0;j<out->count;j++){
            x=get(out,j);if(!isfinite(x))exit(3);
            if(x>refvalue){refvalue=x;reference=i*s->chunk+(int)j;}
            if(fwrite(&x,sizeof(x),1,dump)!=1)exit(2);
        }
    }
    if(best<0||(dump&&best!=reference))exit(3);
    return best;
}

int main(int argc,char **argv) {
    if(argc!=8){fprintf(stderr,"usage: resident networks layers capacity vocab embeddings prefill_tokens eos_id\n");return 2;}
    Session s={0};s.single_pass_greedy=getenv("OPI_GREEDY_I16_SINGLE_PASS")!=NULL;s.layers=atoi(argv[2]);s.capacity=atoi(argv[3]);s.vocab=atoi(argv[4]);
    int batch_tokens=atoi(argv[6]),eos=atoi(argv[7]);
    int tail_min=getenv("OPI_TTFT_TAIL_MIN")?atoi(getenv("OPI_TTFT_TAIL_MIN")):0;
    if(tail_min<0||tail_min>batch_tokens)return 2;
    if(s.layers<1||s.layers>80||s.layers%FUSED_LAYERS||s.capacity<1||s.capacity>8192||
       s.vocab<1||batch_tokens<0||batch_tokens==1||batch_tokens>256||batch_tokens>s.capacity||eos<0||eos>=s.vocab)return 2;
    if(!getenv("OPI_SESSION_FD"))return 2;
    FILE *events=fdopen(atoi(getenv("OPI_SESSION_FD")),"w");if(!events)return 2;
    setvbuf(events,NULL,_IOLBF,0);
    s.blocks=s.layers/FUSED_LAYERS;s.chunk=HEAD_SHARDS==1?s.vocab:HEAD_CHUNK;
    if(s.chunk<1||(s.vocab+s.chunk-1)/s.chunk!=HEAD_SHARDS)return 2;
    int emb_fd=open(argv[5],O_RDONLY);struct stat st;
    if(emb_fd<0||fstat(emb_fd,&st)||(size_t)st.st_size!=(size_t)s.vocab*DIM*sizeof(float))return 2;
    s.embedding=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,emb_fd,0);if(s.embedding==MAP_FAILED)return 2;
    s.decoder=calloc(s.blocks,sizeof(Layer));s.hidden=calloc(DIM,sizeof(float));
    s.history=calloc(s.capacity,sizeof(int32_t));int32_t *prompt=calloc(s.capacity,sizeof(int32_t));
    int32_t *feedback=calloc(s.capacity,sizeof(int32_t));
    if(!s.decoder||!s.hidden||!s.history||!prompt||!feedback)return 2;
    double load_start=now();check(vip_init(),"session init");
    unsigned ins[TENSOR_MAX_INPUTS]={DIM},outs[TENSOR_MAX_OUTPUTS]={DIM},timeout=30000;
    for(int k=0;k<FUSED_LAYERS;k++){
        ins[IN_K(k)]=ins[IN_V(k)]=KV*s.capacity*HD;ins[IN_MASK(k)]=s.capacity+1;
        ins[IN_COS(k)]=ins[IN_SIN(k)]=HD;outs[OUT_K(k)]=outs[OUT_V(k)]=KV*HD;
    }
    check(vip_create_group(s.blocks,&s.decoder_group),"session group");
    check(vip_set_group(s.decoder_group,VIP_GROUP_PROP_SET_TIME_OUT,&timeout),"session group timeout");
    for(int i=0;i<s.blocks;i++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],i*FUSED_LAYERS);
        load_net(&s.decoder[i],path,TENSOR_MAX_INPUTS,TENSOR_MAX_OUTPUTS,ins,outs);
        for(int k=0;k<FUSED_LAYERS;k++)if(KEY_CACHE_TRANSPOSED){
            Tensor *key=&s.decoder[i].in[IN_K(k)];
            if(key->p.num_of_dims!=4||key->p.sizes[0]!=(unsigned)s.capacity||
               key->p.sizes[1]!=HD||key->p.sizes[2]!=KV||key->p.sizes[3]!=1)return 2;
        }
        if(i){
            Tensor *a=&s.decoder[i-1].out[0],*b=&s.decoder[i].in[0];
            if(!same_tensor_contract(a,b)||a->p.data_format!=VIP_BUFFER_FORMAT_FP16||
               a->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE)return 2;
            check(vip_set_input(s.decoder[i].net,0,a->b),"session hidden alias");
        }
        check(vip_add_network(s.decoder_group,s.decoder[i].net),"session add decoder");
    }
    check(vip_create_group(HEAD_SHARDS,&s.head_group),"session head group");
    check(vip_set_group(s.head_group,VIP_GROUP_PROP_SET_TIME_OUT,&timeout),"session head timeout");
    for(int i=0;i<HEAD_SHARDS;i++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],s.layers+i);
        unsigned ni[]={DIM},no[]={(unsigned)(s.vocab-i*s.chunk<s.chunk?s.vocab-i*s.chunk:s.chunk)};
        load_net(&s.head[i],path,1,1,ni,no);
        check(vip_add_network(s.head_group,s.head[i].net),"session add head");
    }
    s.batch=load_prefill(argv[1],batch_tokens,s.blocks,s.capacity,s.decoder);
    const char *tail_dir=getenv("OPI_PREFILL8_DIR");
    if(tail_dir){
        /* Separate graph topology: weights are not claimed to be shared. Both
         * prefill sets alias only the checked decoder KV input buffers. */
        /* This candidate experiment is restricted to 256/448/512-capacity
         * design. load_prefill also checks every actual KV IO contract, so a
         * mistakenly supplied c1280 tail graph cannot pass by filename alone. */
        if(!*tail_dir||!((s.capacity==256&&batch_tokens==16)||((s.capacity==448||s.capacity==512)&&(batch_tokens==16||batch_tokens==32)))||getenv("OPI_PREFILL_SCALAR_BLOCK"))return 2;
        s.tail8=load_prefill(tail_dir,8,s.blocks,s.capacity,s.decoder);
    }
    const char *compact_dir=getenv("OPI_COMPACT64_DIR");
    if(compact_dir){
        if(!*compact_dir||s.capacity!=448||batch_tokens!=32||!s.tail8.tokens)return 2;
        s.compact=load_prefill_history(compact_dir,64,s.blocks,128,s.capacity,s.decoder);
    }
    const char *compact256_dir=getenv("OPI_COMPACT256_DIR");
    if(compact256_dir){
        if(!*compact256_dir||s.capacity!=448||batch_tokens!=32||!s.compact.tokens)return 2;
        s.compact256=load_prefill_history(compact256_dir,64,s.blocks,256,s.capacity,s.decoder);
    }
    reset_session(&s);
    fprintf(stderr,"mask_templates enabled=%d host_bytes=%zu\n",getenv("OPI_MASK_TEMPLATE")!=NULL,mask_template_bytes);
    fprintf(events,"{\"event\":\"ready\",\"layers\":%d,\"capacity\":%d,\"load_seconds\":%.6f,\"driver_bytes\":%lld,\"true_tail8_enabled\":%s,\"single_pass_greedy_enabled\":%s}\n",s.layers,s.capacity,now()-load_start,npu_bytes(),s.tail8.tokens?"true":"false",s.single_pass_greedy?"true":"false");
    uint32_t header[8];size_t header_bytes;
    while((header_bytes=fread(header,1,sizeof(header),stdin))!=0){
        if(header_bytes!=sizeof(header))return 2;
        unsigned id=header[1];int n=(int)header[2],limit=(int)header[3],nf=(int)header[4];
        if(header[0]!=0x4f504931||header[5]>1||header[6]>1||header[7]||
           n<1||n>s.capacity||limit<1||limit>s.capacity||n+limit-1>s.capacity||
           (nf!=0&&nf!=limit))return 2;
        if(fread(prompt,sizeof(int32_t),n,stdin)!=(size_t)n||
           (nf&&fread(feedback,sizeof(int32_t),nf,stdin)!=(size_t)nf))return 2;
        for(int i=0;i<n;i++)if(prompt[i]<0||prompt[i]>=s.vocab)return 2;
        for(int i=0;i<nf;i++)if(feedback[i]<0||feedback[i]>=s.vocab)return 2;
        FILE *dump=NULL;
        if(header[6]){
            if(!getenv("OPI_SESSION_DUMP_DIR"))return 2;
            char path[1024];snprintf(path,sizeof(path),"%s/%u-logits.f32",getenv("OPI_SESSION_DUMP_DIR"),id);
            dump=fopen(path,"wbx");if(!dump)return 2;
        }
        double start=now(),cpu_start=cpu_time(),api=0,first=0;
        int temp_start=temperature();if(temp_start<0||temp_start>=80000)return 2;
        int reused=0;
        if(!header[5])while(reused<s.valid&&reused<n-1&&s.history[reused]==prompt[reused])reused++;
        /* Retain complete prefill batches, so the same prefix is computed with
         * the same scalar/batch partition as a fresh request. Re-evaluate at
         * least the final prompt token: hidden stores the previous request. */
        if(s.batch.tokens&&!getenv("OPI_EXACT_PREFIX"))reused-=reused%s.batch.tokens;
        if(!reused)reset_session(&s);
        else truncate_session(&s,reused);
        double setup_seconds=now()-start,prefill_seconds=0,scalar_prompt_seconds=0,first_head_seconds=0;
        int batch_calls=0,tail_rows=0,scalar_prompt_tokens=0;
        int generated=0,last=-1;
        int tail8_calls=0,compact_calls=0,compact256_calls=0;double tail8_seconds=0,compact_seconds=0,compact256_seconds=0;
        for(int pos=reused;pos<n+limit-1;pos++){
            if((pos-reused)%32==0&&temperature()>=80000)return 2;
            int selected=compact_bank_selection(s.compact.tokens,s.compact256.tokens,pos,reused,n);
            if(selected){
                BatchPrefill *bank=selected==1?&s.compact:&s.compact256;
                double bt=now();run_prefill(bank,s.decoder,s.embedding,prompt,pos,s.hidden,&api);
                double elapsed=now()-bt;prefill_seconds+=elapsed;compact_seconds+=elapsed;batch_calls++;compact_calls++;
                if(selected==2){compact256_calls++;compact256_seconds+=elapsed;}
                memcpy(s.history+pos,prompt+pos,bank->tokens*sizeof(int32_t));pos+=bank->tokens-1;
            }else if(s.batch.tokens&&pos+s.batch.tokens<=n){
                double bt=now();
                run_prefill(&s.batch,s.decoder,s.embedding,prompt,pos,s.hidden,&api);
                prefill_seconds+=now()-bt;batch_calls++;
                memcpy(s.history+pos,prompt+pos,s.batch.tokens*sizeof(int32_t));
                pos+=s.batch.tokens-1;
            }else if(s.tail8.tokens&&pos<n&&n-pos==8){
                /* Only a complete final 8-row block; all earlier bulk calls
                 * and the decoder token feedback order remain unchanged. */
                double bt=now();
                run_prefill(&s.tail8,s.decoder,s.embedding,prompt,pos,s.hidden,&api);
                double elapsed=now()-bt;prefill_seconds+=elapsed;tail8_seconds+=elapsed;
                batch_calls++;tail8_calls++;tail_rows=8;
                memcpy(s.history+pos,prompt+pos,8*sizeof(int32_t));pos+=7;
            }else if(s.batch.tokens&&tail_min&&pos<n&&n-pos>=tail_min&&pos+s.batch.tokens<=s.capacity){
                int valid=n-pos;double bt=now();
                run_prefill_rows(&s.batch,s.decoder,s.embedding,prompt,pos,s.hidden,&api,valid);
                prefill_seconds+=now()-bt;batch_calls++;tail_rows=valid;
                memcpy(s.history+pos,prompt+pos,valid*sizeof(int32_t));pos+=valid-1;
            }else{
                int token=pos<n?prompt[pos]:(nf?feedback[generated-1]:last);
                double st=now();session_scalar(&s,token,pos,&api);s.history[pos]=token;
                if(pos<n){scalar_prompt_seconds+=now()-st;scalar_prompt_tokens++;}
            }
            s.valid=pos+1;
            if(pos<n-1)continue;
            double ht=now();last=session_head(&s,&api,dump);
            if(generated==0)first_head_seconds=now()-ht;
            generated++;
            double elapsed=now()-start;if(generated==1)first=elapsed;
            fprintf(events,"{\"event\":\"token\",\"id\":%u,\"token\":%d,\"elapsed\":%.6f}\n",id,last,elapsed);
            if(!nf&&last==eos)break;
        }
        double total=now()-start,cpu=cpu_time()-cpu_start;
        if(dump)fclose(dump);
        fprintf(events,"{\"event\":\"done\",\"id\":%u,\"prompt_tokens\":%d,\"reused_tokens\":%d,\"new_tokens\":%d,\"kv_tokens\":%d,\"generated_tokens\":%d,\"setup_seconds\":%.6f,\"prefill_seconds\":%.6f,\"scalar_prompt_seconds\":%.6f,\"first_head_seconds\":%.6f,\"batch_calls\":%d,\"tail_rows\":%d,\"scalar_prompt_tokens\":%d,\"ttft_seconds\":%.6f,\"total_seconds\":%.6f,\"decode_tokens_per_second\":%.6f,\"api_seconds\":%.6f,\"cpu_seconds\":%.6f,\"rss_kib\":%ld,\"driver_bytes\":%lld,\"temperature_start_mC\":%d,\"temperature_end_mC\":%d,\"diagnostic\":%s,\"fixed_feedback\":%s,\"finish_reason\":\"%s\",\"true_tail8_calls\":%d,\"true_tail8_seconds\":%.6f,\"compact_calls\":%d,\"compact_seconds\":%.6f,\"compact256_calls\":%d,\"compact256_seconds\":%.6f}\n",
                id,n,reused,n-reused,s.valid,generated,setup_seconds,prefill_seconds,scalar_prompt_seconds,first_head_seconds,batch_calls,tail_rows,scalar_prompt_tokens,first,total,generated>1?(generated-1)/(total-first):0,
                api,cpu,current_rss_kib(),npu_bytes(),temp_start,temperature(),dump?"true":"false",nf?"true":"false",!nf&&last==eos?"eos":"length",tail8_calls,tail8_seconds,compact_calls,compact_seconds,compact256_calls,compact256_seconds);
    }
    if(!feof(stdin))return 2;
    release_session_networks_and_buffers(&s);
    fprintf(stderr,"CLOSE phase=vip_destroy\n");
    check(vip_destroy(),"session release");
    fprintf(stderr,"CLOSE phase=mask_templates\n");free_mask_templates();
    fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld}\n",npu_bytes());
    fclose(events);free(s.decoder);free(s.hidden);free(s.history);free(prompt);free(feedback);
    munmap((void*)s.embedding,st.st_size);close(emb_fd);return 0;
}
