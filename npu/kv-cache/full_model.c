/* Full language stack and LM head on VIPLite; CPU embeds and samples.
 * Reuse the tested IO/quantization utilities from the incremental probe.
 * Compile with MODEL_DIM/MODEL_KV/MODEL_HD/ROPE_BASE from the model manifest.
 */
#include <sys/wait.h>
#include <sys/prctl.h>
#include <signal.h>
#ifndef FUSED_LAYERS
#define FUSED_LAYERS 1
#endif
#ifndef KEY_CACHE_TRANSPOSED
#define KEY_CACHE_TRANSPOSED 0
#endif
#define TENSOR_MAX_INPUTS (1+5*FUSED_LAYERS)
#define TENSOR_MAX_OUTPUTS (1+2*FUSED_LAYERS)
#define IN_K(k) (1+(k))
#define IN_V(k) (1+FUSED_LAYERS+(k))
#define IN_MASK(k) (1+2*FUSED_LAYERS+(k))
#define IN_COS(k) (1+3*FUSED_LAYERS+(k))
#define IN_SIN(k) (1+4*FUSED_LAYERS+(k))
#define OUT_K(k) (1+(k))
#define OUT_V(k) (1+FUSED_LAYERS+(k))
#define main two_layer_probe_example_main
#include "board_probe.c"
#undef main
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "greedy_i16.h"
#ifndef HEAD_SHARDS
#define HEAD_SHARDS 1
#define HEAD_CHUNK 0
#endif
static long long read_frequency(const char *path) {
    long long value=-1;FILE *f=fopen(path,"r");
    if(f){if(fscanf(f,"%lld",&value)!=1)value=-1;fclose(f);}return value;
}
static long current_rss_kib(void) {
    char line[256];long value=-1;FILE *f=fopen("/proc/self/status","r");
    if(f){while(fgets(line,sizeof(line),f))if(sscanf(line,"VmRSS: %ld kB",&value)==1)break;fclose(f);}
    return value;
}

static void load_net(Layer *layer,const char *path,unsigned ni,unsigned no,
                     const unsigned *ins,const unsigned *outs) {
    check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&layer->net),"load model");
    uint32_t actual_in=0,actual_out=0,timeout=30000;
    check(vip_query_network(layer->net,VIP_NETWORK_PROP_INPUT_COUNT,&actual_in),"inputs");
    check(vip_query_network(layer->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&actual_out),"outputs");
    if(actual_in!=ni||actual_out!=no){fprintf(stderr,"wrong network contract\n");exit(2);}
    check(vip_set_network(layer->net,VIP_NETWORK_PROP_SET_TIME_OUT,&timeout),"timeout");
    check(vip_prepare_network(layer->net),"prepare full model");
    for(unsigned i=0;i<ni;i++)tensor_init(layer,&layer->in[i],i,0,ins[i]);
    for(unsigned i=0;i<no;i++)tensor_init(layer,&layer->out[i],i,1,outs[i]);
}
static void free_net(Layer *layer,unsigned ni,unsigned no) {
    for(unsigned i=0;i<ni;i++){check(vip_unmap_buffer(layer->in[i].b),"unmap");check(vip_destroy_buffer(layer->in[i].b),"free");}
    for(unsigned i=0;i<no;i++){check(vip_unmap_buffer(layer->out[i].b),"unmap");check(vip_destroy_buffer(layer->out[i].b),"free");}
    check(vip_destroy_network(layer->net),"free network");
}
static void execute(Layer *layer,unsigned ni,unsigned no,double *run_seconds) {
    for(unsigned i=0;i<ni;i++)check(vip_flush_buffer(layer->in[i].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush full model");
    double start=now();check(vip_run_network(layer->net),"full model inference");*run_seconds+=now()-start;
    for(unsigned i=0;i<no;i++)check(vip_flush_buffer(layer->out[i].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read full model");
}
static void copy_values(Tensor *dst,unsigned offset,Tensor *src,unsigned begin,unsigned count) {
    if(offset+count>dst->count||begin+count>src->count){fprintf(stderr,"copy overflow\n");exit(2);}
    int same=dst->p.data_format==src->p.data_format&&dst->p.quant_format==src->p.quant_format;
    if(same&&dst->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        same=dst->p.quant_data.dfp.fixed_point_pos==src->p.quant_data.dfp.fixed_point_pos;
    unsigned size=dst->p.data_format==VIP_BUFFER_FORMAT_FP32?4:
        (dst->p.data_format==VIP_BUFFER_FORMAT_FP16||dst->p.data_format==VIP_BUFFER_FORMAT_INT16?2:
        (dst->p.data_format==VIP_BUFFER_FORMAT_INT8?1:0));
    if(same&&size)memcpy((char*)dst->data+offset*size,(char*)src->data+begin*size,count*size);
    else for(unsigned j=0;j<count;j++)put(dst,offset+j,get(src,begin+j));
}
/* Publish only the new K rows; never transpose or requantize old history.
 * Batched graph outputs remain [KV,tokens,HD] for both cache layouts. */
static void append_key(Tensor *dst,Tensor *src,int capacity,int pos,int tokens) {
    if(pos<0||tokens<1||pos+tokens>capacity||
       dst->count!=(unsigned)(KV*capacity*HD)||src->count!=(unsigned)(KV*tokens*HD)){
        fprintf(stderr,"append K shape/range mismatch\n");exit(2);
    }
    if(!KEY_CACHE_TRANSPOSED){
        for(int h=0;h<KV;h++)copy_values(dst,(h*capacity+pos)*HD,src,h*tokens*HD,tokens*HD);
        return;
    }
    int raw16=dst->p.data_format==src->p.data_format&&dst->p.quant_format==src->p.quant_format&&
        ((dst->p.data_format==VIP_BUFFER_FORMAT_FP16&&dst->p.quant_format==VIP_BUFFER_QUANTIZE_NONE)||
         (dst->p.data_format==VIP_BUFFER_FORMAT_INT16&&dst->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT&&
          dst->p.quant_data.dfp.fixed_point_pos==src->p.quant_data.dfp.fixed_point_pos));
    for(int h=0;h<KV;h++)for(int j=0;j<HD;j++)for(int t=0;t<tokens;t++){
        unsigned to=(h*HD+j)*capacity+pos+t,from=(h*tokens+t)*HD+j;
        if(raw16)memcpy((char*)dst->data+to*2,(char*)src->data+from*2,2);
        else put(dst,to,get(src,from));
    }
}
/* Opt-in diagnostic readback, excluded from all performance results. */
static FILE *audit_file;
static void audit_outputs(Layer *layer,unsigned no,const char *stage,int pos,int block) {
    if(!audit_file)return;
    for(unsigned o=0;o<no;o++){
        Tensor *t=&layer->out[o];
        check(vip_flush_buffer(t->b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"audit output");
        uint64_t hash=UINT64_C(14695981039346656037);
        for(unsigned j=0;j<t->count;j++){
            float value=get(t,j);uint32_t bits;memcpy(&bits,&value,sizeof(bits));
            for(unsigned b=0;b<4;b++){hash^=(bits>>(8*b))&255;hash*=UINT64_C(1099511628211);}
        }
        fprintf(audit_file,"%s\t%d\t%d\t%u\t%u\t%016llx\n",stage,pos,block,o,t->count,(unsigned long long)hash);
    }
}
#include "batch_prefill.h"
#ifndef OPI_FULL_MODEL_ENTRY
#define OPI_FULL_MODEL_ENTRY main
#endif
int OPI_FULL_MODEL_ENTRY(int argc,char **argv) {
    if(argc!=11){fprintf(stderr,"usage: full_model networks layers capacity vocab embeddings.f32 prompt.i32 max_new tokens.json logits.f32 eos_id\n");return 2;}
    int count=atoi(argv[2]),capacity=atoi(argv[3]),vocab=atoi(argv[4]),max_new=atoi(argv[7]),eos=atoi(argv[10]);
    if(count<1||count>80||capacity<1||capacity>8192||vocab<1||max_new<1||max_new>capacity)return 2;
    if(count%FUSED_LAYERS)return 2;
    int blocks=count/FUSED_LAYERS;
    unsigned ni=TENSOR_MAX_INPUTS,no=TENSOR_MAX_OUTPUTS;
    int emb_fd=open(argv[5],O_RDONLY),prompt_fd=open(argv[6],O_RDONLY);struct stat emb_stat,prompt_stat;
    if(emb_fd<0||prompt_fd<0||fstat(emb_fd,&emb_stat)||fstat(prompt_fd,&prompt_stat))return 2;
    if((size_t)emb_stat.st_size!=(size_t)vocab*DIM*sizeof(float)||prompt_stat.st_size<4||prompt_stat.st_size%4)return 2;
    int prompt_count=prompt_stat.st_size/4;
    int prefill_tokens=getenv("OPI_PREFILL_TOKENS")?atoi(getenv("OPI_PREFILL_TOKENS")):0;
    int cached_prefix=getenv("OPI_CACHED_PREFIX")?atoi(getenv("OPI_CACHED_PREFIX")):0;
    int scalar_prefix=getenv("OPI_PREFILL_SCALAR_PREFIX")?atoi(getenv("OPI_PREFILL_SCALAR_PREFIX")):0;
    if(prefill_tokens<0||prefill_tokens==1||prefill_tokens>256||prefill_tokens>capacity||
       cached_prefix<0||cached_prefix>=prompt_count||scalar_prefix<0||scalar_prefix>prompt_count)return 2;
    if(prompt_count+max_new-1>capacity){fprintf(stderr,"prompt plus output exceeds KV capacity\n");return 2;}
    const float *emb=mmap(NULL,emb_stat.st_size,PROT_READ,MAP_PRIVATE,emb_fd,0);
    const int32_t *prompt=mmap(NULL,prompt_stat.st_size,PROT_READ,MAP_PRIVATE,prompt_fd,0);
    if(emb==MAP_FAILED||prompt==MAP_FAILED)return 2;
    for(int i=0;i<prompt_count;i++)if(prompt[i]<0||prompt[i]>=vocab)return 2;
    const char *feedback_path=getenv("OPI_FEEDBACK_FILE");
    int32_t *feedback=NULL;
    if(feedback_path){
        feedback=calloc(max_new,sizeof(*feedback));
        FILE *f=fopen(feedback_path,"rb");
        if(!feedback||!f)return 2;
        if(fread(feedback,sizeof(*feedback),max_new,f)!=(size_t)max_new||fgetc(f)!=EOF){
            fprintf(stderr,"fixed feedback length mismatch\n");return 2;
        }
        fclose(f);
        for(int i=0;i<max_new;i++)if(feedback[i]<0||feedback[i]>=vocab)return 2;
    }
    int performance=getenv("OPI_PERFORMANCE")!=NULL;
    int fast_greedy=getenv("OPI_FAST_GREEDY")!=NULL;
    int grouped_head=getenv("OPI_GROUP_HEAD")!=NULL;
    if(performance&&(getenv("OPI_AUDIT_OUTPUTS")||getenv("OPI_CAPTURE_PREFIX")||getenv("OPI_PREFILL_POISON_OUTPUT")))return 2;
    if(getenv("OPI_AUDIT_OUTPUTS")){
        if(performance)return 2;
        audit_file=fopen(getenv("OPI_AUDIT_OUTPUTS"),"w");if(!audit_file)return 2;
    }
    int max_start_temp=getenv("OPI_MAX_START_TEMP_MC")?atoi(getenv("OPI_MAX_START_TEMP_MC")):55000;
    if(max_start_temp<35000||max_start_temp>70000)return 2;
    int max_response_temp=getenv("OPI_MAX_RESPONSE_TEMP_MC")?atoi(getenv("OPI_MAX_RESPONSE_TEMP_MC")):0;
    if(max_response_temp&&(max_response_temp<35000||max_response_temp>70000))return 2;
    int thermal_low=getenv("OPI_THERMAL_LOW_MC")?atoi(getenv("OPI_THERMAL_LOW_MC")):0;
    int thermal_high=getenv("OPI_THERMAL_HIGH_MC")?atoi(getenv("OPI_THERMAL_HIGH_MC")):0;
    if((thermal_low||thermal_high)&&(thermal_low<35000||thermal_high>80000||thermal_high-thermal_low<5000))return 2;
    int group_size=getenv("OPI_DECODER_GROUP")?atoi(getenv("OPI_DECODER_GROUP")):0;
    if(group_size<0||group_size>blocks)return 2;
    FILE *token_file=fopen(argv[8],"w"),*logits_file=performance?NULL:fopen(argv[9],"wb");
    if(!token_file||(!performance&&!logits_file))return 2;
    const char *trace_path=getenv("OPI_FULL_TRACE");
    if(trace_path&&prefill_tokens){fprintf(stderr,"batch prefill trace unavailable\n");return 2;}
    if(trace_path&&FUSED_LAYERS!=1){fprintf(stderr,"per-layer hidden trace unavailable in fused graph\n");return 2;}
    if(performance&&trace_path){fprintf(stderr,"performance and trace are exclusive\n");return 2;}
    FILE *trace_file=trace_path?fopen(trace_path,"wb"):NULL;
    if(trace_path&&!trace_file)return 2;
    Layer *decoder=calloc(blocks,sizeof(*decoder)),head[HEAD_SHARDS];
    memset(head,0,sizeof(head));
    float *x=calloc(DIM,sizeof(float)),*logits=calloc(vocab,sizeof(float));
    int *generated=calloc(max_new,sizeof(int));double *latencies=calloc(max_new,sizeof(double));
    double *layer_seconds=calloc(count,sizeof(double));
    double *decode_block_seconds=calloc(blocks,sizeof(double));
    if(!decoder||!x||!logits||!generated||!latencies||!layer_seconds||!decode_block_seconds)return 2;
    if(performance){
        double deadline=now()+300;
        while(temperature()>max_start_temp){
            if(now()>deadline){fprintf(stderr,"cooldown timeout\n");return 2;}
            struct timespec delay={1,0};nanosleep(&delay,NULL);
        }
        if(temperature()<0){fprintf(stderr,"temperature unavailable\n");return 2;}
    }
    int temp_before=temperature();double load_start=now();check(vip_init(),"init full model");
    unsigned ins[TENSOR_MAX_INPUTS]={DIM},outs[TENSOR_MAX_OUTPUTS]={DIM};
    for(int j=0;j<FUSED_LAYERS;j++){
        ins[IN_K(j)]=ins[IN_V(j)]=KV*capacity*HD;ins[IN_MASK(j)]=capacity+1;
        ins[IN_COS(j)]=ins[IN_SIN(j)]=HD;outs[OUT_K(j)]=outs[OUT_V(j)]=KV*HD;
    }
    for(int i=0;i<blocks;i++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],i*FUSED_LAYERS);
        load_net(&decoder[i],path,ni,no,ins,outs);
        for(int k=0;k<FUSED_LAYERS;k++){
            if(KEY_CACHE_TRANSPOSED){
                Tensor *key=&decoder[i].in[IN_K(k)];
                if(key->p.num_of_dims!=4||key->p.sizes[0]!=(unsigned)capacity||
                   key->p.sizes[1]!=HD||key->p.sizes[2]!=KV||key->p.sizes[3]!=1){
                    fprintf(stderr,"transposed K shape mismatch block=%d layer=%d\n",i,k);return 2;
                }
            }
            for(int j=0;j<capacity;j++)put(&decoder[i].in[IN_MASK(k)],j,-10000);
            put(&decoder[i].in[IN_MASK(k)],capacity,0);
        }
    }
    /* A group batches submissions, not graph operators. Hidden stays in the
     * device buffer; KV for each layer is updated after the whole stack. */
    vip_group groups[80]={0};int group_count=group_size?(blocks+group_size-1)/group_size:0;
    if(group_size){
        for(int i=1;i<blocks;i++){
            Tensor *src=&decoder[i-1].out[0],*dst=&decoder[i].in[0];
            if(src->p.data_format!=VIP_BUFFER_FORMAT_FP16||dst->p.data_format!=VIP_BUFFER_FORMAT_FP16||
               src->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE||dst->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE||
               src->p.num_of_dims!=dst->p.num_of_dims||src->count!=dst->count||
               memcmp(src->p.sizes,dst->p.sizes,src->p.num_of_dims*sizeof(src->p.sizes[0]))){
                fprintf(stderr,"group hidden contract mismatch at %d\n",i);return 2;
            }
            check(vip_set_input(decoder[i].net,0,src->b),"link decoder hidden");
        }
        for(int g=0;g<group_count;g++){
            unsigned n=(unsigned)(blocks-g*group_size<group_size?blocks-g*group_size:group_size),timeout=30000;
            check(vip_create_group(n,&groups[g]),"create decoder group");
            check(vip_set_group(groups[g],VIP_GROUP_PROP_SET_TIME_OUT,&timeout),"group timeout");
            for(unsigned j=0;j<n;j++)check(vip_add_network(groups[g],decoder[g*group_size+j].net),"add decoder group");
        }
    }
    int head_chunk=HEAD_SHARDS==1?vocab:HEAD_CHUNK;
    if(head_chunk<1||(vocab+head_chunk-1)/head_chunk!=HEAD_SHARDS)return 2;
    for(int s=0;s<HEAD_SHARDS;s++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],count+s);
        unsigned width=(unsigned)(vocab-s*head_chunk<head_chunk?vocab-s*head_chunk:head_chunk);
        unsigned head_in[]={DIM},head_out[]={width};load_net(&head[s],path,1,1,head_in,head_out);
    }
    vip_group head_group=0;
    if(grouped_head){
        unsigned timeout=30000;
        check(vip_create_group(HEAD_SHARDS,&head_group),"create output-head group");
        check(vip_set_group(head_group,VIP_GROUP_PROP_SET_TIME_OUT,&timeout),"head group timeout");
        for(int s=0;s<HEAD_SHARDS;s++)check(vip_add_network(head_group,head[s].net),"add output-head shard");
    }
    BatchPrefill batch=load_prefill(argv[1],prefill_tokens,blocks,capacity,decoder);
    double load_seconds=now()-load_start;long long allocated=npu_bytes();
    long long npu_freq_before=read_frequency("/sys/class/devfreq/3600000.npu/cur_freq");
    long long cpu_freq_before=read_frequency("/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq");
    double start=now(),cpu_start=cpu_time(),run_seconds=0,first_token=0,previous_token=start;
    double response_start_monotonic=start;
    double head_seconds=0,decode_decoder_seconds=0,decode_head_seconds=0;
    double preparation_seconds=0,preparation_api_seconds=0,preparation_cpu_seconds=0,prefill_api_seconds=0;
    int batched_prompt_tokens=0,prepared_batch_tokens=0;
    int response_temp=temperature();double response_cooldown_seconds=0;
    int n_generated=0,token=0;
    for(int pos=0;pos<prompt_count+max_new-1;pos++) {
        if(cached_prefix&&pos==cached_prefix){
            double stamp=now(),cpu_stamp=cpu_time();
            preparation_seconds=stamp-start;preparation_cpu_seconds=cpu_stamp-cpu_start;
            preparation_api_seconds=run_seconds;prepared_batch_tokens=batched_prompt_tokens;
            if(thermal_low){
                double cool_start=now();char low[16],high[16];
                snprintf(low,sizeof(low),"%d",thermal_low);snprintf(high,sizeof(high),"%d",thermal_high);
                for(int attempt=0;attempt<5;attempt++){
                    if(now()-cool_start>300){fprintf(stderr,"thermal conditioning deadline\n");return 2;}
                    pid_t child=fork();if(child<0)return 2;
                    if(!child){
                        if(prctl(PR_SET_PDEATHSIG,SIGTERM)||getppid()==1)_exit(127);
                        execlp("python3","python3","thermal_condition.py","--min",low,"--max",high,(char*)NULL);_exit(127);
                    }
                    int status=0;if(waitpid(child,&status,0)!=child||!WIFEXITED(status)||WEXITSTATUS(status))return 2;
                    int actual=temperature();
                    if(actual>=thermal_low&&actual<=thermal_high)break;
                    if(attempt==4){fprintf(stderr,"thermal band changed before response\n");return 2;}
                }
                response_cooldown_seconds=now()-cool_start;
                stamp=now();cpu_stamp=cpu_time();
            }else if(max_response_temp){
                double cool_start=now();
                while(temperature()>max_response_temp){
                    if(now()-cool_start>300){fprintf(stderr,"response cooldown timeout\n");return 2;}
                    struct timespec delay={1,0};nanosleep(&delay,NULL);
                }
                response_cooldown_seconds=now()-cool_start;
                stamp=now();cpu_stamp=cpu_time();
            }
            response_temp=temperature();
            response_start_monotonic=stamp;
            run_seconds=0;prefill_api_seconds=0;start=previous_token=stamp;cpu_start=cpu_stamp;
        }
        int boundary=pos<cached_prefix?cached_prefix:prompt_count;
        if(prefill_tokens&&pos>=scalar_prefix&&pos+prefill_tokens<=boundary){
            double prior=run_seconds;
            run_prefill(&batch,decoder,emb,prompt,pos,x,&run_seconds);
            prefill_api_seconds+=run_seconds-prior;batched_prompt_tokens+=prefill_tokens;
            pos+=prefill_tokens-1;
        }else{
        token=pos<prompt_count?prompt[pos]:(feedback?feedback[n_generated-1]:generated[n_generated-1]);
        memcpy(x,emb+(size_t)token*DIM,DIM*sizeof(float));
        float rope_cos[HD],rope_sin[HD];
        for(int j=0;j<HD;j++){
            float a=pos/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
            rope_cos[j]=cosf(a);rope_sin[j]=sinf(a);
        }
        if(group_size){
            for(int j=0;j<DIM;j++)put(&decoder[0].in[0],j,x[j]);
            for(int i=0;i<blocks;i++){
                Layer *l=&decoder[i];
                for(int k=0;k<FUSED_LAYERS;k++){
                    if(pos)put(&l->in[IN_MASK(k)],pos-1,0);
                    for(int j=0;j<HD;j++){
                        put(&l->in[IN_COS(k)],j,rope_cos[j]);put(&l->in[IN_SIN(k)],j,rope_sin[j]);
                    }
                }
                /* Never flush linked hidden from the CPU: it is produced by NPU. */
                for(unsigned j=i?1:0;j<ni;j++)check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush grouped input");
            }
            double prior=run_seconds;
            for(int g=0;g<group_count;g++){
                unsigned n=(unsigned)(blocks-g*group_size<group_size?blocks-g*group_size:group_size);
                double stamp=now();check(vip_run_group(groups[g],n),"run decoder group");run_seconds+=now()-stamp;
            }
            if(pos>=prompt_count)decode_decoder_seconds+=run_seconds-prior;
        }
        for(int i=0;i<blocks;i++) {
            Layer *l=&decoder[i];
            if(!group_size){
                for(int j=0;j<DIM;j++)put(&l->in[0],j,x[j]);
                for(int k=0;k<FUSED_LAYERS;k++){
                    if(pos)put(&l->in[IN_MASK(k)],pos-1,0);
                    for(int j=0;j<HD;j++){
                        put(&l->in[IN_COS(k)],j,rope_cos[j]);put(&l->in[IN_SIN(k)],j,rope_sin[j]);
                    }
                }
                double prior=run_seconds;execute(l,ni,no,&run_seconds);
                layer_seconds[i]+=run_seconds-prior;
                if(pos>=prompt_count){
                    decode_decoder_seconds+=run_seconds-prior;decode_block_seconds[i]+=run_seconds-prior;
                }
            }else{
                for(unsigned j=(i==blocks-1||trace_file)?0:1;j<no;j++)check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read grouped output");
            }
            audit_outputs(l,no,"scalar",pos,i);
            for(int j=0;(!group_size||i==blocks-1||trace_file)&&j<DIM;j++){
                x[j]=get(&l->out[0],j);
                if(!isfinite(x[j])){fprintf(stderr,"NONFINITE position=%d layer=%d component=%d value=%g\n",pos,i,j,x[j]);return 3;}
            }
            for(int k=0;k<FUSED_LAYERS;k++){
                append_key(&l->in[IN_K(k)],&l->out[OUT_K(k)],capacity,pos,1);
                for(int h=0;h<KV;h++)copy_values(&l->in[IN_V(k)],(h*capacity+pos)*HD,&l->out[OUT_V(k)],h*HD,HD);
            }
            if(trace_file&&fwrite(x,sizeof(float),DIM,trace_file)!=(size_t)DIM)return 2;
        }
        }
        if(pos<prompt_count-1)continue;
        int fast_best=-1;float fast_value=-INFINITY;
        if(grouped_head){
            /* Preserve each shard's calibrated input scale and every output.
             * Only submission/synchronization is grouped, not vocabulary work. */
            for(int s=0;s<HEAD_SHARDS;s++){
                for(int j=0;j<DIM;j++)put(&head[s].in[0],j,x[j]);
                check(vip_flush_buffer(head[s].in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush head group input");
            }
            double stamp=now();check(vip_run_group(head_group,HEAD_SHARDS),"run output-head group");
            double elapsed=now()-stamp;run_seconds+=elapsed;head_seconds+=elapsed;
            if(pos>=prompt_count)decode_head_seconds+=elapsed;
            for(int s=0;s<HEAD_SHARDS;s++)check(vip_flush_buffer(head[s].out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read head group output");
        }
        for(int s=0;s<HEAD_SHARDS;s++){
            if(!grouped_head){
                for(int j=0;j<DIM;j++)put(&head[s].in[0],j,x[j]);
                double prior=run_seconds;execute(&head[s],1,1,&run_seconds);
                head_seconds+=run_seconds-prior;
                if(pos>=prompt_count)decode_head_seconds+=run_seconds-prior;
            }
            Tensor *output=&head[s].out[0];
            if(fast_greedy){
                unsigned best=0;
                if(output->p.data_format==VIP_BUFFER_FORMAT_INT16&&
                   output->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT&&
                   output->p.quant_data.dfp.fixed_point_pos>=-110&&output->p.quant_data.dfp.fixed_point_pos<=110){
                    best=greedy_i16((const int16_t*)output->data,output->count);
                }else{
                    float value=-INFINITY;
                    for(unsigned j=0;j<output->count;j++){
                        float current=get(output,j);
                        if(!isfinite(current)){fprintf(stderr,"nonfinite greedy input\n");return 3;}
                        if(current>value){value=current;best=j;}
                    }
                }
                float value=get(output,best);
                if(!isfinite(value)){fprintf(stderr,"nonfinite greedy maximum\n");return 3;}
                /* Each shard may have a different positive DFP scale. Compare
                 * dequantized maxima, preserving earliest global-index ties. */
                if(value>fast_value){fast_value=value;fast_best=s*head_chunk+(int)best;}
            }
            if(!fast_greedy||logits_file)
                for(unsigned j=0;j<output->count;j++)logits[s*head_chunk+j]=get(output,j);
        }
        int best=fast_greedy?fast_best:0;
        if(!fast_greedy||logits_file){
            int reference=0;
            for(int j=0;j<vocab;j++){
                if(!isfinite(logits[j])){fprintf(stderr,"NONFINITE position=%d LM_HEAD token=%d value=%g\n",pos,j,logits[j]);return 3;}
                if(logits[j]>logits[reference])reference=j;
            }
            if(fast_greedy&&reference!=best){fprintf(stderr,"greedy verification mismatch\n");return 3;}
            best=reference;
        }
        if(logits_file&&fwrite(logits,sizeof(float),vocab,logits_file)!=(size_t)vocab)return 2;
        double stamp=now();latencies[n_generated]=stamp-previous_token;previous_token=stamp;
        generated[n_generated++]=best;if(n_generated==1)first_token=stamp-start;
        if(!performance)fprintf(stderr,"generated[%d]=%d elapsed=%.3f\n",n_generated-1,best,stamp-start);
        if(!feedback&&best==eos)break;
    }
    double total=now()-start,cpu=cpu_time()-cpu_start;struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    long rss_after_inference=current_rss_kib();
    long long npu_freq_after=read_frequency("/sys/class/devfreq/3600000.npu/cur_freq");
    long long cpu_freq_after=read_frequency("/sys/devices/system/cpu/cpu6/cpufreq/scaling_cur_freq");
    for(int g=0;g<group_count;g++)check(vip_destroy_group(groups[g]),"destroy decoder group");
    if(grouped_head)check(vip_destroy_group(head_group),"destroy output-head group");
    free_prefill(&batch);
    for(int i=0;i<blocks;i++){free_net(&decoder[i],ni,no);}
    for(int s=0;s<HEAD_SHARDS;s++)free_net(&head[s],1,1);
    check(vip_destroy(),"release full model");
    fprintf(token_file,"{\"decoder_layers\":%d,\"head_on_npu\":true,\"capacity\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%d,\"load_seconds\":%.6f,\"ttft_seconds\":%.6f,\"total_seconds\":%.6f,\"decode_tokens_per_second\":%.6f,\"npu_api_seconds\":%.6f,\"cpu_seconds\":%.6f,\"cpu_percent_one_core\":%.3f,\"max_rss_kib\":%ld,\"driver_allocated_bytes\":%lld,\"driver_after_release_bytes\":%lld,\"temperature_before_mC\":%d,\"temperature_after_mC\":%d,\"tokens\":[",count,capacity,prompt_count,n_generated,load_seconds,first_token,total,n_generated>1?(n_generated-1)/(total-first_token):0,run_seconds,cpu,100*cpu/total,usage.ru_maxrss,allocated,npu_bytes(),temp_before,temperature());
    for(int i=0;i<n_generated;i++)fprintf(token_file,"%s%d",i?",":"",generated[i]);
    fprintf(token_file,"],\"token_latencies_seconds\":[");
    for(int i=0;i<n_generated;i++){fprintf(token_file,"%s%.6f",i?",":"",latencies[i]);}
    fprintf(token_file,"],\"head_api_seconds\":%.6f,\"decode_decoder_api_seconds\":%.6f,\"decode_head_api_seconds\":%.6f,\"layer_api_seconds\":[",head_seconds,decode_decoder_seconds,decode_head_seconds);
    for(int i=0;i<count;i++){
        if(group_size||FUSED_LAYERS!=1)fprintf(token_file,"%snull",i?",":"");
        else fprintf(token_file,"%s%.6f",i?",":"",layer_seconds[i]);
    }
    fprintf(token_file,"],\"block_api_seconds\":[");
    for(int i=0;i<blocks;i++){
        if(group_size)fprintf(token_file,"%snull",i?",":"");
        else fprintf(token_file,"%s%.6f",i?",":"",layer_seconds[i]);
    }
    fprintf(token_file,"],\"decode_block_api_seconds\":[");
    for(int i=0;i<blocks;i++){
        if(group_size)fprintf(token_file,"%snull",i?",":"");
        else fprintf(token_file,"%s%.6f",i?",":"",decode_block_seconds[i]);
    }
    fprintf(token_file,"],\"fixed_feedback\":%s,\"feedback_tokens\":[",feedback?"true":"false");
    if(feedback)for(int i=0;i<max_new;i++)fprintf(token_file,"%s%d",i?",":"",feedback[i]);
    fprintf(token_file,"],\"prefill_batch_size\":%d,\"cached_prefix_tokens\":%d,\"new_prompt_tokens\":%d,\"batched_prompt_tokens\":%d,\"prepared_batch_tokens\":%d,\"preparation_seconds\":%.6f,\"preparation_api_seconds\":%.6f,\"preparation_cpu_seconds\":%.6f,\"prefill_api_seconds\":%.6f",prefill_tokens,cached_prefix,prompt_count-cached_prefix,batched_prompt_tokens,prepared_batch_tokens,preparation_seconds,preparation_api_seconds,preparation_cpu_seconds,prefill_api_seconds);
    fprintf(token_file,",\"prefill_host_hidden\":%s,\"host_output_buffers\":%s",getenv("OPI_PREFILL_HOST_HIDDEN")?"true":"false",getenv("OPI_HOST_OUTPUT_BUFFERS")?"true":"false");
    fprintf(token_file,",\"prefill_clear_output\":%s",getenv("OPI_PREFILL_CLEAR_OUTPUT")?"true":"false");
    fprintf(token_file,",\"prefill_scalar_block\":%d",prefill_tokens?batch.scalar_block:-1);
    fprintf(token_file,",\"key_cache_transposed\":%s,\"key_cache_layout_revision\":2",KEY_CACHE_TRANSPOSED?"true":"false");
    fprintf(token_file,",\"grouped_head\":%s",grouped_head?"true":"false");
    fprintf(token_file,",\"max_response_temperature_mC\":%d,\"response_start_temperature_mC\":%d,\"response_cooldown_seconds\":%.6f",max_response_temp,response_temp,response_cooldown_seconds);
    fprintf(token_file,",\"thermal_low_mC\":%d,\"thermal_high_mC\":%d",thermal_low,thermal_high);
    fprintf(token_file,",\"response_start_monotonic_seconds\":%.6f",response_start_monotonic);
    fprintf(token_file,",\"prefill_rebind_outputs\":%s,\"poison_prefill_output\":%s,\"prefill_flush_output\":%s,\"capture_prefill\":%s",getenv("OPI_PREFILL_REBIND_OUTPUTS")?"true":"false",getenv("OPI_PREFILL_POISON_OUTPUT")?"true":"false",getenv("OPI_PREFILL_FLUSH_OUTPUT")?"true":"false",getenv("OPI_CAPTURE_PREFIX")?"true":"false");
    fprintf(token_file,",\"prefill_scalar_prefix\":%d,\"performance_mode\":%s,\"host_revision\":10,\"decoder_group_size\":%d,\"fusion_width\":%d,\"max_start_temperature_mC\":%d,\"npu_frequency_before_hz\":%lld,\"npu_frequency_after_hz\":%lld,\"cpu6_frequency_before_khz\":%lld,\"cpu6_frequency_after_khz\":%lld,\"fast_greedy\":%s,\"rss_after_inference_kib\":%ld,\"init_buffer_flush\":%s,\"audit_outputs\":%s}\n",scalar_prefix,performance?"true":"false",group_size,FUSED_LAYERS,max_start_temp,npu_freq_before,npu_freq_after,cpu_freq_before,cpu_freq_after,fast_greedy?"true":"false",rss_after_inference,getenv("OPI_INIT_BUFFER_FLUSH")?"true":"false",audit_file?"true":"false");
    if(trace_file)fclose(trace_file);
    if(audit_file){fclose(audit_file);audit_file=NULL;}
    free(decode_block_seconds);
    fclose(token_file);if(logits_file)fclose(logits_file);munmap((void*)emb,emb_stat.st_size);munmap((void*)prompt,prompt_stat.st_size);
    close(emb_fd);close(prompt_fd);free(decoder);free(x);free(logits);free(generated);free(feedback);free(latencies);free(layer_seconds);return 0;
}
