/* Full language stack and LM head on VIPLite; CPU embeds and samples.
 * Reuse the tested IO/quantization utilities from the incremental probe.
 * Compile with MODEL_DIM/MODEL_KV/MODEL_HD/ROPE_BASE from the model manifest.
 */
#define main two_layer_probe_example_main
#include "board_probe.c"
#undef main
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#ifndef HEAD_SHARDS
#define HEAD_SHARDS 1
#define HEAD_CHUNK 0
#endif

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
int main(int argc,char **argv) {
    if(argc!=11){fprintf(stderr,"usage: full_model networks layers capacity vocab embeddings.f32 prompt.i32 max_new tokens.json logits.f32 eos_id\n");return 2;}
    int count=atoi(argv[2]),capacity=atoi(argv[3]),vocab=atoi(argv[4]),max_new=atoi(argv[7]),eos=atoi(argv[10]);
    if(count<1||count>80||capacity<1||capacity>8192||vocab<1||max_new<1||max_new>capacity)return 2;
    int emb_fd=open(argv[5],O_RDONLY),prompt_fd=open(argv[6],O_RDONLY);struct stat emb_stat,prompt_stat;
    if(emb_fd<0||prompt_fd<0||fstat(emb_fd,&emb_stat)||fstat(prompt_fd,&prompt_stat))return 2;
    if((size_t)emb_stat.st_size!=(size_t)vocab*DIM*sizeof(float)||prompt_stat.st_size<4||prompt_stat.st_size%4)return 2;
    int prompt_count=prompt_stat.st_size/4;
    if(prompt_count+max_new-1>capacity){fprintf(stderr,"prompt plus output exceeds KV capacity\n");return 2;}
    const float *emb=mmap(NULL,emb_stat.st_size,PROT_READ,MAP_PRIVATE,emb_fd,0);
    const int32_t *prompt=mmap(NULL,prompt_stat.st_size,PROT_READ,MAP_PRIVATE,prompt_fd,0);
    if(emb==MAP_FAILED||prompt==MAP_FAILED)return 2;
    for(int i=0;i<prompt_count;i++)if(prompt[i]<0||prompt[i]>=vocab)return 2;
    FILE *token_file=fopen(argv[8],"w"),*logits_file=fopen(argv[9],"wb");if(!token_file||!logits_file)return 2;
    const char *trace_path=getenv("OPI_FULL_TRACE");
    FILE *trace_file=trace_path?fopen(trace_path,"wb"):NULL;
    if(trace_path&&!trace_file)return 2;
    Layer *decoder=calloc(count,sizeof(*decoder)),head[HEAD_SHARDS];
    memset(head,0,sizeof(head));
    float *x=calloc(DIM,sizeof(float)),*logits=calloc(vocab,sizeof(float));
    int *generated=calloc(max_new,sizeof(int));double *latencies=calloc(max_new,sizeof(double));
    double *layer_seconds=calloc(count,sizeof(double));
    if(!decoder||!x||!logits||!generated||!latencies||!layer_seconds)return 2;
    int temp_before=temperature();double load_start=now();check(vip_init(),"init full model");
    unsigned ins[]={DIM,KV*capacity*HD,KV*capacity*HD,capacity+1,HD,HD},outs[]={DIM,KV*HD,KV*HD};
    for(int i=0;i<count;i++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],i);
        load_net(&decoder[i],path,6,3,ins,outs);
    }
    int head_chunk=HEAD_SHARDS==1?vocab:HEAD_CHUNK;
    if(head_chunk<1||(vocab+head_chunk-1)/head_chunk!=HEAD_SHARDS)return 2;
    for(int s=0;s<HEAD_SHARDS;s++){
        char path[1024];snprintf(path,sizeof(path),"%s/layer%d.nb",argv[1],count+s);
        unsigned width=(unsigned)(vocab-s*head_chunk<head_chunk?vocab-s*head_chunk:head_chunk);
        unsigned head_in[]={DIM},head_out[]={width};load_net(&head[s],path,1,1,head_in,head_out);
    }
    double load_seconds=now()-load_start;long long allocated=npu_bytes();
    double start=now(),cpu_start=cpu_time(),run_seconds=0,first_token=0,previous_token=start;
    double head_seconds=0,decode_decoder_seconds=0,decode_head_seconds=0;
    int n_generated=0,token=0;
    for(int pos=0;pos<prompt_count+max_new-1;pos++) {
        token=pos<prompt_count?prompt[pos]:generated[n_generated-1];
        memcpy(x,emb+(size_t)token*DIM,DIM*sizeof(float));
        for(int i=0;i<count;i++) {
            Layer *l=&decoder[i];
            for(int j=0;j<DIM;j++)put(&l->in[0],j,x[j]);
            for(int j=0;j<=capacity;j++)put(&l->in[3],j,(j<pos||j==capacity)?0:-10000);
            for(int j=0;j<HD;j++){
                float a=pos/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
                put(&l->in[4],j,cosf(a));put(&l->in[5],j,sinf(a));
            }
            double prior=run_seconds;execute(l,6,3,&run_seconds);
            layer_seconds[i]+=run_seconds-prior;
            if(pos>=prompt_count)decode_decoder_seconds+=run_seconds-prior;
            for(int j=0;j<DIM;j++){
                x[j]=get(&l->out[0],j);
                if(!isfinite(x[j])){fprintf(stderr,"NONFINITE position=%d layer=%d component=%d value=%g\n",pos,i,j,x[j]);return 3;}
            }
            for(int h=0;h<KV;h++)for(int j=0;j<HD;j++){
                unsigned dst=(h*capacity+pos)*HD+j,src=h*HD+j;
                put(&l->in[1],dst,get(&l->out[1],src));put(&l->in[2],dst,get(&l->out[2],src));
            }
            if(trace_file&&fwrite(x,sizeof(float),DIM,trace_file)!=(size_t)DIM)return 2;
        }
        if(pos<prompt_count-1)continue;
        for(int s=0;s<HEAD_SHARDS;s++){
            for(int j=0;j<DIM;j++)put(&head[s].in[0],j,x[j]);
            double prior=run_seconds;execute(&head[s],1,1,&run_seconds);
            head_seconds+=run_seconds-prior;
            if(pos>=prompt_count)decode_head_seconds+=run_seconds-prior;
            for(unsigned j=0;j<head[s].out[0].count;j++)logits[s*head_chunk+j]=get(&head[s].out[0],j);
        }
        int best=0;
        for(int j=0;j<vocab;j++){
            if(!isfinite(logits[j])){fprintf(stderr,"NONFINITE position=%d LM_HEAD token=%d value=%g\n",pos,j,logits[j]);return 3;}
            if(logits[j]>logits[best])best=j;
        }
        if(fwrite(logits,sizeof(float),vocab,logits_file)!=(size_t)vocab)return 2;
        double stamp=now();latencies[n_generated]=stamp-previous_token;previous_token=stamp;
        generated[n_generated++]=best;if(n_generated==1)first_token=stamp-start;
        fprintf(stderr,"generated[%d]=%d elapsed=%.3f\n",n_generated-1,best,stamp-start);
        if(best==eos)break;
    }
    double total=now()-start,cpu=cpu_time()-cpu_start;struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    for(int i=0;i<count;i++){free_net(&decoder[i],6,3);}
    for(int s=0;s<HEAD_SHARDS;s++)free_net(&head[s],1,1);
    check(vip_destroy(),"release full model");
    fprintf(token_file,"{\"decoder_layers\":%d,\"head_on_npu\":true,\"capacity\":%d,\"prompt_tokens\":%d,\"generated_tokens\":%d,\"load_seconds\":%.6f,\"ttft_seconds\":%.6f,\"total_seconds\":%.6f,\"decode_tokens_per_second\":%.6f,\"npu_api_seconds\":%.6f,\"cpu_seconds\":%.6f,\"cpu_percent_one_core\":%.3f,\"max_rss_kib\":%ld,\"driver_allocated_bytes\":%lld,\"driver_after_release_bytes\":%lld,\"temperature_before_mC\":%d,\"temperature_after_mC\":%d,\"tokens\":[",count,capacity,prompt_count,n_generated,load_seconds,first_token,total,n_generated>1?(n_generated-1)/(total-first_token):0,run_seconds,cpu,100*cpu/total,usage.ru_maxrss,allocated,npu_bytes(),temp_before,temperature());
    for(int i=0;i<n_generated;i++)fprintf(token_file,"%s%d",i?",":"",generated[i]);
    fprintf(token_file,"],\"token_latencies_seconds\":[");
    for(int i=0;i<n_generated;i++){fprintf(token_file,"%s%.6f",i?",":"",latencies[i]);}
    fprintf(token_file,"],\"head_api_seconds\":%.6f,\"decode_decoder_api_seconds\":%.6f,\"decode_head_api_seconds\":%.6f,\"layer_api_seconds\":[",head_seconds,decode_decoder_seconds,decode_head_seconds);
    for(int i=0;i<count;i++)fprintf(token_file,"%s%.6f",i?",":"",layer_seconds[i]);
    fprintf(token_file,"]}\n");
    if(trace_file)fclose(trace_file);
    fclose(token_file);fclose(logits_file);munmap((void*)emb,emb_stat.st_size);munmap((void*)prompt,prompt_stat.st_size);
    close(emb_fd);close(prompt_fd);free(decoder);free(x);free(logits);free(generated);free(latencies);free(layer_seconds);return 0;
}
