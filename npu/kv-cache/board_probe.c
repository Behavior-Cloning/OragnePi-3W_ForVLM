/* Incremental two-layer decoder probe. Random model, not a chat benchmark.
 * cc -O2 -Wall -Wextra board_probe.c -lNBGlinker -lVIPhal -lm -o board_probe
 */
#define _POSIX_C_SOURCE 200809L
#include <vip_lite.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

#ifndef MODEL_DIM
#define MODEL_DIM 64
#define MODEL_KV 2
#define MODEL_HD 16
#endif
#ifndef ROPE_BASE
#define ROPE_BASE 10000.0f
#endif
enum { LAYERS=2, DIM=MODEL_DIM, KV=MODEL_KV, HD=MODEL_HD, NI=6, NO=3 };
typedef struct { vip_buffer b; vip_buffer_create_params_t p; void *data; unsigned count; } Tensor;
typedef struct { vip_network net; Tensor in[NI], out[NO]; } Layer;
static Layer layers[LAYERS];
static void check(vip_status_e s, const char *where) {
    if(s != VIP_SUCCESS) { fprintf(stderr,"%s status=%d\n",where,s); exit(2); }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static int temperature(void) {
    int value=-1;FILE *f=fopen("/sys/class/thermal/thermal_zone0/temp","r");
    if(f){if(fscanf(f,"%d",&value)!=1)value=-1;fclose(f);}return value;
}
static double cpu_time(void) {
    struct rusage r;getrusage(RUSAGE_SELF,&r);
    return r.ru_utime.tv_sec+r.ru_utime.tv_usec/1e6+r.ru_stime.tv_sec+r.ru_stime.tv_usec/1e6;
}
static long long npu_bytes(void) {
    FILE *f=fopen("/sys/kernel/debug/viplite/mem_profile","r");
    char line[256];unsigned long long current,peak;long long result=-1;
    if(!f)return -1;
    while(fgets(line,sizeof(line),f))
        if(sscanf(line,"| %llx | %llx",&current,&peak)==2)result=(long long)current;
    fclose(f);return result;
}
static float get(Tensor *t,unsigned i) {
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP16) return ((const __fp16 *)t->data)[i];
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP32) return ((const float *)t->data)[i];
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT8 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return ldexpf(((const int8_t *)t->data)[i],-t->p.quant_data.dfp.fixed_point_pos);
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT16 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return ldexpf(((const int16_t *)t->data)[i],-t->p.quant_data.dfp.fixed_point_pos);
    fprintf(stderr,"unsupported dtype/quant %d/%d\n",t->p.data_format,t->p.quant_format); exit(2);
}
static void put(Tensor *t,unsigned i,float x) {
    if(i>=t->count) {fprintf(stderr,"tensor index overflow\n");exit(2);}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP16) {((__fp16 *)t->data)[i]=x;return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP32) {((float *)t->data)[i]=x;return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT8 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int8_t *)t->data)[i]=(int8_t)fmaxf(-128,fminf(127,y));return;
    }
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT16 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int16_t *)t->data)[i]=(int16_t)fmaxf(-32768,fminf(32767,y));return;
    }
    fprintf(stderr,"unsupported input dtype/quant\n");exit(2);
}
static void tensor_init(Layer *l,Tensor *t,unsigned index,int output,unsigned expected) {
    vip_status_e (*query)(vip_network,vip_uint32_t,vip_enum,void*)=output?vip_query_output:vip_query_input;
    memset(t,0,sizeof(*t));
    check(query(l->net,index,VIP_BUFFER_PROP_NUM_OF_DIMENSION,&t->p.num_of_dims),"query dims");
    if(t->p.num_of_dims>6)exit(2);
    check(query(l->net,index,VIP_BUFFER_PROP_SIZES_OF_DIMENSION,t->p.sizes),"query shape");
    check(query(l->net,index,VIP_BUFFER_PROP_DATA_FORMAT,&t->p.data_format),"query dtype");
    check(query(l->net,index,VIP_BUFFER_PROP_QUANT_FORMAT,&t->p.quant_format),"query quant");
    if(t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        uint8_t fp=0;
        check(query(l->net,index,VIP_BUFFER_PROP_FIXED_POINT_POS,&fp),"query fl");
        t->p.quant_data.dfp.fixed_point_pos=(int8_t)fp;
    }
    t->count=1;
    for(unsigned j=0;j<t->p.num_of_dims;j++)t->count*=t->p.sizes[j];
    if(t->count!=expected){fprintf(stderr,"unexpected IO size %u vs %u\n",t->count,expected);exit(2);}
    char name[64]={0};check(query(l->net,index,VIP_BUFFER_PROP_NAME,name),"query name");
    fprintf(stderr,"%s[%u] %s size=%u dtype=%d quant=%d fl=%d shape=",output?"out":"in",index,name,t->count,t->p.data_format,t->p.quant_format,t->p.quant_data.dfp.fixed_point_pos);
    for(unsigned j=0;j<t->p.num_of_dims;j++)fprintf(stderr,"%u,",t->p.sizes[j]);
    fprintf(stderr,"\n");
    check(vip_create_buffer(&t->p,sizeof(t->p),&t->b),"create buffer");
    t->data=vip_map_buffer(t->b);if(!t->data)exit(2);
    for(unsigned j=0;j<t->count;j++)put(t,j,0);
    check(output?vip_set_output(l->net,index,t->b):vip_set_input(l->net,index,t->b),"bind buffer");
}
static void read_file(const char *path,float *data,size_t n) {
    FILE *f=fopen(path,"rb");if(!f){perror(path);exit(2);}
    if(fread(data,sizeof(float),n,f)!=n || fgetc(f)!=EOF){fprintf(stderr,"bad fixture length\n");exit(2);}fclose(f);
}
int main(int argc,char **argv) {
    if(argc!=8){fprintf(stderr,"usage: board_probe layer0.nb layer1.nb capacity tokens hidden.f32 reference.f32 output.f32\n");return 2;}
    int capacity=atoi(argv[3]),tokens=atoi(argv[4]);
    if(capacity<1||capacity>8192||tokens<1||tokens>capacity)return 2;
    float *hidden=calloc((size_t)tokens*DIM,sizeof(float)),*ref=calloc((size_t)tokens*DIM,sizeof(float));
    if(!hidden||!ref)return 2;
    read_file(argv[5],hidden,(size_t)tokens*DIM);read_file(argv[6],ref,(size_t)tokens*DIM);
    FILE *out_file=fopen(argv[7],"wb");if(!out_file)return 2;
    int temp_before=temperature();double load_start=now();check(vip_init(),"vip_init");
    uint32_t cid=0;check(vip_query_hardware(VIP_QUERY_HW_PROP_CID,sizeof(cid),&cid),"chip id");
    unsigned ins[]={DIM,KV*capacity*HD,KV*capacity*HD,capacity+1,HD,HD},outs[]={DIM,KV*HD,KV*HD};
    for(int n=0;n<LAYERS;n++) {
        Layer *l=&layers[n];check(vip_create_network(argv[1+n],0,VIP_CREATE_NETWORK_FROM_FILE,&l->net),"create network");
        uint32_t ni=0,no=0,timeout_ms=10000;
        check(vip_query_network(l->net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"input count");
        check(vip_query_network(l->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"output count");
        if(ni!=NI||no!=NO)return 2;
        check(vip_set_network(l->net,VIP_NETWORK_PROP_SET_TIME_OUT,&timeout_ms),"timeout");
        check(vip_prepare_network(l->net),"prepare");
        for(unsigned j=0;j<NI;j++)tensor_init(l,&l->in[j],j,0,ins[j]);
        for(unsigned j=0;j<NO;j++)tensor_init(l,&l->out[j],j,1,outs[j]);
        if(getenv("OPI_KV_POISON")) {
            for(unsigned j=0;j<l->in[1].count;j++) {
                put(&l->in[1],j,7);put(&l->in[2],j,-11);
            }
        }
    }
    double load_seconds=now()-load_start,start,run_seconds=0,max_error=0,err2=0,ref2=0,min_cos=1;
    long long driver_bytes=npu_bytes();
    uint64_t profile_us=0;double cpu_start=cpu_time(),quarters[4]={0};int quarter_counts[4]={0};
    start=now();
    for(int pos=0;pos<tokens;pos++) {
        double step_start=now();
        float x[DIM];memcpy(x,hidden+pos*DIM,sizeof(x));
        for(int n=0;n<LAYERS;n++) {
            Layer *l=&layers[n];
            for(int j=0;j<DIM;j++)put(&l->in[0],j,x[j]);
            for(int j=0;j<=capacity;j++)put(&l->in[3],j,(j<pos||j==capacity)?0:-10000);
            for(int j=0;j<HD;j++) {
                float a=pos/powf(ROPE_BASE,(j%(HD/2))/(float)(HD/2));
                put(&l->in[4],j,cosf(a));put(&l->in[5],j,sinf(a));
            }
            for(int j=0;j<NI;j++)check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");
            double t=now();check(vip_run_network(l->net),"run");run_seconds+=now()-t;
            vip_inference_profile_t profile={0};check(vip_query_network(l->net,VIP_NETWORK_PROP_PROFILING,&profile),"profile");profile_us+=profile.inference_time;
            for(int j=0;j<NO;j++)check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");
            const char *trace=getenv("OPI_TRACE_PREFIX");
            if(trace && pos==0){
                char path[1024];snprintf(path,sizeof(path),"%s-layer%d.f32",trace,n);
                FILE *f=fopen(path,"wb");if(!f)return 2;
                for(unsigned o=0;o<NO;o++)for(unsigned j=0;j<l->out[o].count;j++){
                    float value=get(&l->out[o],j);if(fwrite(&value,sizeof(value),1,f)!=1)return 2;
                }
                fclose(f);
            }
            for(int j=0;j<DIM;j++)x[j]=get(&l->out[0],j);
            for(int head=0;head<KV;head++)for(int j=0;j<HD;j++) {
                unsigned dst=(head*capacity+pos)*HD+j,src=head*HD+j;
                put(&l->in[1],dst,get(&l->out[1],src));put(&l->in[2],dst,get(&l->out[2],src));
            }
        }
        double dot=0,xx=0,rr=0;
        for(int j=0;j<DIM;j++) {
            double r=ref[pos*DIM+j],e=x[j]-r;max_error=fmax(max_error,fabs(e));err2+=e*e;ref2+=r*r;
            if(!isfinite(x[j])||!isfinite(r)){fprintf(stderr,"non-finite output/reference at %d/%d\n",pos,j);return 3;}
            dot+=x[j]*r;xx+=x[j]*x[j];rr+=r*r;
        }
        min_cos=fmin(min_cos,dot/sqrt(xx*rr+1e-30));
        if(fwrite(x,sizeof(float),DIM,out_file)!=DIM)return 2;
        int quarter=pos*4/tokens;quarters[quarter]+=now()-step_start;quarter_counts[quarter]++;
    }
    double total=now()-start,cpu_seconds=cpu_time()-cpu_start;fclose(out_file);
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);int temp_after=temperature();
    for(int n=0;n<LAYERS;n++) {
        for(int j=0;j<NI;j++){check(vip_unmap_buffer(layers[n].in[j].b),"unmap");check(vip_destroy_buffer(layers[n].in[j].b),"free input");}
        for(int j=0;j<NO;j++){check(vip_unmap_buffer(layers[n].out[j].b),"unmap");check(vip_destroy_buffer(layers[n].out[j].b),"free output");}
        check(vip_destroy_network(layers[n].net),"free net");
    }
    check(vip_destroy(),"destroy");free(hidden);free(ref);
    printf("{\"chip_id\":\"0x%x\",\"capacity\":%d,\"tokens\":%d,\"layers\":2,\"load_seconds\":%.6f,\"total_seconds\":%.6f,\"run_seconds\":%.6f,\"profile_us\":%llu,\"two_layer_steps_per_second\":%.3f,\"max_abs_error\":%.8f,\"relative_l2\":%.8f,\"min_cosine\":%.8f,\"process_cpu_seconds\":%.6f,\"process_cpu_percent_one_core\":%.2f,\"max_rss_kib\":%ld,\"temperature_before_mC\":%d,\"temperature_after_mC\":%d,\"quarter_step_ms\":[",cid,capacity,tokens,load_seconds,total,run_seconds,(unsigned long long)profile_us,tokens/total,max_error,sqrt(err2/(ref2+1e-30)),min_cos,cpu_seconds,100*cpu_seconds/total,usage.ru_maxrss,temp_before,temp_after);
    for(int i=0;i<4;i++)printf("%s%.6f",i?",":"",quarter_counts[i]?1000*quarters[i]/quarter_counts[i]:0);
    printf("],\"driver_allocated_bytes_after_load\":%lld,\"driver_allocated_bytes_after_release\":%lld}\n",driver_bytes,npu_bytes());
    return min_cos>=.99 && sqrt(err2/(ref2+1e-30))<=.05 ? 0 : 3;
}
