#define OPI_FULL_MODEL_ENTRY unused_language_main
#include "full_model.c"
#include <arm_neon.h>

static void half_neon(__fp16 *out,const int16_t *in,unsigned n,float scale){
 unsigned j=0;float32x4_t sc=vdupq_n_f32(scale);
 for(;j+8<=n;j+=8){
  int16x8_t x=vld1q_s16(in+j);
  float32x4_t lo=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(x))),sc);
  float32x4_t hi=vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(x))),sc);
  vst1q_f16(out+j,vcombine_f16(vcvt_f16_f32(lo),vcvt_f16_f32(hi)));
 }
 for(;j<n;j++)out[j]=(__fp16)((float)in[j]*scale);
}
static int fast_mode;
static void tower_copy(Tensor *dst,unsigned offset,Tensor *src,unsigned begin,unsigned count) {
 if(offset+count>dst->count||begin+count>src->count)exit(2);
 int fl=src->p.quant_data.dfp.fixed_point_pos;
 if(fast_mode&&(dst->p.data_format==VIP_BUFFER_FORMAT_FP32||dst->p.data_format==VIP_BUFFER_FORMAT_FP16)&&src->p.data_format==VIP_BUFFER_FORMAT_INT16&&src->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT&&fl>=-64&&fl<=64){
  const int16_t *in=(const int16_t*)src->data+begin;float *out=(float*)dst->data+offset;float scale=ldexpf(1.0f,-fl);
  if(dst->p.data_format==VIP_BUFFER_FORMAT_FP32){for(unsigned j=0;j<count;j++)out[j]=(float)in[j]*scale;}
  else {__fp16 *half=(__fp16*)dst->data+offset;if(fast_mode==2)half_neon(half,in,count,scale);else for(unsigned j=0;j<count;j++)half[j]=(__fp16)((float)in[j]*scale);}
 }else copy_values(dst,offset,src,begin,count);
}
static int selftest(void){
 int16_t *in=malloc(65536*2);__fp16 *out=malloc(65536*2);if(!in||!out)return 2;
 for(int n=0;n<65536;n++)in[n]=(int16_t)(n-32768);
 for(int fl=-64;fl<=64;fl++){
  float scale=ldexpf(1.0f,-fl);half_neon(out,in,65536,scale);
  for(int n=0;n<65536;n++){__fp16 ref=(__fp16)ldexpf((float)in[n],-fl);if(memcmp(&ref,out+n,2))return 3;}
  half_neon(out+1,in+1,65533,scale);
  for(int n=1;n<65534;n++){__fp16 ref=(__fp16)ldexpf((float)in[n],-fl);if(memcmp(&ref,out+n,2))return 4;}
 }
 free(in);free(out);puts("NEON exhaustive int16 scales plus unaligned tail bit parity passed");return 0;
}

int main(int argc,char **argv){
 if(argc==2&&!strcmp(argv[1],"--selftest"))return selftest();
 fast_mode=getenv("VIT_FAST_COPY")&&atoi(getenv("VIT_FAST_COPY"));
 if(argc!=4)return 2;
 const int count=257*1024;Layer nets[26]={0};float *inputs[2];char path[2048];
 for(int im=0;im<2;im++){inputs[im]=calloc(150528,4);if(!inputs[im])return 2;snprintf(path,sizeof(path),"%s/input%d.f32",argv[2],im+1);read_file(path,inputs[im],150528);}
 check(vip_init(),"init");double load_start=now();
 for(int k=0;k<26;k++){
  Layer *l=&nets[k];if(k==0)snprintf(path,sizeof(path),"%s/stem.nb",argv[1]);else if(k==25)snprintf(path,sizeof(path),"%s/projector.nb",argv[1]);else snprintf(path,sizeof(path),"%s/layer%02d.nb",argv[1],k-1);check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&l->net),"create");unsigned ni=0,no=0;
  check(vip_query_network(l->net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
  check(vip_prepare_network(l->net),"prepare");tensor_init(l,&l->in[0],0,0,k==0?150528:count);tensor_init(l,&l->out[0],0,1,k==25?57344:count);
 }
 double load_seconds=now()-load_start;long long bytes=npu_bytes();
 snprintf(path,sizeof(path),"%s/events.jsonl",argv[3]);FILE *events=fopen(path,"wx");if(!events)return 2;
 fprintf(events,"{\"event\":\"ready\",\"layers\":26,\"rows\":257,\"driver_bytes\":%lld,\"load_seconds\":%.9f}\n",bytes,load_seconds);fflush(events);
 for(int run=-1;run<6;run++){
  if(temperature()>=78000)return 3;
  int im=run<0?0:run%2;double start=now(),copy=0,api[26];
  double t=now();for(int j=0;j<150528;j++)put(&nets[0].in[0],j,inputs[im][j]);copy+=now()-t;
  for(int k=0;k<26;k++){
   Layer *l=&nets[k];t=now();check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");copy+=now()-t;
   t=now();check(vip_run_network(l->net),"run");api[k]=now()-t;
   t=now();check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");
   if(k<25)tower_copy(&nets[k+1].in[0],0,&l->out[0],0,count);
   copy+=now()-t;
  }
  float output[57344];t=now();
  for(int j=0;j<57344;j++){output[j]=get(&nets[25].out[0],j);if(!isfinite(output[j]))return 3;}
  copy+=now()-t;double elapsed=now()-start;
  snprintf(path,sizeof(path),"%s/run%d-output.f32",argv[3],run);FILE *f=fopen(path,"wb");if(!f)return 2;
  if(fwrite(output,4,57344,f)!=57344)return 2;fclose(f);
  fprintf(events,"{\"event\":\"result\",\"run\":%d,\"image\":%d,\"stack_seconds\":%.9f,\"host_copy_sync_seconds\":%.9f,\"temperature_mC\":%d,\"api_seconds\":[",run,im+1,elapsed,copy,temperature());
  for(int k=0;k<26;k++)fprintf(events,"%s%.9f",k?",":"",api[k]);fprintf(events,"]}\n");fflush(events);
 }
 for(int k=0;k<26;k++)free_net(&nets[k],1,1);check(vip_destroy(),"destroy");
 fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld}\n",npu_bytes());fclose(events);free(inputs[0]);free(inputs[1]);return 0;
}
