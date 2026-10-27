#define OPI_FULL_MODEL_ENTRY unused_language_main
#include "full_model.c"

static int fast_mode;
static void tower_copy(Tensor *dst,unsigned offset,Tensor *src,unsigned begin,unsigned count) {
 if(offset+count>dst->count||begin+count>src->count)exit(2);
 int fl=src->p.quant_data.dfp.fixed_point_pos;
 if(fast_mode&&(dst->p.data_format==VIP_BUFFER_FORMAT_FP32||dst->p.data_format==VIP_BUFFER_FORMAT_FP16)&&src->p.data_format==VIP_BUFFER_FORMAT_INT16&&src->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT&&fl>=-64&&fl<=64){
  const int16_t *in=(const int16_t*)src->data+begin;float *out=(float*)dst->data+offset;float scale=ldexpf(1.0f,-fl);
  if(dst->p.data_format==VIP_BUFFER_FORMAT_FP32){for(unsigned j=0;j<count;j++)out[j]=(float)in[j]*scale;}
  else {__fp16 *half=(__fp16*)dst->data+offset;for(unsigned j=0;j<count;j++)half[j]=(__fp16)((float)in[j]*scale);}
 }else copy_values(dst,offset,src,begin,count);
}
static int selftest(void){
 for(int fl=-64;fl<=64;fl++)for(int n=-32768;n<=32767;n++){
  float a=ldexpf((float)n,-fl),b=(float)n*ldexpf(1.0f,-fl);if(memcmp(&a,&b,4))return 3;__fp16 ha=(__fp16)a,hb=(__fp16)b;if(memcmp(&ha,&hb,2))return 4;
 }
 puts("all 8454144 int16/scale cases FP32 and FP16 bit-identical");return 0;
}

int main(int argc,char **argv){
 if(argc==2&&!strcmp(argv[1],"--selftest"))return selftest();
 fast_mode=getenv("VIT_FAST_COPY")&&atoi(getenv("VIT_FAST_COPY"));
 if(argc!=4)return 2;
 const int count=65*1024;Layer nets[26]={0};float *inputs[2];char path[2048];
 for(int im=0;im<2;im++){inputs[im]=calloc(37632,4);if(!inputs[im])return 2;snprintf(path,sizeof(path),"%s/input%d.f32",argv[2],im+1);read_file(path,inputs[im],37632);}
 check(vip_init(),"init");double load_start=now();
 for(int k=0;k<26;k++){
  Layer *l=&nets[k];if(k==0)snprintf(path,sizeof(path),"%s/stem.nb",argv[1]);else if(k==25)snprintf(path,sizeof(path),"%s/projector.nb",argv[1]);else snprintf(path,sizeof(path),"%s/layer%02d.nb",argv[1],k-1);check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&l->net),"create");unsigned ni=0,no=0;
  check(vip_query_network(l->net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
  check(vip_prepare_network(l->net),"prepare");tensor_init(l,&l->in[0],0,0,k==0?37632:count);tensor_init(l,&l->out[0],0,1,k==25?14336:count);
 }
 double load_seconds=now()-load_start;long long bytes=npu_bytes();
 snprintf(path,sizeof(path),"%s/events.jsonl",argv[3]);FILE *events=fopen(path,"wx");if(!events)return 2;
 fprintf(events,"{\"event\":\"ready\",\"layers\":26,\"rows\":65,\"driver_bytes\":%lld,\"load_seconds\":%.9f}\n",bytes,load_seconds);fflush(events);
 for(int run=-1;run<6;run++){
  if(temperature()>=78000)return 3;
  int im=run<0?0:run%2;double start=now(),copy=0,api[26];
  double t=now();for(int j=0;j<37632;j++)put(&nets[0].in[0],j,inputs[im][j]);copy+=now()-t;
  for(int k=0;k<26;k++){
   Layer *l=&nets[k];t=now();check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");copy+=now()-t;
   t=now();check(vip_run_network(l->net),"run");api[k]=now()-t;
   t=now();check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");
   if(k<25)tower_copy(&nets[k+1].in[0],0,&l->out[0],0,count);
   copy+=now()-t;
  }
  float output[14336];t=now();
  for(int j=0;j<14336;j++){output[j]=get(&nets[25].out[0],j);if(!isfinite(output[j]))return 3;}
  copy+=now()-t;double elapsed=now()-start;
  snprintf(path,sizeof(path),"%s/run%d-output.f32",argv[3],run);FILE *f=fopen(path,"wb");if(!f)return 2;
  if(fwrite(output,4,14336,f)!=14336)return 2;fclose(f);
  fprintf(events,"{\"event\":\"result\",\"run\":%d,\"image\":%d,\"stack_seconds\":%.9f,\"host_copy_sync_seconds\":%.9f,\"temperature_mC\":%d,\"api_seconds\":[",run,im+1,elapsed,copy,temperature());
  for(int k=0;k<26;k++)fprintf(events,"%s%.9f",k?",":"",api[k]);fprintf(events,"]}\n");fflush(events);
 }
 for(int k=0;k<26;k++)free_net(&nets[k],1,1);check(vip_destroy(),"destroy");
 fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld}\n",npu_bytes());fclose(events);free(inputs[0]);free(inputs[1]);return 0;
}
