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
 if(argc!=3)return 2;
 const char *fd=getenv("VIT_EVENT_FD");if(!fd)return 2;
 FILE *events=fdopen(atoi(fd),"w");if(!events)return 2;setvbuf(events,NULL,_IOLBF,0);fast_mode=1;
 const int count=263168;Layer nets[26]={0};char path[2048];float *input=calloc(150528,4),*output=calloc(57344,4);if(!input||!output)return 2;
 check(vip_init(),"init");double load=now();
 for(int k=0;k<26;k++){
  Layer *l=&nets[k];
  if(k==0)snprintf(path,sizeof(path),"%s/stem.nb",argv[1]);else if(k==25)snprintf(path,sizeof(path),"%s/projector.nb",argv[1]);else snprintf(path,sizeof(path),"%s/layer%02d.nb",argv[1],k-1);
  check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&l->net),"create");check(vip_prepare_network(l->net),"prepare");
  unsigned ni=0,no=0;check(vip_query_network(l->net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
  tensor_init(l,&l->in[0],0,0,k==0?150528:count);tensor_init(l,&l->out[0],0,1,k==25?57344:count);
 }
 fprintf(events,"{\"event\":\"ready\",\"driver_bytes\":%lld,\"load_seconds\":%.9f}\n",npu_bytes(),now()-load);
 int rc=0;unsigned last=0;
 while(1){
  uint32_t h[3];size_t n=fread(h,4,3,stdin);if(n==0)break;
  if(n!=3||h[0]!=0x56495431){rc=2;break;}if(h[1]==0&&h[2]==0)break;
  if(h[1]<=last||h[2]!=150528||fread(input,4,150528,stdin)!=150528){rc=2;break;}last=h[1];
  for(int j=0;j<150528;j++)if(!isfinite(input[j])){rc=3;break;}if(rc)break;
  if(temperature()>=78000){rc=3;break;}
  double start=now(),api=0;
  for(int j=0;j<150528;j++)put(&nets[0].in[0],j,input[j]);
  for(int k=0;k<26;k++){
   Layer *l=&nets[k];check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");
   double t=now();check(vip_run_network(l->net),"run");api+=now()-t;
   check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");
   if(k<25)tower_copy(&nets[k+1].in[0],0,&l->out[0],0,count);
  }
  for(int j=0;j<57344;j++){output[j]=get(&nets[25].out[0],j);if(!isfinite(output[j])){rc=3;break;}}if(rc)break;
  double encoded=now()-start;
  snprintf(path,sizeof(path),"%s/%u.f32",argv[2],h[1]);FILE *f=fopen(path,"wb");if(!f){rc=2;break;}size_t written=fwrite(output,4,57344,f);int closed=fclose(f);if(written!=57344||closed){rc=2;break;}
  fprintf(events,"{\"event\":\"encoded\",\"id\":%u,\"visual_tokens\":64,\"seconds\":%.9f,\"api_seconds\":%.9f,\"with_write_seconds\":%.9f}\n",h[1],encoded,api,now()-start);
 }
 for(int k=0;k<26;k++)free_net(&nets[k],1,1);check(vip_destroy(),"destroy");
 fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld,\"returncode\":%d}\n",npu_bytes(),rc);fclose(events);free(input);free(output);return rc;
}
