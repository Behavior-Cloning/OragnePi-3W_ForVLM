/* Isolated MiniCPM-V4.6 text decoder: all attention/MLP/head on VIPLite. */
#define MODEL_DIM 1024
#define MODEL_KV 2
#define MODEL_HD 256
#define FUSED_LAYERS 1
#define OPI_FULL_MODEL_ENTRY unused_full_model_entry
#include "full_model.c"
static Layer net[24],heads[8];
static vip_group decoder_group,head_group;
static const unsigned linear_in[]={1024,24576,262144},linear_out[]={1024,24576,262144};
static const unsigned full_in[]={1024,65024,65024,128,64,64},full_out[]={1024,512,512};
static void reset_state(void){
 for(int i=0;i<24;i++)for(int p=1;p<=2;p++){
  Tensor *t=&net[i].in[p];if(t->p.data_format!=VIP_BUFFER_FORMAT_FP16)exit(2);memset(t->data,0,t->count*2);
 }
}
static int forward_token(int token,int pos,const __fp16 *emb,int with_head,double *apis){
 for(int j=0;j<1024;j++)put(&net[0].in[0],j,emb[(size_t)token*1024+j]);
 for(int i=0;i<24;i++){
  Layer *l=&net[i];int full=i%4==3;
  if(full){
   for(int k=0;k<128;k++)put(&l->in[3],k,k>=127-pos?0:-10000);
   for(int j=0;j<64;j++){double angle=pos/pow(10000000.0,2.0*(j%32)/64.0);put(&l->in[4],j,cos(angle));put(&l->in[5],j,sin(angle));}
  }
  for(unsigned j=i?1:0;j<(full?6u:3u);j++)check(vip_flush_buffer(l->in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"group flush");
 }
 double stamp=now();check(vip_run_group(decoder_group,24),"run grouped decoder");apis[0]+=now()-stamp;
 for(int i=0;i<24;i++){
  Layer *l=&net[i];int full=i%4==3;
  for(unsigned j=i==23?0:1;j<3;j++)check(vip_flush_buffer(l->out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"group read");
  if(!full){copy_values(&l->in[1],0,&l->out[1],0,24576);copy_values(&l->in[2],0,&l->out[2],0,262144);}
  else for(int p=1;p<=2;p++){
   Tensor *t=&l->in[p];if(t->p.data_format!=VIP_BUFFER_FORMAT_FP16)exit(2);
   for(int h=0;h<2;h++){memmove((__fp16*)t->data+h*127*256,(__fp16*)t->data+h*127*256+256,126*256*2);copy_values(t,(h*127+126)*256,&l->out[p],h*256,256);}
  }
 }
 for(int j=0;j<1024;j++)if(!isfinite(get(&net[23].out[0],j)))return -1;
 if(!with_head)return 0;
 for(int i=0;i<8;i++){copy_values(&heads[i].in[0],0,&net[23].out[0],0,1024);check(vip_flush_buffer(heads[i].in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"group head flush");}
 stamp=now();check(vip_run_group(head_group,8),"run grouped head");apis[1]+=now()-stamp;
 float best=-INFINITY;int id=-1;
 for(int i=0;i<8;i++){
  check(vip_flush_buffer(heads[i].out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"group head read");
  for(unsigned j=0;j<heads[i].out[0].count;j++){float v=get(&heads[i].out[0],j);if(!isfinite(v))return -1;if(v>best){best=v;id=i*32768+j;}}
 }
 return id;
}

int main(int argc,char **argv){
 if(argc!=5){fprintf(stderr,"usage: mini_runtime networkdir prompt.i32 out.jsonl rounds\n");return 2;}
 char p[1024];snprintf(p,sizeof(p),"%s/embeddings.f16",argv[1]);int fd=open(p,O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st)||st.st_size!=(off_t)248094*1024*2)return 2;
 const __fp16 *emb=mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);if(emb==MAP_FAILED)return 2;
 FILE *f=fopen(argv[2],"rb");if(!f)return 2;int32_t prompt[128];size_t n=fread(prompt,4,128,f);if(n<1||n+16>128||!feof(f))return 2;fclose(f);for(size_t i=0;i<n;i++)if(prompt[i]<0||prompt[i]>=248094)return 2;
 FILE *out=fopen(argv[3],"w");if(!out)return 2;setvbuf(out,NULL,_IOLBF,0);
 double start=now();check(vip_init(),"init");
 for(int i=0;i<24;i++){snprintf(p,sizeof(p),"%s/l%d.nb",argv[1],i);int full=i%4==3;load_net(&net[i],p,full?6:3,3,full?full_in:linear_in,full?full_out:linear_out);}
 for(int i=0;i<8;i++){snprintf(p,sizeof(p),"%s/head%d.nb",argv[1],i);unsigned ins[]={1024},outs[]={(unsigned)(i==7?248094-7*32768:32768)};load_net(&heads[i],p,1,1,ins,outs);}
 unsigned group_timeout=30000;
 check(vip_create_group(24,&decoder_group),"create decoder group");check(vip_set_group(decoder_group,VIP_GROUP_PROP_SET_TIME_OUT,&group_timeout),"group timeout");
 for(int i=0;i<24;i++){
  if(i){Tensor *a=&net[i-1].out[0],*b=&net[i].in[0];if(a->count!=b->count||a->p.data_format!=VIP_BUFFER_FORMAT_FP16||b->p.data_format!=VIP_BUFFER_FORMAT_FP16||a->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE||b->p.quant_format!=VIP_BUFFER_QUANTIZE_NONE)return 2;check(vip_set_input(net[i].net,0,a->b),"hidden alias");}
  check(vip_add_network(decoder_group,net[i].net),"add decoder");
 }
 check(vip_create_group(8,&head_group),"create head group");check(vip_set_group(head_group,VIP_GROUP_PROP_SET_TIME_OUT,&group_timeout),"head timeout");for(int i=0;i<8;i++)check(vip_add_network(head_group,heads[i].net),"add head");
 fprintf(out,"{\"event\":\"ready\",\"load_seconds\":%.6f,\"npu_bytes\":%lld,\"rss_kib\":%ld,\"prompt_tokens\":%zu,\"scope\":\"text_decoder_W8_FC_INT16_boundaries_FP16_sensitive_ops_grouped\"}\n",now()-start,npu_bytes(),current_rss_kib(),n);
 int rounds=atoi(argv[4]);if(rounds<1||rounds>5)return 2;
 for(int round=0;round<rounds;round++){
  if(temperature()>75000){fprintf(out,"{\"event\":\"thermal_stop\"}\n");break;}
  if(getenv("MINI_PROMPTS_DIR")){
   char casepath[1024];snprintf(casepath,sizeof(casepath),"%s/%d.i32",getenv("MINI_PROMPTS_DIR"),round);FILE *cf=fopen(casepath,"rb");if(!cf)return 2;n=fread(prompt,4,128,cf);if(n<1||n+16>128||!feof(cf))return 2;fclose(cf);for(size_t i=0;i<n;i++)if(prompt[i]<0||prompt[i]>=248094)return 2;
   fprintf(out,"{\"event\":\"case_input\",\"round\":%d,\"prompt_tokens\":%zu}\n",round,n);
  }
  reset_state();double apis[32]={0},times[16]={0},cpu=cpu_time();int ids[16],before=temperature();start=now();int id=-1;
  for(size_t i=0;i<n;i++){id=forward_token(prompt[i],i,emb,i==n-1,apis);if(id<0)goto bad;}
  ids[0]=id;times[0]=now()-start;
  for(int k=1;k<16;k++){double t=now();id=forward_token(ids[k-1],n+k-1,emb,1,apis);if(id<0)goto bad;ids[k]=id;times[k]=now()-t;}
  double elapsed=now()-start;
  fprintf(out,"{\"event\":\"round\",\"round\":%d,\"ttft_seconds\":%.9f,\"decode_tokens_per_second\":%.6f,\"elapsed_seconds\":%.6f,\"cpu_seconds\":%.6f,\"temp_before_mC\":%d,\"temp_after_mC\":%d,\"rss_kib\":%ld,\"npu_bytes\":%lld,\"ids\":[",round,times[0],15/(elapsed-times[0]),elapsed,cpu_time()-cpu,before,temperature(),current_rss_kib(),npu_bytes());
  for(int k=0;k<16;k++)fprintf(out,"%s%d",k?",":"",ids[k]);fprintf(out,"],\"token_seconds\":[");for(int k=0;k<16;k++)fprintf(out,"%s%.9f",k?",":"",times[k]);fprintf(out,"],\"api_seconds_by_group\":[");for(int k=0;k<2;k++)fprintf(out,"%s%.9f",k?",":"",apis[k]);fprintf(out,"]}\n");continue;
 bad:fprintf(out,"{\"event\":\"nonfinite_output\",\"round\":%d}\n",round);break;
 }
 check(vip_destroy_group(decoder_group),"destroy decoder group");check(vip_destroy_group(head_group),"destroy head group");
 for(int i=0;i<24;i++)free_net(&net[i],i%4==3?6:3,3);for(int i=0;i<8;i++)free_net(&heads[i],1,1);check(vip_destroy(),"destroy");fprintf(out,"{\"event\":\"closed\",\"npu_bytes\":%lld}\n",npu_bytes());fclose(out);munmap((void*)emb,st.st_size);close(fd);return 0;
}
