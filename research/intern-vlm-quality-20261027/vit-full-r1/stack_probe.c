#define OPI_FULL_MODEL_ENTRY unused_language_main
#include "full_model.c"
int main(int argc,char **argv){
 if(argc!=4)return 2;
 const int count=257*1024;Layer nets[24]={0};float *inputs[2];char path[2048];
 for(int im=0;im<2;im++){inputs[im]=calloc(count,4);if(!inputs[im])return 2;snprintf(path,sizeof(path),"%s/input%d.f32",argv[2],im+1);read_file(path,inputs[im],count);}
 check(vip_init(),"init");double load_start=now();
 for(int k=0;k<24;k++){
  Layer *l=&nets[k];snprintf(path,sizeof(path),"%s/layer%02d.nb",argv[1],k);check(vip_create_network(path,0,VIP_CREATE_NETWORK_FROM_FILE,&l->net),"create");unsigned ni=0,no=0;
  check(vip_query_network(l->net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l->net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
  check(vip_prepare_network(l->net),"prepare");tensor_init(l,&l->in[0],0,0,count);tensor_init(l,&l->out[0],0,1,count);
 }
 double load_seconds=now()-load_start;long long bytes=npu_bytes();
 snprintf(path,sizeof(path),"%s/events.jsonl",argv[3]);FILE *events=fopen(path,"wx");if(!events)return 2;
 fprintf(events,"{\"event\":\"ready\",\"layers\":24,\"rows\":257,\"driver_bytes\":%lld,\"load_seconds\":%.9f}\n",bytes,load_seconds);fflush(events);
 for(int run=-1;run<6;run++){
  if(temperature()>=78000)return 3;
  int im=run<0?0:run%2;double start=now(),copy=0,api[24];
  double t=now();for(int j=0;j<count;j++)put(&nets[0].in[0],j,inputs[im][j]);copy+=now()-t;
  for(int k=0;k<24;k++){
   Layer *l=&nets[k];t=now();check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");copy+=now()-t;
   t=now();check(vip_run_network(l->net),"run");api[k]=now()-t;
   t=now();check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");
   if(k<23)copy_values(&nets[k+1].in[0],0,&l->out[0],0,count);
   copy+=now()-t;
  }
  double elapsed=now()-start;
  snprintf(path,sizeof(path),"%s/run%d-output.f32",argv[3],run);FILE *f=fopen(path,"wb");if(!f)return 2;
  for(int j=0;j<count;j++){float x=get(&nets[23].out[0],j);if(!isfinite(x))return 3;if(fwrite(&x,4,1,f)!=1)return 2;}fclose(f);
  /* Trace export is outside timed inference; on each image's first hot request. */
  if(run==0||run==1)for(int k=0;k<24;k++){
   snprintf(path,sizeof(path),"%s/image%d-layer%02d.f32",argv[3],im+1,k);f=fopen(path,"wb");if(!f)return 2;
   for(int j=0;j<count;j++){float x=get(&nets[k].out[0],j);if(!isfinite(x))return 3;if(fwrite(&x,4,1,f)!=1)return 2;}fclose(f);
  }
  fprintf(events,"{\"event\":\"result\",\"run\":%d,\"image\":%d,\"stack_seconds\":%.9f,\"host_copy_sync_seconds\":%.9f,\"temperature_mC\":%d,\"api_seconds\":[",run,im+1,elapsed,copy,temperature());
  for(int k=0;k<24;k++)fprintf(events,"%s%.9f",k?",":"",api[k]);fprintf(events,"]}\n");fflush(events);
 }
 for(int k=0;k<24;k++)free_net(&nets[k],1,1);check(vip_destroy(),"destroy");
 fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld}\n",npu_bytes());fclose(events);free(inputs[0]);free(inputs[1]);return 0;
}
