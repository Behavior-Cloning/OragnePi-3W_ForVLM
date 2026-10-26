#define OPI_FULL_MODEL_ENTRY unused_language_main
#include "full_model.c"
int main(int argc,char **argv){
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
  int im=run<0?0:run%2;double start=now(),copy=0,api[26],flush_sum=0,invalidate_sum=0,convert_sum=0;
  double t=now();for(int j=0;j<150528;j++)put(&nets[0].in[0],j,inputs[im][j]);copy+=now()-t;
  for(int k=0;k<26;k++){
   Layer *l=&nets[k];t=now();check(vip_flush_buffer(l->in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");double dt=now()-t;copy+=dt;flush_sum+=dt;
   t=now();check(vip_run_network(l->net),"run");api[k]=now()-t;
   t=now();check(vip_flush_buffer(l->out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"invalidate");dt=now()-t;copy+=dt;invalidate_sum+=dt;t=now();
   if(k<25)copy_values(&nets[k+1].in[0],0,&l->out[0],0,count);
   dt=now()-t;copy+=dt;convert_sum+=dt;
  }
  float output[57344];t=now();
  for(int j=0;j<57344;j++){output[j]=get(&nets[25].out[0],j);if(!isfinite(output[j]))return 3;}
  copy+=now()-t;double elapsed=now()-start;
  snprintf(path,sizeof(path),"%s/run%d-output.f32",argv[3],run);FILE *f=fopen(path,"wb");if(!f)return 2;
  if(fwrite(output,4,57344,f)!=57344)return 2;fclose(f);
  fprintf(events,"{\"event\":\"breakdown\",\"run\":%d,\"flush_s\":%.9f,\"invalidate_s\":%.9f,\"convert_s\":%.9f}\n",run,flush_sum,invalidate_sum,convert_sum);
  fprintf(events,"{\"event\":\"result\",\"run\":%d,\"image\":%d,\"stack_seconds\":%.9f,\"host_copy_sync_seconds\":%.9f,\"temperature_mC\":%d,\"api_seconds\":[",run,im+1,elapsed,copy,temperature());
  for(int k=0;k<26;k++)fprintf(events,"%s%.9f",k?",":"",api[k]);fprintf(events,"]}\n");fflush(events);
 }
 for(int k=0;k<26;k++)free_net(&nets[k],1,1);check(vip_destroy(),"destroy");
 fprintf(events,"{\"event\":\"closed\",\"driver_bytes\":%lld}\n",npu_bytes());fclose(events);free(inputs[0]);free(inputs[1]);return 0;
}
