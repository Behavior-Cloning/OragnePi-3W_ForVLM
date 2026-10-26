#define OPI_FULL_MODEL_ENTRY unused_model_main
#include "full_model.c"
int main(int argc,char **argv){
 if(argc!=5)return 2;
 int count=65*1024;float *input=calloc(count,4);read_file(argv[2],input,count);
 check(vip_init(),"init");Layer l={0};check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&l.net),"load");
 unsigned ni=0,no=0;check(vip_query_network(l.net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l.net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
 check(vip_prepare_network(l.net),"prepare");tensor_init(&l,&l.in[0],0,0,count);tensor_init(&l,&l.out[0],0,1,count);
 for(int j=0;j<count;j++)put(&l.in[0],j,input[j]);
 double ts[10];int before=temperature();long long bytes=npu_bytes();
 for(int i=-2;i<10;i++){
  check(vip_flush_buffer(l.in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");double t=now();check(vip_run_network(l.net),"run");double dt=now()-t;
  check(vip_flush_buffer(l.out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read");if(i>=0)ts[i]=dt;
 }
 FILE *f=fopen(argv[3],"wb");if(!f)return 2;for(int j=0;j<count;j++){float v=get(&l.out[0],j);if(!isfinite(v))return 3;if(fwrite(&v,4,1,f)!=1)return 2;}fclose(f);
 int after=temperature();free_net(&l,1,1);check(vip_destroy(),"destroy");free(input);
 f=fopen(argv[4],"w");if(!f)return 2;fprintf(f,"{\"scope\":\"ViT first block only, 65 rows\",\"driver_bytes\":%lld,\"released_bytes\":%lld,\"temp_before\":%d,\"temp_after\":%d,\"api_seconds\":[",bytes,npu_bytes(),before,after);
 for(int i=0;i<10;i++)fprintf(f,"%s%.9f",i?",":"",ts[i]);fprintf(f,"]}\n");fclose(f);return 0;
}
