#define OPI_FULL_MODEL_ENTRY unused_model_main
#include "full_model.c"
int main(int argc,char **argv){
 if(argc!=7)return 2;
 int input_count=atoi(argv[5]),count=atoi(argv[6]),rows=count; if(input_count<1||count<1)return 2;
 float *input=calloc(input_count,4),*first=calloc(count,4);if(!input||!first)return 2;read_file(argv[2],input,input_count);
 double load_start=now();check(vip_init(),"init");Layer l={0};check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&l.net),"load");
 unsigned ni=0,no=0;check(vip_query_network(l.net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"ni");check(vip_query_network(l.net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"no");if(ni!=1||no!=1)return 2;
 check(vip_prepare_network(l.net),"prepare");tensor_init(&l,&l.in[0],0,0,input_count);tensor_init(&l,&l.out[0],0,1,count);
 double load_seconds=now()-load_start;
 double ts[10],host[10],flush[10],readback[10];int before=temperature();long long bytes=npu_bytes(),different=0;
 for(int i=-2;i<10;i++){
  double h=now();for(int j=0;j<input_count;j++)put(&l.in[0],j,input[j]);double he=now()-h;
  double f=now();check(vip_flush_buffer(l.in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");double fe=now()-f;
  double t=now();check(vip_run_network(l.net),"run");double dt=now()-t;
  double rd=now();check(vip_flush_buffer(l.out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read");
  for(int j=0;j<count;j++){
   float v=get(&l.out[0],j);if(!isfinite(v))return 3;
   if(i==0)first[j]=v;else if(i>0&&memcmp(first+j,&v,4))different++;
  }
  double re=now()-rd;
  if(i>=0){ts[i]=dt;host[i]=he;flush[i]=fe;readback[i]=re;}
 }
 FILE *f=fopen(argv[3],"wb");if(!f||fwrite(first,4,count,f)!=(size_t)count)return 2;fclose(f);
 int after=temperature();free_net(&l,1,1);check(vip_destroy(),"destroy");free(input);free(first);
 f=fopen(argv[4],"w");if(!f)return 2;fprintf(f,"{\"scope\":\"ViT endpoint only\",\"rows\":%d,\"driver_bytes\":%lld,\"released_bytes\":%lld,\"temp_before\":%d,\"temp_after\":%d,\"differing_elements\":%lld,\"load_seconds\":%.9f",rows,bytes,npu_bytes(),before,after,different,load_seconds);
 const char *names[4]={"api_seconds","input_conversion_seconds","flush_seconds","readback_seconds"};double *values[4]={ts,host,flush,readback};
 for(int k=0;k<4;k++){fprintf(f,",\"%s\":[",names[k]);for(int i=0;i<10;i++)fprintf(f,"%s%.9f",i?",":"",values[k][i]);fprintf(f,"]");}
 fprintf(f,"}\n");fclose(f);return different?3:0;
}
