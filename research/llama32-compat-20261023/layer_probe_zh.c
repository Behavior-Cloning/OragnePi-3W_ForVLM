/* Six calibration + nine Chinese heldout layer0 states. IDs 100+ map to actual positions in heldout-zh-fixtures.json. */
#define main inherited_two_layer_probe_main
#include "board_probe.c"
#undef main
int main(int argc,char **argv){
 if(argc!=4){fprintf(stderr,"usage: layer_probe network.nb fixture_dir output_dir\n");return 2;}
 Layer l={0};unsigned ins[]={2048,8*32*64,8*32*64,33,64,64},outs[]={2048,8*64,8*64};
 check(vip_init(),"init");check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&l.net),"create");
 unsigned timeout=10000;check(vip_set_network(l.net,VIP_NETWORK_PROP_SET_TIME_OUT,&timeout),"timeout");check(vip_prepare_network(l.net),"prepare");
 for(unsigned j=0;j<6;j++)tensor_init(&l,&l.in[j],j,0,ins[j]);
 for(unsigned j=0;j<3;j++)tensor_init(&l,&l.out[j],j,1,outs[j]);
 int positions[]={0,1,2,4,8,14,100,101,102,110,111,112,120,121,122};double worst=0;long long bytes=npu_bytes();
 for(unsigned i=0;i<sizeof(positions)/sizeof(positions[0]);i++){
  char path[1024];snprintf(path,sizeof(path),"%s/p%d-inputs.f32",argv[2],positions[i]);FILE *f=fopen(path,"rb");if(!f)return 2;
  for(unsigned j=0;j<6;j++)for(unsigned k=0;k<ins[j];k++){float x;if(fread(&x,4,1,f)!=1)return 2;put(&l.in[j],k,x);}fclose(f);
  for(unsigned j=0;j<6;j++)check(vip_flush_buffer(l.in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");
  double t=now();check(vip_run_network(l.net),"run");t=now()-t;
  check(vip_flush_buffer(l.out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read");
  float ref[2048];snprintf(path,sizeof(path),"%s/p%d-reference.f32",argv[2],positions[i]);read_file(path,ref,2048);
  double e2=0,r2=0,maxerr=0;
  snprintf(path,sizeof(path),"%s/p%d-output.f32",argv[3],positions[i]);f=fopen(path,"wb");if(!f)return 2;
  for(int j=0;j<2048;j++){float z=get(&l.out[0],j);if(!isfinite(z))return 3;double e=z-ref[j];e2+=e*e;r2+=ref[j]*ref[j];maxerr=fmax(maxerr,fabs(e));fwrite(&z,4,1,f);}fclose(f);
  double rel=sqrt(e2/(r2+1e-30));worst=fmax(worst,rel);
  printf("{\"event\":\"fixture\",\"fixture_id\":%d,\"run_seconds\":%.8f,\"hidden_relative_l2\":%.8f,\"hidden_max_abs\":%.8f,\"temperature_mC\":%d}\n",positions[i],t,rel,maxerr,temperature());
 }
 for(int j=0;j<6;j++){check(vip_unmap_buffer(l.in[j].b),"unmap");check(vip_destroy_buffer(l.in[j].b),"free");}
 for(int j=0;j<3;j++){check(vip_unmap_buffer(l.out[j].b),"unmap");check(vip_destroy_buffer(l.out[j].b),"free");}
 check(vip_destroy_network(l.net),"destroy network");check(vip_destroy(),"destroy");
 printf("{\"event\":\"closed\",\"driver_loaded\":%lld,\"driver_released\":%lld,\"worst_hidden_relative_l2\":%.8f}\n",bytes,npu_bytes(),worst);
 return 0; /* Numerical error is reported, not an automatic model-quality pass. */
}
