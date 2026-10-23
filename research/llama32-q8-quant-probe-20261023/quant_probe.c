/* One layer or full-vocabulary head shard. Frozen input validation, not text generation. */
#define main inherited_probe_main
#include "board_probe.c"
#undef main
int main(int argc,char **argv){
 if(argc!=5){fprintf(stderr,"usage: quant_probe network.nb fixture_dir output_dir shard(-1=decoder,0..3=head)\n");return 2;}
 int shard=atoi(argv[4]),head=shard>=0; if(shard>3||shard < -1)return 2;
 Layer l={0};unsigned ins[]={2048,8*256*64,8*256*64,257,64,64},outs[]={2048,512,512};
 unsigned ni=head?1:6,no=head?1:3,nf=head?16:40;if(head)outs[0]=32064;
 check(vip_init(),"init");check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&l.net),"create");
 unsigned timeout=10000;check(vip_set_network(l.net,VIP_NETWORK_PROP_SET_TIME_OUT,&timeout),"timeout");check(vip_prepare_network(l.net),"prepare");
 for(unsigned j=0;j<ni;j++)tensor_init(&l,&l.in[j],j,0,ins[j]);
 for(unsigned j=0;j<no;j++)tensor_init(&l,&l.out[j],j,1,outs[j]);
 long long loaded=npu_bytes();float ref[32064];
 for(unsigned i=0;i<nf;i++){
  char path[1024];snprintf(path,sizeof(path),"%s/%c%u-inputs.f32",argv[2],head?'h':'p',i);FILE *f=fopen(path,"rb");if(!f)return 2;
  for(unsigned j=0;j<ni;j++){for(unsigned k=0;k<ins[j];k++){float x;if(fread(&x,4,1,f)!=1)return 2;put(&l.in[j],k,x);}}
  fclose(f);
  for(unsigned j=0;j<ni;j++)check(vip_flush_buffer(l.in[j].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush");
  for(int repeat=0;repeat<3;repeat++){
   double t=now();check(vip_run_network(l.net),"run");t=now()-t;
   printf("{\"event\":\"timing\",\"head_shard\":%d,\"fixture_id\":%u,\"repeat\":%d,\"run_seconds\":%.9f,\"temperature_mC\":%d}\n",shard,i,repeat,t,temperature());
  }
  for(unsigned j=0;j<no;j++){
   check(vip_flush_buffer(l.out[j].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read");
   if(head)snprintf(path,sizeof(path),"%s/h%u-s%d-reference.f32",argv[2],i,shard);else snprintf(path,sizeof(path),"%s/p%u-ref%u.f32",argv[2],i,j);
   read_file(path,ref,outs[j]);double e2=0,r2=0,maxerr=0;unsigned ai=0,ri=0;float am=-INFINITY,rm=-INFINITY;
   if(head)snprintf(path,sizeof(path),"%s/h%u-s%d-output.f32",argv[3],i,shard);else snprintf(path,sizeof(path),"%s/p%u-out%u.f32",argv[3],i,j);
   f=fopen(path,"wb");if(!f)return 2;
   for(unsigned k=0;k<outs[j];k++){float z=get(&l.out[j],k);if(!isfinite(z))return 3;double e=z-ref[k];e2+=e*e;r2+=ref[k]*ref[k];maxerr=fmax(maxerr,fabs(e));if(z>am){am=z;ai=k;}if(ref[k]>rm){rm=ref[k];ri=k;}fwrite(&z,4,1,f);}fclose(f);
   printf("{\"event\":\"fixture\",\"head_shard\":%d,\"fixture_id\":%u,\"output\":%u,\"relative_l2\":%.9f,\"max_abs\":%.9f,\"argmax\":%u,\"reference_argmax\":%u}\n",shard,i,j,sqrt(e2/(r2+1e-30)),maxerr,ai,ri);
  }
 }
 for(unsigned j=0;j<ni;j++){check(vip_unmap_buffer(l.in[j].b),"unmap");check(vip_destroy_buffer(l.in[j].b),"free");}
 for(unsigned j=0;j<no;j++){check(vip_unmap_buffer(l.out[j].b),"unmap");check(vip_destroy_buffer(l.out[j].b),"free");}
 check(vip_destroy_network(l.net),"destroy network");check(vip_destroy(),"destroy");
 printf("{\"event\":\"closed\",\"driver_loaded\":%lld,\"driver_released\":%lld}\n",loaded,npu_bytes());return 0;
}
