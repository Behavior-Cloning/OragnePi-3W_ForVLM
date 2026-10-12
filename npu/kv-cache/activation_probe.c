/* Black-box activation checks with the same VIPLite IO helpers as the model. */
#define main decoder_probe_main
#include "board_probe.c"
#undef main
int main(int argc,char **argv) {
    if(argc!=6){fprintf(stderr,"usage: activation_probe graph.nb count cases input.f32 output.f32\n");return 2;}
    unsigned count=(unsigned)atoi(argv[2]),cases=(unsigned)atoi(argv[3]);
    if(!count||count>100000||!cases||cases>100)return 2;
    float *input=calloc((size_t)count*cases,sizeof(float)),*output=calloc(count,sizeof(float));
    if(!input||!output)return 2;
    read_file(argv[4],input,(size_t)count*cases);
    FILE *f=fopen(argv[5],"wb");if(!f)return 2;
    Layer layer={0};check(vip_init(),"init activation");
    check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&layer.net),"load activation");
    check(vip_prepare_network(layer.net),"prepare activation");
    tensor_init(&layer,&layer.in[0],0,0,count);tensor_init(&layer,&layer.out[0],0,1,count);
    for(unsigned c=0;c<cases;c++){
        for(unsigned i=0;i<count;i++)put(&layer.in[0],i,input[(size_t)c*count+i]);
        check(vip_flush_buffer(layer.in[0].b,VIP_BUFFER_OPER_TYPE_FLUSH),"flush activation");
        check(vip_run_network(layer.net),"run activation");
        check(vip_flush_buffer(layer.out[0].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"read activation");
        for(unsigned i=0;i<count;i++)output[i]=get(&layer.out[0],i);
        if(fwrite(output,sizeof(float),count,f)!=count)return 2;
    }
    check(vip_unmap_buffer(layer.in[0].b),"unmap");check(vip_destroy_buffer(layer.in[0].b),"free");
    check(vip_unmap_buffer(layer.out[0].b),"unmap");check(vip_destroy_buffer(layer.out[0].b),"free");
    check(vip_destroy_network(layer.net),"free activation");check(vip_destroy(),"release activation");
    fclose(f);free(input);free(output);return 0;
}
