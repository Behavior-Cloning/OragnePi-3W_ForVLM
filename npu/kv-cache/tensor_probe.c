/* Generic bounded graph probe using the same production tensor conversion. */
#define FUSED_LAYERS 2
#define OPI_FULL_MODEL_ENTRY full_model_example_main
#include "full_model.c"

int main(int argc,char **argv) {
    if(argc!=6)return 2;
    FILE *input=fopen(argv[2],"rb"),*output=fopen(argv[3],"wb");
    if(!input||!output)return 2;
    unsigned ni=0,no=0;int repeats=atoi(argv[4]);if(repeats<1||repeats>100)return 2;
    check(vip_init(),"probe init");Layer layer={0};
    check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&layer.net),"probe load");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"inputs");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"outputs");
    if(ni>TENSOR_MAX_INPUTS||no>TENSOR_MAX_OUTPUTS)return 2;
    check(vip_prepare_network(layer.net),"prepare");
    for(unsigned output_flag=0;output_flag<2;output_flag++)for(unsigned i=0;i<(output_flag?no:ni);i++){
        unsigned dims=0,sizes[6]={0},count=1;
        vip_status_e (*query)(vip_network,vip_uint32_t,vip_enum,void*)=output_flag?vip_query_output:vip_query_input;
        check(query(layer.net,i,VIP_BUFFER_PROP_NUM_OF_DIMENSION,&dims),"dims");if(dims>6)return 2;
        check(query(layer.net,i,VIP_BUFFER_PROP_SIZES_OF_DIMENSION,sizes),"shape");
        for(unsigned j=0;j<dims;j++)count*=sizes[j];
        Tensor *t=output_flag?&layer.out[i]:&layer.in[i];tensor_init(&layer,t,i,output_flag,count);
        if(!output_flag)for(unsigned j=0;j<count;j++){
            float value;if(fread(&value,sizeof(value),1,input)!=1)return 2;put(t,j,value);
        }
    }
    if(fgetc(input)!=EOF)return 2;
    fclose(input);
    double seconds=0;for(int i=0;i<repeats;i++)execute(&layer,ni,no,&seconds);
    for(unsigned i=0;i<no;i++)for(unsigned j=0;j<layer.out[i].count;j++){
        float value=get(&layer.out[i],j);if(!isfinite(value))return 3;
        if(fwrite(&value,sizeof(value),1,output)!=1)return 2;
    }
    fclose(output);long long allocated=npu_bytes();free_net(&layer,ni,no);check(vip_destroy(),"destroy");
    FILE *result=fopen(argv[5],"w");if(!result)return 2;
    fprintf(result,"{\"repeats\":%d,\"mean_api_seconds\":%.9f,\"allocated_bytes\":%lld,\"released_bytes\":%lld}\n",repeats,seconds/repeats,allocated,npu_bytes());
    fclose(result);return 0;
}
