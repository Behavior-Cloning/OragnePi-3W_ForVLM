/* Generic bounded graph probe using the same production tensor conversion. */
#define FUSED_LAYERS 2
#define OPI_FULL_MODEL_ENTRY full_model_example_main
#include "full_model.c"

int main(int argc,char **argv) {
    if(argc!=6)return 2;
    FILE *input=fopen(argv[2],"rb"),*output=fopen(argv[3],"wb");
    if(!input||!output)return 2;
    unsigned ni=0,no=0;int repeats=atoi(argv[4]);if(repeats<1||repeats>100)return 2;
    int start_temp=temperature();double cpu_start=cpu_time();
    check(vip_init(),"probe init");Layer layer={0};
    check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&layer.net),"probe load");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"inputs");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"outputs");
    if(ni>TENSOR_MAX_INPUTS||no>TENSOR_MAX_OUTPUTS)return 2;
    int npd=getenv("OPI_PROBE_NPD")!=NULL;
    if(npd){vip_bool_e enabled=1;check(vip_set_network(layer.net,VIP_NETWORK_PROP_SET_ENABLE_NPD,&enabled),"enable NPD");}
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
    int resident=getenv("OPI_PROBE_RESIDENT_COPIES")?atoi(getenv("OPI_PROBE_RESIDENT_COPIES")):0;
    if(resident<0||resident>16|| (resident&&npd))return 2;
    Layer *copies=calloc(resident?resident:1,sizeof(Layer));if(!copies)return 2;
    unsigned ins[TENSOR_MAX_INPUTS],outs[TENSOR_MAX_OUTPUTS];
    for(unsigned j=0;j<ni;j++)ins[j]=layer.in[j].count;
    for(unsigned j=0;j<no;j++)outs[j]=layer.out[j].count;
    for(int c=0;c<resident;c++)load_net(&copies[c],argv[1],ni,no,ins,outs);
    size_t input_count=0;for(unsigned i=0;i<ni;i++)input_count+=layer.in[i].count;
    float *alternate=NULL,*original=NULL;
    if(getenv("OPI_PROBE_ALTERNATE")){
        alternate=calloc(input_count,sizeof(float));original=calloc(input_count,sizeof(float));
        if(!alternate||!original)return 2;
        read_file(getenv("OPI_PROBE_ALTERNATE"),alternate,input_count);
        size_t offset=0;
        for(unsigned i=0;i<ni;i++)for(unsigned j=0;j<layer.in[i].count;j++)original[offset++]=get(&layer.in[i],j);
    }
    size_t output_count=0;for(unsigned i=0;i<no;i++)output_count+=layer.out[i].count;
    float *first=calloc(output_count*(alternate?2:1),sizeof(float));if(!first)return 2;
    unsigned long long mismatches=0;int differing_repeats=0;
    double seconds=0,resident_seconds=0;unsigned long long resident_mismatches=0;
    for(int i=0;i<repeats;i++){
        int phase=alternate?i%2:0;
        if(alternate){
            const float *values=phase?alternate:original;size_t offset=0;
            for(unsigned o=0;o<ni;o++)for(unsigned j=0;j<layer.in[o].count;j++)put(&layer.in[o],j,values[offset++]);
        }
        for(int c=0;c<resident;c++){
            for(unsigned o=0;o<ni;o++)copy_values(&copies[c].in[o],0,&layer.in[o],0,layer.in[o].count);
            execute(&copies[c],ni,no,&resident_seconds);
        }
        execute(&layer,ni,no,&seconds);size_t offset=0;int changed=0;
        for(unsigned o=0;o<no;o++)for(unsigned j=0;j<layer.out[o].count;j++,offset++){
            float value=get(&layer.out[o],j);if(!isfinite(value))return 3;
            size_t at=phase*output_count+offset;
            if(i<(alternate?2:1))first[at]=value;
            else if(memcmp(&value,&first[at],sizeof(float))){mismatches++;changed=1;}
            for(int c=0;c<resident;c++){
                float other=get(&copies[c].out[o],j);
                if(!isfinite(other)||memcmp(&other,&first[at],sizeof(float)))resident_mismatches++;
            }
        }
        differing_repeats+=changed;
    }
    free(first);free(alternate);free(original);
    for(unsigned i=0;i<no;i++)for(unsigned j=0;j<layer.out[i].count;j++){
        float value=get(&layer.out[i],j);if(!isfinite(value))return 3;
        if(fwrite(&value,sizeof(value),1,output)!=1)return 2;
    }
    fclose(output);long long allocated=npu_bytes();
    for(int c=0;c<resident;c++)free_net(&copies[c],ni,no);
    free(copies);free_net(&layer,ni,no);check(vip_destroy(),"destroy");
    FILE *result=fopen(argv[5],"w");if(!result)return 2;
    fprintf(result,"{\"repeats\":%d,\"mean_api_seconds\":%.9f,\"allocated_bytes\":%lld,\"released_bytes\":%lld,\"differing_repeats\":%d,\"mismatched_elements\":%llu,\"npd_enabled\":%s,\"temperature_before_mC\":%d,\"temperature_after_mC\":%d,\"cpu_seconds\":%.6f,\"resident_copies\":%d,\"resident_mismatched_elements\":%llu,\"resident_api_seconds\":%.6f}\n",repeats,seconds/repeats,allocated,npu_bytes(),differing_repeats,mismatches,npd?"true":"false",start_temp,temperature(),cpu_time()-cpu_start,resident,resident_mismatches,resident_seconds);
    fclose(result);return 0;
}
