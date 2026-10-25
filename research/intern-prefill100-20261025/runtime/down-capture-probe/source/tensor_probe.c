/* Generic bounded graph probe using the same production tensor conversion. */
#ifndef FUSED_LAYERS
#define FUSED_LAYERS 2
#endif
#define OPI_FULL_MODEL_ENTRY full_model_example_main
#include "full_model.c"

typedef struct {double input_flush_s,run_s,output_invalidate_s,wall_s;} ProbeIo;
static void probe_execute_io(Layer *l,unsigned ni,unsigned no,double *api,ProbeIo *io) {
    double a=now();
    for(unsigned i=0;i<ni;i++)check(vip_flush_buffer(l->in[i].b,VIP_BUFFER_OPER_TYPE_FLUSH),"probe input flush");
    double b=now();check(vip_run_network(l->net),"probe inference");double c=now();
    for(unsigned i=0;i<no;i++)check(vip_flush_buffer(l->out[i].b,VIP_BUFFER_OPER_TYPE_INVALIDATE),"probe output invalidate");
    double end=now();*api+=c-b;
    io->input_flush_s=b-a;io->run_s=c-b;io->output_invalidate_s=end-c;io->wall_s=end-a;
}
static unsigned long long logical_tensor_bytes(const Tensor *t) {
    unsigned width=t->p.data_format==VIP_BUFFER_FORMAT_FP32?4:
       (t->p.data_format==VIP_BUFFER_FORMAT_FP16||t->p.data_format==VIP_BUFFER_FORMAT_INT16?2:1);
    return (unsigned long long)t->count*width;
}

int main(int argc,char **argv) {
    if(argc!=6)return 2;
    FILE *input=fopen(argv[2],"rb"),*output=fopen(argv[3],"wb");
    if(!input||!output)return 2;
    unsigned ni=0,no=0;int repeats=atoi(argv[4]);if(repeats<1||repeats>100)return 2;
    int io_profile=getenv("OPI_PROBE_IO_PROFILE")!=NULL;ProbeIo io[100]={{0}};
    int start_temp=temperature();double cpu_start=cpu_time();
    check(vip_init(),"probe init");Layer layer={0};
    check(vip_create_network(argv[1],0,VIP_CREATE_NETWORK_FROM_FILE,&layer.net),"probe load");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_INPUT_COUNT,&ni),"inputs");
    check(vip_query_network(layer.net,VIP_NETWORK_PROP_OUTPUT_COUNT,&no),"outputs");
    if(ni!=TENSOR_MAX_INPUTS||(no!=5&&no!=7))return 2;
    int npd=getenv("OPI_PROBE_NPD")!=NULL;
    if(npd){vip_bool_e enabled=1;check(vip_set_network(layer.net,VIP_NETWORK_PROP_SET_ENABLE_NPD,&enabled),"enable NPD");}
    int preload=getenv("OPI_PROBE_SRAM_PRELOAD")?atoi(getenv("OPI_PROBE_SRAM_PRELOAD")):-1;
    vip_status_e preload_status=VIP_SUCCESS;
    if(preload>=0){
        if(preload>1)return 2;
        vip_bool_e enabled=preload;
        preload_status=vip_set_network(layer.net,VIP_NETWORK_PROP_SET_VIPSRAM_PRELOAD,&enabled);
        fprintf(stderr,"SRAM preload requested=%d status=%d\n",preload,(int)preload_status);
    }
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
    int query_rows=(int)(layer.in[0].count/DIM),context=(int)(layer.in[1].count/(KV*HD));
    if((context!=128&&context!=512)||(query_rows!=8&&query_rows!=16&&query_rows!=32&&query_rows!=64&&query_rows!=128&&query_rows!=256))return 2;
    if(layer.in[0].count!=(unsigned)(query_rows*DIM))return 2;
    for(unsigned i=1;i<=4;i++)if(layer.in[i].count!=(unsigned)(KV*context*HD))return 2;
    for(unsigned i=5;i<=6;i++)if(layer.in[i].count!=(unsigned)(7*query_rows*(context+query_rows)))return 2;
    for(unsigned i=7;i<=10;i++)if(layer.in[i].count!=(unsigned)(query_rows*HD))return 2;
    if(layer.out[0].count!=(unsigned)(query_rows*DIM))return 2;
    for(unsigned i=1;i<=4;i++)if(layer.out[i].count!=(unsigned)(KV*query_rows*HD))return 2;
    if(no==7){
        if(context!=512||query_rows!=32)return 2;
        for(unsigned i=5;i<7;i++)if(layer.out[i].count!=32u*4864u)return 2;
    }
    unsigned long long input_bytes=0,output_bytes=0;
    for(unsigned i=0;i<ni;i++)input_bytes+=logical_tensor_bytes(&layer.in[i]);
    for(unsigned i=0;i<no;i++)output_bytes+=logical_tensor_bytes(&layer.out[i]);
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
    double seconds=0,resident_seconds=0,api_seconds[100]={0};unsigned long long resident_mismatches=0;
    int warmup=getenv("OPI_PROBE_WARMUP")?atoi(getenv("OPI_PROBE_WARMUP")):0;
    if(warmup<0||warmup>10||(warmup&&npd))return 2;
    double warmup_seconds=0;
    for(int i=0;i<warmup;i++)execute(&layer,ni,no,&warmup_seconds);
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
        if(io_profile)probe_execute_io(&layer,ni,no,&api_seconds[i],&io[i]);
        else execute(&layer,ni,no,&api_seconds[i]);
        seconds+=api_seconds[i];size_t offset=0;int changed=0;
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
    /* Append bounded probe diagnostics without changing production execution. */
    if(fseek(result,-2,SEEK_CUR))return 2;
    fprintf(result,",\"sram_preload_requested\":%d,\"sram_preload_status\":%d,\"warmup_repeats\":%d,\"api_seconds\":[",preload,(int)preload_status,warmup);
    for(int i=0;i<repeats;i++)fprintf(result,"%s%.9f",i?",":"",api_seconds[i]);
    fprintf(result,"],\"scope\":\"two_layer_probe\",\"capacity\":%d,\"query_rows\":%d,\"fusion_layers\":2,\"input_logical_bytes\":%llu,\"output_logical_bytes\":%llu,\"io_profile_enabled\":%s,\"io_profile\":[",context,query_rows,input_bytes,output_bytes,io_profile?"true":"false");
    if(io_profile)for(int i=0;i<repeats;i++)fprintf(result,"%s{\"input_flush_s\":%.9f,\"run_s\":%.9f,\"output_invalidate_s\":%.9f,\"wall_s\":%.9f}",i?",":"",io[i].input_flush_s,io[i].run_s,io[i].output_invalidate_s,io[i].wall_s);
    fprintf(result,"]}\n");
    fclose(result);return 0;
}
