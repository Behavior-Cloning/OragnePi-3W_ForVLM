/* Diagnostic timing only: never changes tensor contents or graph execution. */
#define TIMING_MAX_RECORDS 1024
#define TIMING_MAX_NETS 40
typedef struct {int stage,pos,rows,past,valid,nets;double prepare,api,publish,query;
    unsigned sdk_us[TIMING_MAX_NETS],sdk_cycles[TIMING_MAX_NETS];int sdk_status[TIMING_MAX_NETS];} TimingRecord;
static int timing_enabled=0,timing_count=0;
static TimingRecord timing_records[TIMING_MAX_RECORDS];
static inline void timing_record(int stage,int pos,int rows,int past,int valid,double start,double api_start,double api_end,double finish,Layer *nets,int count) {
    if(!timing_enabled)return;
    if(timing_count>=TIMING_MAX_RECORDS||count<0||count>TIMING_MAX_NETS)exit(2);
    TimingRecord *r=&timing_records[timing_count++];memset(r,0,sizeof(*r));
    r->stage=stage;r->pos=pos;r->rows=rows;r->past=past;r->valid=valid;r->nets=count;
    r->prepare=api_start-start;r->api=api_end-api_start;r->publish=finish-api_end;
    double query_start=now();
    for(int i=0;i<count;i++){
        vip_inference_profile_t p={0};r->sdk_status[i]=(int)vip_query_network(nets[i].net,VIP_NETWORK_PROP_PROFILING,&p);
        r->sdk_us[i]=p.inference_time;r->sdk_cycles[i]=p.total_cycle;
    }
    r->query=now()-query_start;
}
static inline void timing_emit(FILE *f) {
    fprintf(f,",\"timing_enabled\":%s,\"timing_trace\":[",timing_enabled?"true":"false");
    for(int i=0;i<timing_count;i++){
        TimingRecord *r=&timing_records[i];
        fprintf(f,"%s{\"stage\":%d,\"pos\":%d,\"rows\":%d,\"past\":%d,\"valid\":%d,\"prepare_s\":%.9f,\"api_s\":%.9f,\"publish_s\":%.9f,\"query_s\":%.9f,\"networks\":[",i?",":"",r->stage,r->pos,r->rows,r->past,r->valid,r->prepare,r->api,r->publish,r->query);
        for(int j=0;j<r->nets;j++)fprintf(f,"%s{\"status\":%d,\"sdk_us\":%u,\"sdk_cycles\":%u}",j?",":"",r->sdk_status[j],r->sdk_us[j],r->sdk_cycles[j]);
        fprintf(f,"]}");
    }
    fprintf(f,"]}\n");
}
