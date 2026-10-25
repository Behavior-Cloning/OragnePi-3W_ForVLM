/* Opt-in diagnostics only; no tensor or graph computation changes. */
#ifndef OPI_RUNTIME_PROFILE_H
#define OPI_RUNTIME_PROFILE_H
#define PROFILE_PHASES 4
#define PROFILE_STAGES 9
#define PROFILE_NETWORKS 80
static const char *profile_phase_names[PROFILE_PHASES]={"scalar_prompt","decode_decoder","first_head","decode_head"};
static const char *profile_stage_names[PROFILE_STAGES]={"embedding_rope","input_prepare","input_flush","npu_run","output_invalidate","kv_publish","hidden_read","sampling","logit_dump"};
typedef struct {
    int enabled,network_enabled;
    unsigned calls[PROFILE_PHASES],network_samples[PROFILE_PHASES];
    double seconds[PROFILE_PHASES][PROFILE_STAGES],network_query_seconds;
    unsigned long long network_us[PROFILE_PHASES][PROFILE_NETWORKS],network_cycles[PROFILE_PHASES][PROFILE_NETWORKS];
} RuntimeProfile;
static void profile_reset(RuntimeProfile *p) {
    int enabled=p->enabled,network=p->network_enabled;
    memset(p,0,sizeof(*p));p->enabled=enabled;p->network_enabled=network;
}
static double profile_stamp(RuntimeProfile *p){return p->enabled?now():0;}
static void profile_mark(RuntimeProfile *p,int phase,int stage,double *stamp){
    if(p->enabled){double t=now();p->seconds[phase][stage]+=t-*stamp;*stamp=t;}
}
static void profile_networks(RuntimeProfile *p,int phase,Layer *layers,int count) {
    if(!p->network_enabled)return;
    if(count>PROFILE_NETWORKS)exit(2);
    double t=now();
    for(int i=0;i<count;i++){
        vip_inference_profile_t q={0};
        check(vip_query_network(layers[i].net,VIP_NETWORK_PROP_PROFILING,&q),"profile network");
        p->network_us[phase][i]+=q.inference_time;
        p->network_cycles[phase][i]+=q.total_cycle;
    }
    p->network_query_seconds+=now()-t;p->network_samples[phase]++;
}
static void profile_emit(FILE *f,const RuntimeProfile *p,int blocks) {
    if(!p->enabled)return;
    fprintf(f,",\"stage_profile\":{\"enabled\":true,\"network_queries_enabled\":%s,\"network_query_seconds\":%.9f,\"phases\":{",p->network_enabled?"true":"false",p->network_query_seconds);
    for(int a=0;a<PROFILE_PHASES;a++){
        fprintf(f,"%s\"%s\":{\"calls\":%u,\"seconds\":{",a?",":"",profile_phase_names[a],p->calls[a]);
        for(int b=0;b<PROFILE_STAGES;b++)fprintf(f,"%s\"%s\":%.9f",b?",":"",profile_stage_names[b],p->seconds[a][b]);
        fprintf(f,"},\"network_samples\":%u,\"network_sum_inference_us\":[",p->network_samples[a]);
        int n=a<2?blocks:HEAD_SHARDS;
        for(int i=0;i<n;i++)fprintf(f,"%s%llu",i?",":"",p->network_us[a][i]);
        fprintf(f,"],\"network_sum_cycles\":[");
        for(int i=0;i<n;i++)fprintf(f,"%s%llu",i?",":"",p->network_cycles[a][i]);
        fprintf(f,"]}");
    }
    fprintf(f,"}}");
}
#endif
