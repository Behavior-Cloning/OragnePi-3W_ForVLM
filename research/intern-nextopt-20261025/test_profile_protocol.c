#include <vip_lite.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define HEAD_SHARDS 4
typedef struct {vip_network net;} Layer;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void check(vip_status_e s,const char *where){if(s!=VIP_SUCCESS){fprintf(stderr,"%s",where);exit(2);}}
static vip_status_e test_query(vip_network net,vip_enum prop,void *ptr){(void)net;(void)prop;vip_inference_profile_t *p=ptr;p->inference_time=123;p->total_cycle=456;return VIP_SUCCESS;}
#define vip_query_network test_query
#include "runtime_profile.h"
int main(void){RuntimeProfile p={.enabled=1,.network_enabled=1};Layer l[12]={{0}};double t=profile_stamp(&p);profile_mark(&p,1,3,&t);p.calls[1]=31;profile_networks(&p,1,l,12);profile_networks(&p,1,l,12);printf("{\"event\":\"done\"");profile_emit(stdout,&p,12);puts("}");profile_reset(&p);if(!p.enabled||!p.network_enabled||p.calls[1]||p.network_us[1][0])return 3;return 0;}
