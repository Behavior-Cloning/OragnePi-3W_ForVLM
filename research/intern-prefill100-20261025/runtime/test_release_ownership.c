#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#define TENSOR_MAX_INPUTS 11
#define TENSOR_MAX_OUTPUTS 5
#define HEAD_SHARDS 4
struct Buffer {int refs,mapped;unsigned char *data;};
struct Network {struct Buffer *in[11],*out[5];int ni,no,grouprefs;};
struct Group {struct Network **nets;int count;};
typedef struct Buffer *vip_buffer;
typedef struct Network *vip_network;
typedef struct Group *vip_group;
typedef struct {vip_buffer b;void *data;} Tensor;
typedef struct {vip_network net;Tensor in[11],out[5];} Layer;
typedef struct {Layer *layers,*scalar;vip_group group;int grouped,tokens,blocks;} BatchPrefill;
typedef struct {Layer *decoder,head[4];BatchPrefill batch,tail8,bulk;vip_group decoder_group,head_group;int blocks;} Session;
static unsigned alive_buffers,alive_networks,alive_groups,checks;
static void check(int rc,const char *where){if(rc){fprintf(stderr,"mock failed: %s\n",where);exit(2);}}
static int vip_destroy_group(vip_group g){
 assert(g);for(int i=0;i<g->count;i++){assert(g->nets[i]->grouprefs>0);g->nets[i]->grouprefs--;}
 free(g->nets);free(g);alive_groups--;return 0;
}
static int vip_destroy_network(vip_network n){
 assert(n&&n->grouprefs==0);
 for(int i=0;i<n->ni;i++){assert(n->in[i]->refs>0);n->in[i]->refs--;}
 for(int i=0;i<n->no;i++){assert(n->out[i]->refs>0);n->out[i]->refs--;}
 free(n);alive_networks--;return 0;
}
static int vip_unmap_buffer(vip_buffer b){assert(b&&b->mapped);b->mapped=0;return 0;}
static int vip_destroy_buffer(vip_buffer b){
 assert(b&&b->mapped==0&&b->refs==0);free(b->data);free(b);alive_buffers--;return 0;
}
#include "session_release.h"
static void create_layer(Layer *l,int ni,int no){
 l->net=calloc(1,sizeof(*l->net));assert(l->net);alive_networks++;
 l->net->ni=ni;l->net->no=no;
 for(int out=0;out<2;out++)for(int i=0;i<(out?no:ni);i++){
  Tensor *t=out?&l->out[i]:&l->in[i];t->b=calloc(1,sizeof(*t->b));assert(t->b);alive_buffers++;
  t->b->mapped=1;t->b->refs=1;t->data=t->b->data=calloc(8,1);assert(t->data);
  if(out)l->net->out[i]=t->b;else l->net->in[i]=t->b;
 }
}
static void bind(Layer *l,int port,vip_buffer b){l->net->in[port]->refs--;l->net->in[port]=b;b->refs++;}
static vip_group group(Layer *l,int count){vip_group g=calloc(1,sizeof(*g));assert(g);alive_groups++;g->count=count;g->nets=calloc(count,sizeof(*g->nets));assert(g->nets);for(int i=0;i<count;i++){g->nets[i]=l[i].net;g->nets[i]->grouprefs++;}return g;}
static void create_bank(BatchPrefill *b,Layer *d,int count,int grouped,int scalar,int tokens){
 b->blocks=count;b->tokens=tokens;b->layers=calloc(count,sizeof(Layer));assert(b->layers);
 for(int i=0;i<count;i++){
  create_layer(&b->layers[i],11,5);
  for(int p=1;p<5;p++)bind(&b->layers[i],p,d[i].in[p].b);
  if(i)bind(&b->layers[i],0,b->layers[i-1].out[0].b);
 }
 if(grouped){b->grouped=1;b->group=group(b->layers,count);}
 if(scalar){b->scalar=calloc(1,sizeof(Layer));assert(b->scalar);create_layer(b->scalar,11,5);for(int p=1;p<5;p++)bind(b->scalar,p,d[count-1].in[p].b);}
}
static void run(int count,int tail,int grouped,int scalar){
 Session s={0};s.blocks=count;s.decoder=calloc(count,sizeof(Layer));assert(s.decoder);
 for(int i=0;i<count;i++){create_layer(&s.decoder[i],11,5);if(i)bind(&s.decoder[i],0,s.decoder[i-1].out[0].b);}
 for(int i=0;i<4;i++)create_layer(&s.head[i],1,1);
 s.decoder_group=group(s.decoder,count);s.head_group=group(s.head,4);
 create_bank(&s.batch,s.decoder,count,grouped,scalar,16);
 create_bank(&s.bulk,s.decoder,count,grouped,0,128);
 if(tail)create_bank(&s.tail8,s.decoder,count,grouped,0,8);
 release_session_networks_and_buffers(&s);
 assert(alive_buffers==0&&alive_networks==0&&alive_groups==0);
 release_session_networks_and_buffers(&s); /* Re-entry must be harmless. */
 assert(alive_buffers==0&&alive_networks==0&&alive_groups==0);
 free(s.decoder);checks++;
}
int main(void){for(int repeat=0;repeat<8;repeat++)for(int count=1;count<=12;count+=11)for(int tail=0;tail<2;tail++)for(int grouped=0;grouped<2;grouped++)for(int scalar=0;scalar<2;scalar++)run(count,tail,grouped,scalar);printf("{\"sessions\":%u,\"double_close\":true,\"remaining_buffers\":%u,\"remaining_networks\":%u,\"remaining_groups\":%u}\n",checks,alive_buffers,alive_networks,alive_groups);return 0;}
