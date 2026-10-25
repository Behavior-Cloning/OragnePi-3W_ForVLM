#define _POSIX_C_SOURCE 200809L
#define __fp16 uint16_t /* Host only: opaque16bit storage; put emulates two exact IEEE constants. */
#include <vip_lite.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef struct { vip_buffer b; vip_buffer_create_params_t p; void *data; unsigned count; } Tensor;
static void put(Tensor *t,unsigned i,float x) {
    if(i>=t->count) {fprintf(stderr,"tensor index overflow\n");exit(2);}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP16) {if(x!=0 && x!=-10000)exit(9);uint16_t bits=x==0?0:0xf0e2;memcpy((unsigned char*)t->data+2*i,&bits,2);return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_FP32) {((float *)t->data)[i]=x;return;}
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT8 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int8_t *)t->data)[i]=(int8_t)fmaxf(-128,fminf(127,y));return;
    }
    if(t->p.data_format==VIP_BUFFER_FORMAT_INT16 && t->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT) {
        float y=roundf(ldexpf(x,t->p.quant_data.dfp.fixed_point_pos));
        ((int16_t *)t->data)[i]=(int16_t)fmaxf(-32768,fminf(32767,y));return;
    }
    fprintf(stderr,"unsupported input dtype/quant\n");exit(2);
}
static int same_tensor_contract(const Tensor *a,const Tensor *b) {
    if(a->count!=b->count||a->p.num_of_dims!=b->p.num_of_dims||
       a->p.data_format!=b->p.data_format||a->p.quant_format!=b->p.quant_format||
       memcmp(a->p.sizes,b->p.sizes,a->p.num_of_dims*sizeof(a->p.sizes[0])))return 0;
    if(a->p.quant_format==VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT)
        return a->p.quant_data.dfp.fixed_point_pos==b->p.quant_data.dfp.fixed_point_pos;
    return a->p.quant_format==VIP_BUFFER_QUANTIZE_NONE;
}


typedef struct {Tensor in[11];} Layer;
typedef struct {Layer *layers;int tokens,previous_position;} BatchPrefill;
typedef struct {Layer *decoder;BatchPrefill batch,tail8;int valid,blocks,capacity;} Session;
#define FUSED_LAYERS 2
#define DIM 896
#define KV 2
#define HD 64
#define IN_K(k) (1+(k))
#define IN_V(k) (3+(k))
#define IN_MASK(k) (5+(k))
#include "mask_templates.h"
static void reset_session(Session *s) {
    int templated=getenv("OPI_MASK_TEMPLATE")!=NULL;
    s->valid=0;s->batch.previous_position=0;s->tail8.previous_position=0;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Layer *l=&s->decoder[i];
        if(templated)restore_mask_template(&l->in[IN_MASK(k)],s->capacity,0,1,0);
        else{for(int j=0;j<s->capacity;j++)put(&l->in[IN_MASK(k)],j,-10000);
             put(&l->in[IN_MASK(k)],s->capacity,0);}
        for(unsigned j=0;j<l->in[IN_K(k)].count;j++){
            put(&l->in[IN_K(k)],j,0);put(&l->in[IN_V(k)],j,0);
        }
        BatchPrefill *banks[2]={&s->batch,&s->tail8};
        for(int bank=0;bank<2;bank++)if(banks[bank]->tokens){
            Tensor *m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=s->capacity,repeat=DIM/(KV*HD);
            if(templated)restore_mask_template(m,c,n,repeat,0);
            else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
                put(m,(r*n+t)*(c+n)+j,(j>=c&&j<=c+t)?0:-10000);
        }
    }
}

/* Causal prefix entries are independent of a discarded dynamic suffix.
 * Close every stale mask column before reusing the prefix; batch masks only
 * open columns monotonically during generation, so rewinding the cursor alone
 * is insufficient. Stale finite K/V values remain unreachable behind masks. */
static void truncate_session(Session *s,int keep) {
    int templated=getenv("OPI_MASK_TEMPLATE")!=NULL;
    s->valid=keep;s->batch.previous_position=keep;s->tail8.previous_position=keep;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Tensor *m=&s->decoder[i].in[IN_MASK(k)];
        if(templated)restore_mask_template(m,s->capacity,0,1,keep);
        else{for(int j=0;j<s->capacity;j++)put(m,j,j<keep?0:-10000);
             put(m,s->capacity,0);}
        BatchPrefill *banks[2]={&s->batch,&s->tail8};
        for(int bank=0;bank<2;bank++)if(banks[bank]->tokens){
            m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=s->capacity,repeat=DIM/(KV*HD);
            if(templated)restore_mask_template(m,c,n,repeat,keep);
            else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
                put(m,(r*n+t)*(c+n)+j,(j<keep||(j>=c&&j<=c+t))?0:-10000);
        }
    }
}

static void create_tensor(Tensor *x,int c,int n) {
 x->p.data_format=VIP_BUFFER_FORMAT_INT16;x->p.quant_format=VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT;
 x->p.quant_data.dfp.fixed_point_pos=0;x->count=n?7*n*(c+n):c+1;
 x->p.num_of_dims=4;x->p.sizes[0]=n?c+n:c+1;x->p.sizes[1]=n?n:1;x->p.sizes[2]=n?7:1;x->p.sizes[3]=1;
 x->data=calloc(x->count,2);if(!x->data)exit(2);
}
static unsigned checked=0;
static void check_mask(Tensor *a,int c,int n,int keep) {
 Tensor expected=*a;expected.data=calloc(a->count,2);if(!expected.data)exit(2);
 if(!n){for(int j=0;j<c;j++)put(&expected,j,j<keep?0:-10000);put(&expected,c,0);}
 else for(int r=0;r<7;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
  put(&expected,(r*n+t)*(c+n)+j,(j<keep||(j>=c&&j<=c+t))?0:-10000);
 if(memcmp(a->data,expected.data,a->count*2)){fprintf(stderr,"mask mismatch n%d keep%d\n",n,keep);exit(3);}
 free(expected.data);checked++;
}
static void run_case(int c,int has_tail) {
 Session s={0};s.blocks=12;s.capacity=c;s.decoder=calloc(12,sizeof(Layer));s.batch.layers=calloc(12,sizeof(Layer));s.tail8.layers=calloc(12,sizeof(Layer));s.batch.tokens=16;s.tail8.tokens=has_tail?8:0;
 for(int i=0;i<12;i++)for(int k=0;k<2;k++){
  create_tensor(&s.decoder[i].in[IN_MASK(k)],c,0);create_tensor(&s.batch.layers[i].in[IN_MASK(k)],c,16);
  if(has_tail)create_tensor(&s.tail8.layers[i].in[IN_MASK(k)],c,8);
  for(int kind=0;kind<2;kind++){Tensor *kv=&s.decoder[i].in[kind?IN_V(k):IN_K(k)];create_tensor(kv,KV*c*HD-1,0);memset(kv->data,0x7f,kv->count*2);}
 }
 int keeps[]={0,64,128,64,200,128,0,192,96,0};
 for(unsigned z=0;z<sizeof(keeps)/sizeof(keeps[0]);z++){
  int keep=keeps[z];
  if(!keep){reset_session(&s);for(int i=0;i<12;i++)for(int k=0;k<2;k++)for(int kind=0;kind<2;kind++){
   Tensor *x=&s.decoder[i].in[kind?IN_V(k):IN_K(k)];for(unsigned j=0;j<x->count;j++)if(((int16_t*)x->data)[j])exit(4);
  }}else truncate_session(&s,keep);
  if(s.valid!=keep||s.batch.previous_position!=keep||s.tail8.previous_position!=keep)exit(5);
  for(int i=0;i<12;i++)for(int k=0;k<2;k++){
   check_mask(&s.decoder[i].in[IN_MASK(k)],c,0,keep);check_mask(&s.batch.layers[i].in[IN_MASK(k)],c,16,keep);
   if(has_tail)check_mask(&s.tail8.layers[i].in[IN_MASK(k)],c,8,keep);
   /* Simulate both graph banks opening stale history columns; a future
    * truncate must close these even when the other bank ran most recently. */
   memset(s.batch.layers[i].in[IN_MASK(k)].data,0,s.batch.layers[i].in[IN_MASK(k)].count*2);
   if(has_tail)memset(s.tail8.layers[i].in[IN_MASK(k)].data,0,s.tail8.layers[i].in[IN_MASK(k)].count*2);
  }
 }
 for(int i=0;i<12;i++)for(int k=0;k<2;k++){
  free(s.decoder[i].in[IN_MASK(k)].data);free(s.batch.layers[i].in[IN_MASK(k)].data);if(has_tail)free(s.tail8.layers[i].in[IN_MASK(k)].data);
  free(s.decoder[i].in[IN_K(k)].data);free(s.decoder[i].in[IN_V(k)].data);
 }
 free(s.decoder);free(s.batch.layers);free(s.tail8.layers);
}
int main(void){run_case(1280,0);run_case(1280,1);run_case(256,1);free_mask_templates();if(mask_templates||mask_template_bytes)exit(6);printf("{\"checked_mask_tensors\":%u,\"cache_freed\":true,\"kv_reset_zero\":true}\n",checked);return 0;}
