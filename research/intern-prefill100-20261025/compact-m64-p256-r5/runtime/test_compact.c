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
typedef struct {Layer *layers;int tokens,past_capacity,previous_position;} BatchPrefill;
typedef struct {Layer *decoder;BatchPrefill batch,tail8,compact,compact256;int valid,blocks,capacity;} Session;
#define FUSED_LAYERS 2
#define DIM 896
#define KV 2
#define HD 64
#define IN_K(k) (1+(k))
#define IN_V(k) (3+(k))
#define IN_MASK(k) (5+(k))
#include "mask_templates.h"
#include "compact_history.h"
static void reset_session(Session *s) {
    int templated=getenv("OPI_MASK_TEMPLATE")!=NULL;
    s->valid=0;s->batch.previous_position=0;s->tail8.previous_position=0;s->compact.previous_position=0;s->compact256.previous_position=0;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Layer *l=&s->decoder[i];
        if(templated)restore_mask_template(&l->in[IN_MASK(k)],s->capacity,0,1,0);
        else{for(int j=0;j<s->capacity;j++)put(&l->in[IN_MASK(k)],j,-10000);
             put(&l->in[IN_MASK(k)],s->capacity,0);}
        for(unsigned j=0;j<l->in[IN_K(k)].count;j++){
            put(&l->in[IN_K(k)],j,0);put(&l->in[IN_V(k)],j,0);
        }
        BatchPrefill *banks[4]={&s->batch,&s->tail8,&s->compact,&s->compact256};
        for(int bank=0;bank<4;bank++)if(banks[bank]->tokens){
            Tensor *m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=banks[bank]->past_capacity,repeat=DIM/(KV*HD);
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
    s->valid=keep;s->batch.previous_position=keep;s->tail8.previous_position=keep;s->compact.previous_position=keep<128?keep:128;s->compact256.previous_position=keep<256?keep:256;
    for(int i=0;i<s->blocks;i++)for(int k=0;k<FUSED_LAYERS;k++){
        Tensor *m=&s->decoder[i].in[IN_MASK(k)];
        if(templated)restore_mask_template(m,s->capacity,0,1,keep);
        else{for(int j=0;j<s->capacity;j++)put(m,j,j<keep?0:-10000);
             put(m,s->capacity,0);}
        BatchPrefill *banks[4]={&s->batch,&s->tail8,&s->compact,&s->compact256};
        for(int bank=0;bank<4;bank++)if(banks[bank]->tokens){
            m=&banks[bank]->layers[i].in[IN_MASK(k)];
            int n=banks[bank]->tokens,c=banks[bank]->past_capacity,repeat=DIM/(KV*HD);
            if(templated)restore_mask_template(m,c,n,repeat,keep<c?keep:c);
            else for(int r=0;r<repeat;r++)for(int t=0;t<n;t++)for(int j=0;j<c+n;j++)
                put(m,(r*n+t)*(c+n)+j,(j<(keep<c?keep:c)||(j>=c&&j<=c+t))?0:-10000);
        }
    }
}

static void create_tensor(Tensor *x,int c,int n) {
 x->p.data_format=VIP_BUFFER_FORMAT_INT16;x->p.quant_format=VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT;
 if(getenv("OPI_TEST_MASK_FP16")){x->p.data_format=VIP_BUFFER_FORMAT_FP16;x->p.quant_format=VIP_BUFFER_QUANTIZE_NONE;}
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
static void check_packing(void) {
 for(int past=128;past<=256;past+=128)for(int key=0;key<2;key++)for(int keep=0;keep<=past;keep++){
  Tensor src={0},dst={0};src.count=KV*448*HD;dst.count=KV*past*HD;
  src.p.data_format=dst.p.data_format=VIP_BUFFER_FORMAT_INT16;
  src.p.quant_format=dst.p.quant_format=VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT;
  src.p.quant_data.dfp.fixed_point_pos=dst.p.quant_data.dfp.fixed_point_pos=7;
  src.p.num_of_dims=dst.p.num_of_dims=4;
  int tr=key&&KEY_CACHE_TRANSPOSED;
  src.p.sizes[0]=tr?448:HD;src.p.sizes[1]=tr?HD:448;src.p.sizes[2]=KV;src.p.sizes[3]=1;
  dst.p.sizes[0]=tr?past:HD;dst.p.sizes[1]=tr?HD:past;dst.p.sizes[2]=KV;dst.p.sizes[3]=1;
  src.data=malloc(src.count*2);dst.data=malloc(dst.count*2);
  for(unsigned j=0;j<src.count;j++)((uint16_t*)src.data)[j]=(uint16_t)(j*37+19);
  memset(dst.data,0x7a,dst.count*2);pack_history(&dst,&src,past,448,keep,key);
  for(int h=0;h<KV;h++)for(int t=0;t<past;t++)for(int d=0;d<HD;d++){
   int di=tr?(h*HD+d)*past+t:(h*past+t)*HD+d;
   int si=tr?(h*HD+d)*448+t:(h*448+t)*HD+d;
   if(((uint16_t*)dst.data)[di]!=(t<keep?((uint16_t*)src.data)[si]:0))exit(11);
  }
  dst.p.quant_data.dfp.fixed_point_pos=8;if(history_contract(&dst,&src,past,448,key))exit(12);
  free(src.data);free(dst.data);
 }
}
static void check_dispatch(void) {
 const int cases[][4]={{32,40,0,0},{32,72,1,0},{32,136,2,0},{32,264,2,2},{64,72,1,0},{64,136,2,0},{64,264,2,2},{128,72,1,0},{128,136,1,1},{128,264,1,2},{256,72,0,1},{0,264,0,0}};
 for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
  int pos=cases[i][0],end=pos+cases[i][1],counts[3]={0};
  while(pos<end){int bank=compact_bank_selection(64,64,pos,cases[i][0],end);if(bank){counts[bank]++;pos+=64;}else pos+=(end-pos>=32?32:1);}
  if(counts[1]!=cases[i][2]||counts[2]!=cases[i][3])exit(20);
 }
 if(compact_bank_selection(64,64,127,128,400)||compact_bank_selection(64,64,257,128,400)||compact_bank_selection(64,0,192,128,400)||compact_bank_selection(0,0,64,64,400))exit(21);
}
int main(void){check_dispatch();
 Session s={0};s.blocks=12;s.capacity=448;s.decoder=calloc(12,sizeof(Layer));
 BatchPrefill *banks[]={&s.batch,&s.tail8,&s.compact,&s.compact256};int caps[]={448,448,128,256},rows[]={32,8,64,64};
 for(int b=0;b<4;b++){banks[b]->layers=calloc(12,sizeof(Layer));banks[b]->tokens=rows[b];banks[b]->past_capacity=caps[b];}
 for(int i=0;i<12;i++)for(int k=0;k<2;k++){
  create_tensor(&s.decoder[i].in[IN_MASK(k)],448,0);
  for(int b=0;b<4;b++)create_tensor(&banks[b]->layers[i].in[IN_MASK(k)],caps[b],rows[b]);
  for(int kind=0;kind<2;kind++){Tensor *x=&s.decoder[i].in[kind?IN_V(k):IN_K(k)];create_tensor(x,KV*448*HD-1,0);memset(x->data,0x7f,x->count*2);}
 }
 int keep[]={0,64,128,392,64,128,0,423,96,0};
 for(unsigned z=0;z<sizeof(keep)/sizeof(keep[0]);z++){
  int n=keep[z];if(n)truncate_session(&s,n);else reset_session(&s);
  if(s.valid!=n||s.compact.previous_position!=(n<128?n:128)||s.compact256.previous_position!=(n<256?n:256))exit(13);
  for(int i=0;i<12;i++)for(int k=0;k<2;k++){
   check_mask(&s.decoder[i].in[IN_MASK(k)],448,0,n);
   for(int b=0;b<4;b++){Tensor *x=&banks[b]->layers[i].in[IN_MASK(k)];check_mask(x,caps[b],rows[b],n<caps[b]?n:caps[b]);memset(x->data,0,x->count*2);}
   if(!n)for(int kind=0;kind<2;kind++){Tensor *x=&s.decoder[i].in[kind?IN_V(k):IN_K(k)];for(unsigned j=0;j<x->count;j++)if(((uint16_t*)x->data)[j])exit(14);}
  }
 }
 for(int i=0;i<12;i++)for(int k=0;k<2;k++){
  free(s.decoder[i].in[IN_MASK(k)].data);free(s.decoder[i].in[IN_K(k)].data);free(s.decoder[i].in[IN_V(k)].data);
  for(int b=0;b<4;b++)free(banks[b]->layers[i].in[IN_MASK(k)].data);
 }
 free(s.decoder);for(int b=0;b<4;b++)free(banks[b]->layers);free_mask_templates();check_packing();
 printf("{\"masks\":%u,\"pack_cases\":772,\"transposed_key\":%d}\n",checked,KEY_CACHE_TRANSPOSED);return 0;
}
