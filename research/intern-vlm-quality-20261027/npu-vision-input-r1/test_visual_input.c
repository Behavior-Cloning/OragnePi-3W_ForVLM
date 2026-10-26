#include <assert.h>
#include "runtime/visual_input.h"
static FILE *fixture(uint32_t start,uint32_t cap,const float *v,int elements){
 FILE *f=tmpfile();assert(f);uint32_t m[2]={start,cap};
 assert(fwrite(m,4,2,f)==2);if(elements)assert(fwrite(v,4,elements,f)==(size_t)elements);rewind(f);return f;
}
int main(void){
 float table[24];for(int i=0;i<24;i++)table[i]=i;
 float v[8]={100,101,102,103,200,201,202,203};int cap=-1;
 assert(opi_input_value(table,2,2,1,4)==9);
 FILE *f=fixture(2,1,v,8);assert(opi_read_visual(f,2,5,4,&cap));fclose(f);assert(cap==1);
 assert(opi_input_value(table,0,1,2,4)==2);
 assert(opi_input_value(table,0,2,2,4)==102);
 assert(opi_input_value(table,0,3,2,4)==202);
 assert(opi_input_value(table,0,4,2,4)==2);
 /* Padded batch rows repeat the last valid position, not the next position. */
 for(int t=0;t<8;t++){int pos=2+(t<2?t:1);assert(opi_input_value(table,0,pos,1,4)==(t==0?101:201));}
 opi_clear_visual();assert(!opi_visual_rows);assert(opi_input_value(table,2,2,1,4)==9);
 f=fixture(4,1,v,8);assert(!opi_read_visual(f,2,5,4,&cap));fclose(f);
 f=fixture(2,3,v,8);assert(!opi_read_visual(f,2,5,4,&cap));fclose(f);
 f=fixture(2,1,v,7);assert(!opi_read_visual(f,2,5,4,&cap));fclose(f);
 v[7]=NAN;f=fixture(2,1,v,8);assert(!opi_read_visual(f,2,5,4,&cap));fclose(f);
 v[7]=INFINITY;f=fixture(2,1,v,8);assert(!opi_read_visual(f,2,5,4,&cap));fclose(f);
 f=fixture(2,1,v,8);assert(!opi_read_visual(f,0,5,4,&cap));fclose(f);
 f=fixture(2,1,v,8);assert(!opi_read_visual(f,65,100,4,&cap));fclose(f);
 assert(!opi_visual_rows);puts("visual row selection, padding, reset and malformed payload checks passed");return 0;
}
