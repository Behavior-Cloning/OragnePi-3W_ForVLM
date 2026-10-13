"""Exercise the real C partial-KV publisher with canaries in both cache layouts."""
from pathlib import Path
import subprocess
import tempfile
import unittest


class PartialKVTest(unittest.TestCase):
    def test_only_valid_rows_are_written_with_original_source_stride(self):
        source=Path(__file__).with_name('full_model.c').read_text()
        functions=source[source.index('static void copy_values('):source.index('/* Opt-in diagnostic readback')]
        prefix=r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define KV 3
#define HD 4
#define VIP_BUFFER_FORMAT_FP32 1
#define VIP_BUFFER_FORMAT_FP16 2
#define VIP_BUFFER_FORMAT_INT16 3
#define VIP_BUFFER_FORMAT_INT8 4
#define VIP_BUFFER_QUANTIZE_NONE 0
#define VIP_BUFFER_QUANTIZE_DYNAMIC_FIXED_POINT 1
typedef struct {
 unsigned count; void *data;
 struct { int data_format,quant_format; struct {struct {int fixed_point_pos;} dfp;} quant_data;} p;
} Tensor;
static float get(Tensor *t,unsigned i){assert(i<t->count);return t->p.data_format==1?((float*)t->data)[i]:((int16_t*)t->data)[i];}
static void put(Tensor *t,unsigned i,float v){assert(i<t->count);if(t->p.data_format==1)((float*)t->data)[i]=v;else ((int16_t*)t->data)[i]=(int16_t)v;}
'''
        main=r'''
int main(void){
 for(int format=1;format<=3;format+=2)for(int valid=1;valid<=8;valid++){
  int n=8,c=19,pos=11;
  float src_bytes[KV*8*HD],dst_bytes[KV*19*HD];
  Tensor src={0},dst={0};src.data=src_bytes;dst.data=dst_bytes;
  src.count=KV*n*HD;dst.count=KV*c*HD;
  src.p.data_format=dst.p.data_format=format;
  src.p.quant_format=dst.p.quant_format=format==1?0:1;
  for(unsigned i=0;i<src.count;i++)put(&src,i,100+i);
  for(unsigned i=0;i<dst.count;i++)put(&dst,i,-999);
  if(valid==n)append_key(&dst,&src,c,pos,n);
  else append_key_rows(&dst,&src,c,pos,valid,n);
  for(int h=0;h<KV;h++)for(int t=0;t<c;t++)for(int j=0;j<HD;j++){
   unsigned index=KEY_CACHE_TRANSPOSED?(h*HD+j)*c+t:(h*c+t)*HD+j;
   float expected=t>=pos&&t<pos+valid?100+(h*n+t-pos)*HD+j:-999;
   assert(get(&dst,index)==expected);
  }
 }
 return 0;
}
'''
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'partial.c';path.write_text(prefix+functions+main)
            for transposed in (0,1):
                binary=root/f'partial-{transposed}'
                subprocess.run(['cc','-O1','-Wall','-Wextra','-Werror','-fsanitize=address,undefined',
                                f'-DKEY_CACHE_TRANSPOSED={transposed}',str(path),'-o',str(binary)],check=True)
                subprocess.run([str(binary)],check=True)


if __name__=='__main__':unittest.main()
