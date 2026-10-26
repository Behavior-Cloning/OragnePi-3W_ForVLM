#include "clip.h"
#include "clip-impl.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
using json=nlohmann::ordered_json;
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
int main(int argc,char **argv){
 if(argc!=2){std::cerr<<"usage: vision-worker mmproj.gguf\n";return 2;}
 ggml_backend_load_all();clip_context_params p{};p.use_gpu=false;p.flash_attn_type=CLIP_FLASH_ATTN_TYPE_DISABLED;p.warmup=false;
 auto t=Clock::now();auto c=clip_init(argv[1],p);if(!c.ctx_v) return 3;
 std::cout<<json({{"event","ready"},{"load_ms",ms(t)},{"backend","CPU"},{"embedding_dim",clip_n_mmproj_embd(c.ctx_v)}}).dump()<<std::endl;
 std::string line;
 while(std::getline(std::cin,line))try{
  auto q=json::parse(line);if(q.value("op","")=="close")break;
  int side=q.at("side"),threads=q.value("threads",2);
  if(side<56||side>448||side%28||threads<1||threads>8)throw std::runtime_error("invalid shape/threads");
  clip_image_f32 img;img.set_size({side,side},false,false);std::vector<float> pixels(size_t(side)*side*3);
  std::ifstream in(q.at("input").get<std::string>(),std::ios::binary);in.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*4);
  if(!in||in.peek()!=EOF)throw std::runtime_error("input byte count mismatch");
  for(float x:pixels)if(!std::isfinite(x))throw std::runtime_error("nonfinite input");
  img.cpy_buf(pixels);std::vector<float> emb;t=Clock::now();bool ok=clip_image_encode(c.ctx_v,threads,&img,emb);double elapsed=ms(t);
  const int tokens=(side/28)*(side/28),dim=clip_n_mmproj_embd(c.ctx_v);
  if(!ok||emb.size()!=size_t(tokens)*dim)throw std::runtime_error("encode or embedding count failure");
  double norm=0;for(float x:emb){if(!std::isfinite(x))throw std::runtime_error("nonfinite output");norm+=double(x)*x;}
  std::ofstream out(q.at("output").get<std::string>(),std::ios::binary);out.write(reinterpret_cast<char*>(emb.data()),emb.size()*4);if(!out)throw std::runtime_error("write failure");
  std::cout<<json({{"event","encoded"},{"id",q.value("id","")},{"side",side},{"image_tokens",tokens},{"dim",dim},{"threads",threads},{"encoder_projector_ms",elapsed},{"output_floats",emb.size()},{"l2_norm",std::sqrt(norm)}}).dump()<<std::endl;
 }catch(const std::exception &e){std::cout<<json({{"event","error"},{"error",e.what()}}).dump()<<std::endl;}
 clip_free(c.ctx_v);if(c.ctx_a)clip_free(c.ctx_a);if(c.ctx_gen_a)clip_free(c.ctx_gen_a);
 std::cout<<json({{"event","closed"}}).dump()<<std::endl;return 0;
}
