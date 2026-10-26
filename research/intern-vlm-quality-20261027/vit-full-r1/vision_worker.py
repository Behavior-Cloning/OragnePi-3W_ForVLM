from pathlib import Path
import os,subprocess,selectors,json,struct,time
class VisionWorker:
 def __init__(self,binary,graphs,output,log):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.seq=0;self.buf=bytearray()
  rd,wr=os.pipe();self.fd=rd;self.sel=selectors.DefaultSelector();self.sel.register(rd,selectors.EVENT_READ)
  env=dict(os.environ);env['VIT_EVENT_FD']=str(wr);self.log=open(log,'xb')
  self.p=subprocess.Popen(['taskset','-c','7',binary,graphs,str(self.output)],stdin=subprocess.PIPE,stdout=self.log,stderr=self.log,pass_fds=(wr,),env=env);os.close(wr)
  try:self.ready=self.event();assert self.ready['event']=='ready'
  except BaseException:self.abort();raise
 def event(self,timeout=180):
  deadline=time.monotonic()+timeout
  while b'\n' not in self.buf:
   if not self.sel.select(max(0,deadline-time.monotonic())):raise TimeoutError('vision event')
   b=os.read(self.fd,65536)
   if not b:raise RuntimeError('vision closed unexpectedly')
   self.buf.extend(b)
  line,_,rest=self.buf.partition(b'\n');self.buf=bytearray(rest);return json.loads(line)
 def encode(self,pixels):
  assert len(pixels)==150528*4
  self.seq+=1;self.p.stdin.write(struct.pack('<3I',0x56495431,self.seq,150528)+pixels);self.p.stdin.flush();e=self.event()
  assert e['event']=='encoded' and e['id']==self.seq and e['visual_tokens']==64
  f=self.output/(str(self.seq)+'.f32');data=f.read_bytes();assert len(data)==57344*4;f.unlink();return e,data
 def abort(self):
  if self.p.poll() is None:
   self.p.terminate()
   try:self.p.wait(10)
   except subprocess.TimeoutExpired:self.p.kill();self.p.wait()
  self.sel.close();os.close(self.fd);self.log.close()
 def close(self):
  try:
   self.p.stdin.write(struct.pack('<3I',0x56495431,0,0));self.p.stdin.flush();e=self.event();self.p.wait(20)
   assert e['event']=='closed' and e['driver_bytes']==0 and self.p.returncode==0
   return e
  finally:self.abort()
