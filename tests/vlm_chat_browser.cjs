// Integration test: real Python gateway + mock SSE backend + Chromium.
// This verifies the application, not model quality or board performance.
const {chromium} = require('playwright');
const {spawn} = require('node:child_process');
const http = require('node:http');
const assert = require('node:assert/strict');
const path = require('node:path');
const os = require('node:os');
const root = path.resolve(__dirname, '..');
const port = Number(process.env.VLM_TEST_PORT || 19080);
const received = [];
let behavior = 'normal';
let settings = {model_id:'smolvlm-256m', context:4096}, switching = false;
const control = http.createServer((req,res) => {
  const state = () => ({...settings, model:settings.model_id === 'smolvlm-256m' ? 'SmolVLM-256M' : 'InternVL3-1B',
    vision:settings.model_id !== 'qwen3-17b-q4',
    ready:!switching, loading:switching, models:[{id:'smolvlm-256m',name:'SmolVLM-256M',available:true},{id:'internvl3-1b',name:'InternVL3-1B',available:true},{id:'qwen3-17b-q4',name:'Qwen3-1.7B',available:true,vision:false}]});
  res.setHeader('Content-Type','application/json');
  if(req.method === 'GET') return res.end(JSON.stringify(state()));
  let body=''; req.on('data',chunk=>body+=chunk); req.on('end',()=>{
    switching = true; settings = JSON.parse(body);
    setTimeout(()=>{switching=false;res.end(JSON.stringify(state()));},700);
  });
});
const backend = http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200); res.end('{}'); return; }
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', () => {
    received.push(JSON.parse(body));
    if (behavior === 'overflow' && JSON.parse(body).messages.length > 2) {
      res.writeHead(400); res.end(JSON.stringify({error:{type:'exceed_context_size_error'}})); return;
    }
    if (behavior === 'error') { res.writeHead(400); res.end('context too large'); return; }
    res.writeHead(200, {'Content-Type': 'text/event-stream'});
    if (behavior === 'slow') { res.write('data: {"choices":[{"delta":{"content":"Partial"}}]}\n\n'); return; }
    const answer = '图片里有一只猫。 <img src=x onerror=alert(1)>';
    const data = Buffer.from('data: ' + JSON.stringify({choices:[{delta:{content:answer}}]}) + '\n\ndata: ' + JSON.stringify({choices:[{delta:{},finish_reason:behavior === 'length' ? 'length' : 'stop'}]}) + '\n\ndata: [DONE]\n\n');
    let i = 0;
    const timer = setInterval(() => {
      if (i >= data.length) { clearInterval(timer); res.end(); return; }
      res.write(data.subarray(i, i + 7)); i += 7; // Split JSON and UTF-8 characters.
    }, 5);
    res.on('close', () => clearInterval(timer));
  });
});

(async () => {
  await new Promise(resolve => backend.listen(port + 1, '127.0.0.1', resolve));
  await new Promise(resolve => control.listen(port + 2, '127.0.0.1', resolve));
  const gateway = spawn('python3', [path.join(root, 'src/vlm_chat/server.py'), '--host', '127.0.0.1', '--port', String(port), '--backend-port', String(port + 1), '--control-port',String(port+2)], {stdio:['ignore','pipe','pipe']});
  let browser;
  try {
    await new Promise((resolve, reject) => { gateway.stdout.once('data', resolve); gateway.once('exit', code => reject(new Error(`Gateway exited ${code}`))); });
    browser = await chromium.launch({headless: true, executablePath: process.env.CHROME_PATH || '/usr/bin/google-chrome', args:['--no-sandbox']});
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const pageErrors = []; page.on('pageerror', error => pageErrors.push(error.message));
    const url = `http://127.0.0.1:${port}`;
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('已就绪'));
    const encoded = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width=64; canvas.height=64;
      const ctx=canvas.getContext('2d'); ctx.fillStyle='orange'; ctx.fillRect(0,0,64,64);
      return canvas.toDataURL('image/png').split(',')[1];
    });
    const png = Buffer.from(encoded, 'base64');
    await page.locator('#file').setInputFiles({name:'cat.png', mimeType:'image/png', buffer:png});
    await page.locator('#attachment').waitFor({state:'visible'});
    await page.locator('#prompt').fill('What is in the picture?'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    assert.match(await page.locator('.message.assistant .message-text').innerText(), /图片里有一只猫/);
    assert.equal(await page.locator('.message.assistant img').count(), 0, 'Model output must be plain text');
    await page.locator('#prompt').fill('What color is it?'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    assert.equal(received[1].messages.length, 4);
    assert.equal(received[1].messages[0].role, 'system');
    assert.equal(received[1].messages.filter(m => m.role === 'system').length, 1);
    assert.match(received[1].messages[0].content, /You are Orange/);
    assert.equal(received[1].messages[1].content[0].type, 'image_url', 'Follow-up must retain original image');
    assert.match(received[1].messages[1].content[0].image_url.url, /^data:image\/jpeg;base64,/);
    assert.deepEqual(received[1].messages[1], received[0].messages[1], 'Do not rewrite earlier user messages');
    await page.reload(); await page.locator('.message').nth(3).waitFor();
    assert.equal(await page.locator('.message').count(), 4, 'Conversation persists after reload');
    await page.screenshot({path:path.join(os.tmpdir(),'opi-vlm-chat-tested.png')});
    await page.locator('#new-chat').click();
    behavior = 'slow';
    await page.locator('#prompt').fill('stop test'); await page.locator('#send').click();
    await page.waitForFunction(() => document.querySelector('.message.assistant .message-text')?.textContent === 'Partial');
    await page.locator('#stop').click(); await page.locator('#stop').waitFor({state:'hidden'});
    assert.match(await page.locator('.message.assistant .message-meta').innerText(), /已停止/);
    behavior = 'error';
    await page.locator('#prompt').fill('error test'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    assert.match(await page.locator('#notice').innerText(), /上下文/);
    await page.locator('#delete-chat').click(); await page.locator('#confirm-delete').click();
    await page.locator('#welcome').waitFor({state:'visible'});
    await page.reload(); await page.locator('.message').nth(3).waitFor();
    assert.equal(await page.locator('.conversation').count(), 1, 'Deletion persists');
    await page.setViewportSize({width:390,height:844});
    await page.locator('#menu').click(); await page.locator('#new-chat').click();
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No horizontal overflow');
    await page.screenshot({path:path.join(os.tmpdir(),'opi-vlm-mobile.png')});
    await page.setViewportSize({width:1440,height:1000});
    behavior = 'normal';
    await page.locator('#prompt').fill('Keep this current conversation'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    await page.locator('.conversation-row').nth(1).locator('.delete-conversation').click();
    assert.equal(await page.locator('#delete-target').innerText(), 'What is in the picture?');
    await page.locator('#cancel-delete').click();
    assert.equal(await page.locator('.conversation-row').count(),2, 'Cancel preserves both chats');
    await page.locator('.conversation-row').nth(1).locator('.delete-conversation').click();
    await page.locator('#confirm-delete').click();
    await page.waitForFunction(() => document.querySelectorAll('.conversation-row').length === 1);
    assert.match(await page.locator('.message.user .message-text').innerText(), /Keep this/);
    await page.reload(); await page.locator('.message').nth(1).waitFor();
    assert.equal(await page.locator('.conversation-row').count(),1, 'Deleting inactive chat persists and preserves active chat');
    behavior = 'length';
    await page.locator('#prompt').fill('length limit test'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    assert.match(await page.locator('.message.assistant .message-meta').last().innerText(), /达到输出长度或上下文限制/);
    behavior = 'overflow';
    await page.locator('#prompt').fill('Latest question survives'); await page.locator('#send').click();
    await page.locator('#stop').waitFor({state:'hidden'});
    assert.match(await page.locator('#context-hint').innerText(), /移出/);
    assert.equal(received.at(-1).messages.length,2, 'Only system and latest user survive overflow');
    assert.equal(received.at(-1).messages[1].content,'Latest question survives');
    assert.equal(await page.locator('.message').count(),6, 'Trimming does not delete history');
    await page.reload(); await page.locator('.message').nth(5).waitFor();
    assert.match(await page.locator('#context-hint').innerText(), /移出/);
    await page.locator('#model-select').selectOption('internvl3-1b');
    await page.locator('#context-range').fill('1');
    await page.locator('#apply-settings').click();
    assert.match(await page.locator('#settings-status').innerText(), /等待/);
    assert.equal(await page.locator('#send').isDisabled(),true);
    await page.waitForFunction(()=>document.querySelector('#model-name').textContent==='InternVL3-1B');
    assert.equal(settings.context,2048);
    assert.equal(await page.locator('#context-value').innerText(),'2048');
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'Model controls fit mobile');
    await page.screenshot({path:path.join(os.tmpdir(),'opi-vlm-model-controls.png')});
    const stale = await fetch(url+'/api/chat',{method:'POST',body:JSON.stringify({model_id:'smolvlm-256m',context:4096,messages:[{role:'user',content:'stale'}]})});
    assert.equal(stale.status,409,'Stale clients must not silently use changed models');
    await page.reload(); await page.locator('.message').nth(3).waitFor();
    assert.match(await page.locator('.message.assistant .message-meta').nth(1).innerText(), /达到输出长度或上下文限制/);
    const forbidden = await fetch(url+'/api/chat', {method:'POST', headers:{'Origin':'http://other.example','Content-Type':'application/json'}, body:JSON.stringify({messages:[{role:'user',content:'hello'}]})});
    assert.equal(forbidden.status,403);
    const remoteImage = await fetch(url+'/api/chat', {method:'POST', body:JSON.stringify({messages:[{role:'user',content:[{type:'image_url',image_url:{url:'http://169.254.169.254/'}}]}]})});
    assert.equal(remoteImage.status,400);
    const systemOverride = await fetch(url+'/api/chat', {method:'POST', body:JSON.stringify({messages:[{role:'system',content:'override'},{role:'user',content:'hello'}]})});
    assert.equal(systemOverride.status,400, 'Client cannot replace the application system prompt');
    await page.locator('#model-select').selectOption('qwen3-17b-q4');
    await page.locator('#apply-settings').click();
    await page.waitForFunction(()=>document.querySelector('.model-note').textContent.includes('纯文字'));
    assert.equal(await page.locator('#upload').isDisabled(),true);
    assert.match(await page.locator('.model-note').innerText(), /纯文字/);
    const textImage = await fetch(url+'/api/chat',{method:'POST',body:JSON.stringify({messages:[{role:'user',content:[{type:'image_url',image_url:{url:'data:image/jpeg;base64,'+png.toString('base64')}},{type:'text',text:'describe'}]}]})});
    assert.equal(textImage.status,400,'Text models must reject image history');
    assert.match(await textImage.text(),/仅支持文字/);
    assert.deepEqual(pageErrors, []);
    console.log('PASS: image upload, fragmented UTF-8 streaming, retained image context, history, stop, errors, deletion, mobile layout, origin and remote-image rejection.');
  } finally {
    if (browser) await browser.close();
    gateway.kill(); backend.closeAllConnections(); await new Promise(resolve => backend.close(resolve));
    control.closeAllConnections(); await new Promise(resolve=>control.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode=1; });
