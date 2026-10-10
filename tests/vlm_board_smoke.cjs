// Live board acceptance: photo understanding, visual follow-up, persisted history.
// Requires VLM_TEST_IMAGE and optionally VLM_URL (default board LAN address).
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

(async () => {
  const url = process.env.VLM_URL || 'http://192.168.88.29:8080';
  const photo = process.env.VLM_TEST_IMAGE;
  if (!photo || !fs.existsSync(photo)) throw new Error('Set VLM_TEST_IMAGE to a real test photo');
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || '/usr/bin/google-chrome',args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    page.setDefaultTimeout(360000);
    const observations = [];
    const errors=[]; page.on('pageerror',e => errors.push(e.message));
    await page.goto(url);
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('已就绪'));
    const model = await page.locator('#model-name').innerText();
    for (const prompt of ['我的宠物兔叫露娜。', '我的宠物叫什么？']) {
      await page.locator('#prompt').fill(prompt);
      const start = Date.now();
      await page.locator('#send').click();
      await page.locator('#stop').waitFor({state:'hidden'});
      const answer = await page.locator('.message.assistant .message-text').last().innerText();
      const result = {prompt, answer, seconds:(Date.now()-start)/1000};
      observations.push(result); console.log(JSON.stringify(result));
      assert.ok(answer.trim(), 'Live Chinese response must not be empty');
    }
    await page.locator('#new-chat').click();
    await page.locator('#file').setInputFiles(photo);
    await page.locator('#attachment').waitFor({state:'visible'});
    for (const prompt of ['Describe this image in two short sentences.', 'What color is the animal in the picture?']) {
      await page.locator('#prompt').fill(prompt);
      const start=Date.now();
      await page.locator('#send').click();
      await page.locator('#stop').waitFor({state:'hidden'});
      const text=await page.locator('.message.assistant .message-text').last().innerText();
      const meta=await page.locator('.message.assistant .message-meta').last().innerText();
      assert.ok(text.trim(), 'Model should answer');
      assert.ok(!/失败|停止/.test(meta),meta);
      const result = {prompt,answer:text,seconds:(Date.now()-start)/1000};
      observations.push(result); console.log(JSON.stringify(result));
    }
    await page.reload(); await page.locator('.message').nth(3).waitFor();
    assert.equal(await page.locator('.message-image').count(),1);
    assert.equal(await page.locator('.message.assistant').count(),2);
    assert.deepEqual(errors,[]);
    await page.waitForFunction(() => document.querySelector('#status').textContent.includes('已就绪'));
    await page.locator('.message.assistant').last().scrollIntoViewIfNeeded();
    await page.waitForTimeout(400);
    await page.screenshot({path:path.join(os.tmpdir(),'opi-vlm-live-chat.png')});
    const overflow = await fetch(url + '/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:[{role:'user',content:'x '.repeat(5999)},
        {role:'assistant',content:'OK'}, {role:'user',content:'y '.repeat(5999)}]})});
    const overflowBody = await overflow.json();
    assert.equal(overflow.status,400, 'Oversized context must be explicitly rejected');
    assert.match(overflowBody.detail, /context|exceed|too long/i);
    const health = await (await fetch(url+'/api/health')).json();
    assert.equal(health.ready,true, 'Model must remain ready after rejected input');
    fs.writeFileSync(path.join(os.tmpdir(),'opi-vlm-live-acceptance.json'), JSON.stringify({model,observations,
      overflow:{status:overflow.status,body:overflowBody},health},null,2));
    console.log('PASS: Chinese turns, real photo, visual follow-up, persisted history, context rejection, healthy service and screenshot.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
