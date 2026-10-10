// Live text-only UI acceptance; does not score model quality.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
(async () => {
  const out = path.resolve(__dirname, '../outputs/text-web-20261010');
  fs.mkdirSync(out, {recursive:true});
  const browser = await chromium.launch({headless:true, executablePath:'/usr/bin/google-chrome', args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    page.setDefaultTimeout(180000);
    await page.goto(process.env.VLM_URL || 'http://192.168.88.29:8080/');
    await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('已就绪'));
    assert.equal(await page.locator('#upload').isDisabled(), true);
    assert.match(await page.locator('#model-name').innerText(), /Qwen/);
    const observations=[];
    for (const prompt of ['你好，我叫阿沐，今天有点累。用一两句话陪我聊聊。','你还记得我叫什么吗？']) {
      await page.locator('#prompt').fill(prompt);
      await page.locator('#send').click();
      await page.locator('#stop').waitFor({state:'hidden'});
      const answer = await page.locator('.message.assistant .message-text').last().innerText();
      const meta = await page.locator('.message.assistant .message-meta').last().innerText();
      assert.ok(answer.trim());
      assert.ok(!/失败|停止|达到输出长度/.test(meta),meta);
      observations.push({prompt,answer,meta});
    }
    await page.screenshot({path:path.join(out,'chat.png')});
    fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(observations,null,2));
    console.log(JSON.stringify(observations));
    console.log('PASS: live text model, streamed two-turn chat and disabled image upload.');
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});
