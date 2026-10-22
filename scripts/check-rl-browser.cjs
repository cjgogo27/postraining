// Optional Playwright/Edge checks; no production or model dependency.
const fs=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const folder=path.resolve(process.env.BROWSER_CHECK_OUT||'runs/rl-browser');
const base=process.env.BROWSER_CHECK_BASE||'http://127.0.0.1:4327/postraining/';
const checks=[],errors=[];
(async()=>{
  await fs.mkdir(folder,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
  try {
    for(const viewport of [{width:1280,height:900},{width:390,height:844}]) {
      const context=await browser.newContext({viewport,reducedMotion:'reduce'});
      const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
      const go=async route=>{
        const response=await page.goto(base+route,{waitUntil:'networkidle'});
        assert.equal(response.status(),200);
        assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`overflow ${route}`);
        assert.equal(await page.locator('.katex-error').count(),0);
      };
      const mark=name=>checks.push({viewport:viewport.width,name,status:'PASS'});
      await go('04-online-rl-distillation/02-ppo-advantage-and-kl/');
      const widget=page.locator('[data-ppo-explorer]');
      for(const [id,answer] of [['high-positive','1.20'],['high-negative','-1.50'],['low-positive','0.50'],['low-negative','-0.80']]) {
        await widget.locator(`[data-case="${id}"]`).click();
        assert.ok((await widget.locator('output').innerText()).includes(`最大化的目标=${answer}`));
      }
      await widget.locator('[data-case="high-negative"]').focus();await page.keyboard.press('Enter');
      assert.equal(await widget.locator('select').inputValue(),'-1');
      await widget.locator('input').fill('1.1');await widget.locator('input').dispatchEvent('input');
      assert.match(await widget.locator('output').innerText(),/目标=-1.10/);
      await widget.screenshot({path:path.join(folder,`ppo-${viewport.width}-light.png`)});
      mark('ppo_four_presets_slider_keyboard_and_no_page_overflow');
      const themes=page.locator('starlight-theme-select select:visible');let menu=false;
      if(await themes.count()===0){await page.getByRole('button',{name:'菜单',exact:true}).click();menu=true;}
      await page.locator('starlight-theme-select select:visible').first().selectOption('dark');
      if(menu) await page.getByRole('button',{name:'菜单',exact:true}).click();
      assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
      await widget.screenshot({path:path.join(folder,`ppo-${viewport.width}-dark.png`)});
      const details=page.locator('.section-qa-answer details').first();
      await details.locator('summary').click();assert.notEqual(await details.getAttribute('open'),null);
      await details.locator('summary').click();assert.equal(await details.getAttribute('open'),null);
      mark('ppo_theme_and_role_answer_toggle');
      await go('04-online-rl-distillation/03-grpo-and-rlvr/');
      const gspo=page.locator('details').filter({has:page.locator('summary',{hasText:'GSPO的几何平均'})});
      await gspo.locator('summary').click();assert.notEqual(await gspo.getAttribute('open'),null);
      assert.equal(await gspo.locator('.katex').count(),1);assert.match(await gspo.innerText(),/0.725/);
      mark('gspo_details_formula_and_hand_value');
      await go('00-reading-guides/06-rl-map/');
      assert.match(await page.locator('main').innerText(),/0.3和0.8/);mark('online_offline_reading_map');
      await go('07-appendix/e-agentic-multi-agent-rl/');
      assert.match(await page.locator('main').innerText(),/0.09375/);
      const baseline=page.locator('details').filter({has:page.locator('summary',{hasText:'为什么基线'})});
      await baseline.locator('summary').click();assert.notEqual(await baseline.getAttribute('open'),null);
      mark('agentic_marl_credit_and_details');
      await go('07-appendix/f-rl-reading-list/');
      assert.equal(await page.locator('main h2').filter({hasText:/^\d+\./}).count(),10);
      assert.match(await page.locator('main').innerText(),/2026-10-08/);mark('ten_primary_readings_and_dates');
      await context.close();
    }
    const context=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
    const page=await context.newPage();await page.goto(base+'04-online-rl-distillation/02-ppo-advantage-and-kl/');
    assert.match(await page.locator('[data-ppo-explorer] output').innerText(),/目标=1.20/);
    assert.match(await page.locator('main').innerText(),/0.78/);
    checks.push({name:'no_js_static_hand_case_and_gae',status:'PASS'});
    await context.close();assert.deepEqual(errors,[]);
    const report={status:'PASS',browser:await browser.version(),checks,page_errors:errors,
      scope:'Two viewports, local build; no real phone/Safari/Firefox, model inference, performance or all-page audit'};
    await fs.writeFile(path.join(folder,'report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
