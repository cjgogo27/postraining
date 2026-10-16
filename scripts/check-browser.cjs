// Optional browser QA: requires Playwright and a running local preview. No model/API calls.
const fs=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const repo=path.resolve(__dirname,'..');
const folder=path.resolve(repo,process.env.BROWSER_CHECK_OUT || 'runs/browser-checks');
const base=process.env.BROWSER_CHECK_BASE || 'http://127.0.0.1:4325/postraining/';
const checks=[],errors=[];
const run=async()=>{
  await fs.mkdir(folder,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
  try {
    for(const viewport of [{width:1280,height:900},{width:390,height:844}]) {
      const context=await browser.newContext({viewport,reducedMotion:'reduce'});
      const page=await context.newPage();
      page.on('pageerror',error=>errors.push(error.message));
      const go=async(route)=>{await page.goto(base+route,{waitUntil:'networkidle'});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,`Page overflow: ${route}`);};
      const width=viewport.width;
      await go('03-supervised-preference/04-dpo-objective/?reader=engineer');
      assert.equal(await page.locator('[data-reader-select]').inputValue(),'工程师');
      assert.equal(await page.locator('[data-reader-role][open]').getAttribute('data-reader-role'),'工程师');
      await page.locator('[data-reader-select]').selectOption('学生');
      await page.goto(base+'03-supervised-preference/04-dpo-objective/',{waitUntil:'networkidle'});
      assert.equal(await page.locator('[data-reader-select]').inputValue(),'学生');
      const answer=page.locator('.section-qa-answer details').first();
      await answer.locator('summary').click();assert.equal(await answer.getAttribute('open')!==null,true);
      await answer.locator('summary').click();assert.equal(await answer.getAttribute('open'),null);
      checks.push({viewport:width,name:'role_query_storage_and_answer_toggle',status:'PASS'});
      const dpo=page.locator('[data-dpo-calculator]');
      assert.match(await dpo.locator('output').innerText(),/0.598139/);
      await dpo.locator('[data-preset="equal"]').click();assert.match(await dpo.locator('output').innerText(),/0.693147/);
      await dpo.locator('[data-preset="bothDown"]').click();assert.match(await dpo.locator('output').innerText(),/margin=0.160944/);
      await dpo.locator('[data-field="pc"]').fill('1');assert.doesNotMatch(await dpo.locator('output').innerText(),/margin=/);
      await dpo.locator('[data-preset="hand"]').focus();await page.keyboard.press('Enter');
      assert.match(await dpo.locator('output').innerText(),/0.598139/);
      await dpo.screenshot({path:path.join(folder,`dpo-${width}.png`)});
      checks.push({viewport:width,name:'dpo_presets_validation_reset_and_keyboard',status:'PASS'});
      await go('02-data-supervision/02-template-and-mask/');
      const token=page.locator('[data-token-explorer]');
      assert.equal(await token.locator('[data-row="shift"]').isVisible(),false);
      await token.locator('[data-view="mask"]').click();assert.equal(await token.locator('[data-row="mask"]').isVisible(),true);
      await token.locator('[data-index="8"]').click();assert.match(await token.locator('output').innerText(),/label=256/);
      await token.locator('[data-index="0"]').click();assert.match(await token.locator('output').innerText(),/loss_mask=0/);
      await token.locator('[data-index="7"]').click();assert.match(await token.locator('output').innerText(),/0…7/);
      await token.screenshot({path:path.join(folder,`token-${width}.png`)});
      checks.push({viewport:width,name:'token_views_shift_eos_attention_and_loss_mask',status:'PASS'});
      await go('04-online-rl-distillation/03-grpo-and-rlvr/');
      const rl=page.locator('[data-rl-explorer]');
      for(const index of [0,1,2,3]) await rl.locator(`[data-step="${index}"]`).click();
      assert.equal(await rl.locator('[data-probability]').last().innerText(),'0.2749');
      await rl.screenshot({path:path.join(folder,`rl-${width}.png`)});
      for(const scenario of ['wrong','right']){
        await rl.locator('select').selectOption(scenario);await rl.locator('[data-step="3"]').click();
        assert.equal(await rl.locator('[data-probability]').first().innerText(),'0.2500');
        assert.match(await rl.locator('output').innerText(),/没有组内区分信号/);
      }
      await rl.locator('[data-reset]').click();assert.equal(await rl.locator('select').inputValue(),'mixed');
      assert.equal(await rl.locator('[data-step="0"]').getAttribute('aria-pressed'),'true');
      checks.push({viewport:width,name:'rl_four_steps_constant_groups_and_reset',status:'PASS'});
      await go('06-tools-practice/02-offline-sft-dpo/');
      const output=page.locator('[data-output-comparison]');
      assert.equal(await output.locator('option').count(),23);
      const last=await output.locator('option').last().getAttribute('value');
      await output.locator('select').selectOption(last);
      assert.equal(await output.locator('[data-sample]:visible').count(),1);
      assert.equal(await output.locator('[data-sample]:visible').getAttribute('data-sample'),last);
      await output.screenshot({path:path.join(folder,`outputs-${width}.png`)});
      checks.push({viewport:width,name:'fixed_23_sample_output_selection',status:'PASS'});
      let menuOpened=false;
      if(await page.locator('starlight-theme-select select:visible').count()===0){
        await page.getByRole('button',{name:'菜单',exact:true}).click();menuOpened=true;
      }
      const theme=page.locator('starlight-theme-select select:visible').first();
      for(const name of ['light','dark']) {
        await theme.selectOption(name);assert.equal(await page.locator('html').getAttribute('data-theme'),name);
      }
      if(menuOpened) await page.getByRole('button',{name:'菜单',exact:true}).click();
      await output.screenshot({path:path.join(folder,`outputs-dark-${width}.png`)});
      checks.push({viewport:width,name:'theme_switch_and_no_page_horizontal_overflow',status:'PASS'});
      await go('00-reading-guides/05-essential-foundations/');
      assert.match(await page.locator('main').innerText(),/0.2203/);
      await context.close();
    }
    assert.deepEqual(errors,[]);
    // No JS fallback must retain primary content and all expandable role questions.
    const nojs=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}});
    const page=await nojs.newPage();await page.goto(base+'02-data-supervision/02-template-and-mask/');
    assert.equal(await page.locator('[data-reader-role]').count(),3);
    assert.equal(await page.locator('[data-token-explorer] tbody tr').count(),4);
    checks.push({name:'no_javascript_content_fallback',status:'PASS'});
    await nojs.close();
    const report={status:'PASS',browser:await browser.version(),headless:true,viewports:[1280,390],checks,count:checks.length,page_errors:errors,scope:'Local built site, four teaching widgets, role questions, themes, overflow and JS fallback; not every browser, search or every article screenshot'};
    await fs.writeFile(path.join(folder,'report.json'),JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify(report));
  } finally {await browser.close();}
};
run().catch(error=>{console.error(error);process.exitCode=1;});
