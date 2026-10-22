// Optional visual/interaction checks for the integrated teaching path.
const fs=require('node:fs/promises'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.BROWSER_CHECK_BASE||'http://127.0.0.1:4328/postraining/';
const folder=path.resolve(process.env.BROWSER_CHECK_OUT||'runs/lifecycle-browser');
const checks=[],errors=[];
(async()=>{
  await fs.mkdir(folder,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
  try{
    for(const viewport of [{width:1280,height:900},{width:390,height:844}]){
      const context=await browser.newContext({viewport,reducedMotion:'reduce'}),page=await context.newPage();
      page.on('pageerror',e=>errors.push(e.message));
      const width=async()=>assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,'page overflow');
      const go=async route=>{const response=await page.goto(base+route,{waitUntil:'networkidle'});assert.equal(response.status(),200);await width();assert.equal(await page.locator('.katex-error').count(),0);};
      const pass=name=>checks.push({viewport:viewport.width,name,status:'PASS'});
      // Element screenshots can include the fixed navigation when the element
      // is taller than the viewport. Hide overlays for capture only, not tests.
      const capture=async(target,name)=>{
        await page.evaluate(()=>{for(const element of document.querySelectorAll('*')){
          if(['fixed','sticky'].includes(getComputedStyle(element).position))element.setAttribute('data-capture-overlay','');
        }});
        await target.screenshot({path:path.join(folder,name),style:'[data-capture-overlay]{visibility:hidden !important;}'});
      };
      await go('01-system-overview/03-posttraining-system-architecture/');
      const map=page.locator('[data-system-map]');
      assert.equal(await map.locator('.paths>section').count(),4);
      assert.equal(await map.locator('.feedback>p').count(),2);
      assert.match(await map.innerText(),/权重传输/);
      assert.match(await page.locator('main').innerText(),/28个位置/);
      await capture(map,`system-map-${viewport.width}-light.png`);
      pass('four_learning_signals_two_feedbacks_and_batch_ledger');
      let menu=false;
      if(await page.locator('starlight-theme-select select:visible').count()===0){await page.getByRole('button',{name:'菜单',exact:true}).click();menu=true;}
      await page.locator('starlight-theme-select select:visible').first().selectOption('dark');
      if(menu)await page.getByRole('button',{name:'菜单',exact:true}).click();
      await capture(map,`system-map-${viewport.width}-dark.png`);
      pass('system_map_dark_theme');
      await go('00-reading-guides/07-rl-foundations/');
      assert.match(await page.locator('main').innerText(),/V=0.6/);assert.ok(await page.locator('main .katex').count()>=1);
      pass('mdp_return_bellman_foundations');
      await go('00-reading-guides/04-stage-connections/');
      assert.match(await page.locator('main').innerText(),/parent_checkpoint_hash/);pass('stage_contract_and_manifest');
      await go('05-evaluation-diagnosis/04-training-rollout-orchestration/');
      assert.match(await page.locator('main').innerText(),/本地loss总和乘2\/6/);
      const budget=page.locator('details').filter({has:page.locator('summary',{hasText:'从状态字节到资源预算'})});
      await budget.locator('summary').click();assert.notEqual(await budget.getAttribute('open'),null);
      assert.match(await budget.innerText(),/128MiB/);await width();
      await capture(budget,`budget-${viewport.width}.png`);
      pass('parallel_layout_global_reduction_and_budget_details');
      await go('06-tools-practice/04-stage-handoff-and-resume/');
      assert.match(await page.locator('main').innerText(),/0.05694594/);
      assert.match(await page.locator('main').innerText(),/两个真实本机Gloo进程/);
      const answer=page.locator('.section-qa-answer details').first();
      await answer.locator('summary').click();assert.notEqual(await answer.getAttribute('open'),null);
      pass('actual_lifecycle_and_two_rank_scope_role_answer');
      await context.close();
    }
    const context=await browser.newContext({javaScriptEnabled:false,viewport:{width:390,height:844}}),page=await context.newPage();
    await page.goto(base+'01-system-overview/03-posttraining-system-architecture/');
    assert.equal(await page.locator('[data-system-map] .paths>section').count(),4);
    checks.push({name:'no_js_lifecycle_map_remains_readable',status:'PASS'});await context.close();
    assert.deepEqual(errors,[]);
    const report={status:'PASS',browser:await browser.version(),count:checks.length,checks,page_errors:errors,
      scope:'Local built site, desktop/mobile viewports and selected teaching pages; no real device or all-browser audit'};
    await fs.writeFile(path.join(folder,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
