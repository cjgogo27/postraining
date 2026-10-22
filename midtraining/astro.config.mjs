import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  site:'https://cjgogo27.github.io',base:'/postraining/midtraining',trailingSlash:'always',
  outDir:fileURLToPath(new URL('../dist/midtraining/',import.meta.url)),
  cacheDir:fileURLToPath(new URL('../.astro-midtraining/',import.meta.url)),
  markdown:{processor:unified({remarkPlugins:[remarkMath],rehypePlugins:[rehypeKatex]})},
  integrations:[starlight({
    title:'《听懂大模型中训练》',description:'从已有基座到领域与上下文适配：数据、训练、基础设施和验收。',
    locales:{root:{label:'简体中文',lang:'zh-CN'}},
    social:[{icon:'github',label:'GitHub',href:'https://github.com/cjgogo27/postraining/tree/main/midtraining'}],
    editLink:{baseUrl:'https://github.com/cjgogo27/postraining/edit/main/midtraining/src/content/docs/'},
    sidebar:[
      {label:'从这里开始',items:[{autogenerate:{directory:'00-reading-guides'}}]},
      {label:'第1章 系统总览',items:[{autogenerate:{directory:'01-system-overview'}}]},
      {label:'第2章 数据与任务',items:[{autogenerate:{directory:'02-data-tasks'}}]},
      {label:'第3章 能力适配',items:[{autogenerate:{directory:'03-capability-adaptation'}}]},
      {label:'第4章 训练与基础设施',items:[{autogenerate:{directory:'04-training-infrastructure'}}]},
      {label:'第5章 评测与诊断',items:[{autogenerate:{directory:'05-evaluation-diagnosis'}}]},
      {label:'第6章 工具与实践',items:[{autogenerate:{directory:'06-tools-practice'}}]},
      {label:'附录',items:[{autogenerate:{directory:'07-appendix'}}]},
      {label:'另读《听懂大模型后训练》',link:'https://cjgogo27.github.io/postraining/'},
    ],
    tableOfContents:{minHeadingLevel:2,maxHeadingLevel:3},pagination:true,customCss:['./src/styles/custom.css'],
  })],
});
