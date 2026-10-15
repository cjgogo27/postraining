import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';

export default defineConfig({
  site: 'https://cjgogo27.github.io',
  base: '/postraining',
  trailingSlash: 'always',
  markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }) },
  integrations: [starlight({
    title: '《听懂大模型后训练》',
    description: '从监督信号到策略更新：大模型后训练的数据、算法、系统与验证。',
    locales: { root: { label: '简体中文', lang: 'zh-CN' } },
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/cjgogo27/postraining' }],
    editLink: { baseUrl: 'https://github.com/cjgogo27/postraining/edit/main/src/content/docs/' },
    sidebar: [
      { label: '从这里开始', items: [{ autogenerate: { directory: '00-reading-guides' } }] },
      { label: '第 1 章 系统总览', items: [{ autogenerate: { directory: '01-system-overview' } }] },
      { label: '第 2 章 数据与监督接口', items: [{ autogenerate: { directory: '02-data-supervision' } }] },
      { label: '第 3 章 监督与偏好优化', items: [{ autogenerate: { directory: '03-supervised-preference' } }] },
      { label: '第 4 章 在线强化学习与蒸馏', items: [{ autogenerate: { directory: '04-online-rl-distillation' } }] },
      { label: '第 5 章 评测、系统与诊断', items: [{ autogenerate: { directory: '05-evaluation-diagnosis' } }] },
      { label: '第 6 章 工具与实践', items: [{ autogenerate: { directory: '06-tools-practice' } }] },
      { label: '附录', items: [{ autogenerate: { directory: '07-appendix' } }] },
    ],
    tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
    pagination: true,
    customCss: ['./src/styles/custom.css'],
  })],
});
