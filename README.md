# MARINER: A 3E-Driven Benchmark for Fine-Grained Perception and Complex Reasoning in Open-Water Environments

## 🌟 Overview  🌟 概述

- `docs/index.html`
  页面主体，标题、作者、摘要、方法、结果、资源链接、BibTeX 都在这里改。
- `docs/static/css/style.css`
  页面样式。
- `docs/static/img/`
  所有占位图都在这里，后面直接用你的真实图片替换即可。
- `docs/.nojekyll`
  让 GitHub Pages 按静态文件方式直接发布。

## 先改哪里

优先改这些位置：

1. `docs/index.html` 里的论文标题、作者、单位、venue。
2. 顶部按钮链接：`Paper`、`arXiv`、`Code`、`Dataset`、`Demo Video`、`Appendix`。
3. 摘要段落。
4. `Overview / Method / Results / FAQ / Acknowledgement` 的占位文字。
5. `BibTeX` 和 `Contact`。

## 图片替换建议

当前占位图文件如下：

- `docs/static/img/hero-placeholder.svg`
- `docs/static/img/overview-placeholder.svg`
- `docs/static/img/method-placeholder.svg`
- `docs/static/img/results-placeholder-a.svg`
- `docs/static/img/results-placeholder-b.svg`
- `docs/static/img/video-placeholder.svg`

最简单的替换方式：

1. 保持文件名不变，直接用你的图片覆盖这些文件。
2. 或者新增图片文件，然后在 `docs/index.html` 里把对应 `src` 改掉。

推荐格式：

- 普通展示图：`.png` 或 `.jpg`
- 线稿/流程图：`.svg`
- teaser 或 demo 封面：宽图优先

## GitHub Pages 发布

仓库推送到 GitHub 后：

1. 打开仓库 `Settings`
2. 进入 `Pages`
3. 选择 `Deploy from a branch`
4. Branch 选 `main`
5. Folder 选 `/docs`
6. 保存

发布后地址一般是：

`https://你的用户名.github.io/仓库名/`

## 后续可继续补的内容

- 嵌入 YouTube / Bilibili 视频
- 加作者主页链接
- 加 Google Scholar / GitHub 图标
- 加结果表格或 benchmark leaderboard
- 加 poster、slides、supplementary 下载入口

## 本地预览

这是纯静态页面，直接双击 `docs/index.html` 就能看。

如果你后面要我继续，我可以直接帮你做这几类增强：

- 改成更接近顶会论文项目页的排版风格
- 把按钮和作者区换成带图标的版本
- 加视频嵌入
- 加结果表格
- 加中英文双语版本
