# CST-MIL local demonstration

[简体中文安装说明](../../README.md#快速启动) · [English](../../README.en.md#quick-start)

本应用展示最新五来源全量语义模型的风险类别、动态讨论主题和原文证据。它使用React、Express、串行推理队列、常驻Python worker和SQLite；不在启动时训练或调用付费模型API。

先按根目录README建立 `cst_mil/.venv`，运行 `python -m cst_mil.deployment setup` 准备经过校验的模型资产。然后在本目录运行：

```powershell
npm ci
npm start
```

浏览器地址为 **http://127.0.0.1:3210**。以后只需 `npm start`。模型身份随健康接口、结果、历史和导出记录；分数、Top-3与主题均为核查线索，不是自动裁定。

开发检查：

```powershell
npm run build
npx tsx --test tests/ui-source.test.ts tests/portable-deployment.test.ts
```

涉及真实模型的集成与浏览器测试需先安装资产。内置校园案例为编写功能示例；历史算法对比页展示早期SCCD实验，不代表最新发布模型的性能。
