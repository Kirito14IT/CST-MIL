import { test, expect, type Page } from '@playwright/test';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { loadDeployment } from '../../config/deployment.mjs';

const deployment = loadDeployment();
const screenshotDir = resolve(process.env.CAMPUS_SCREENSHOT_DIR ?? 'test-results/unified-browser/screenshots');

async function capture(page: Page, name: string) {
  await page.evaluate(() => {
    if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
    window.scrollTo({ top: 0, behavior: 'instant' });
  });
  await page.screenshot({ path: resolve(screenshotDir, name), fullPage: true, animations: 'disabled' });
}

test('真实模型：校园评论到核查、导出、历史及指标展示', async ({ page, request }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await expect.poll(async () => (await (await request.get('/api/health')).json()).worker.status, { timeout: 95_000 }).toBe('ready');
  const ready = await (await request.get('/api/health')).json();
  expect(ready.worker.modelHash).toBe(deployment.modelSha256);
  expect(ready.worker.modelVersion).toBe(deployment.modelVersion);
  expect(ready.worker.deploymentHash).toBe(deployment.deploymentHash);
  if (deployment.schemaVersion === 2) expect(ready.worker.encoderWarmed).toBe(true);
  await mkdir(screenshotDir, { recursive: true });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '短文本主题与风险分析' })).toBeVisible();
  await capture(page, '01-新建分析.png');
  const title = `校园作业核查-${Date.now()}`;
  await page.getByLabel('会话名称', { exact: false }).fill(title);
  await page.getByLabel('评论内容', { exact: true }).fill('同学A：这周的小组作业我们怎么分工？\n同学B：我来整理数据，周三之前发给大家。\n同学C：那我负责把结果画成图。\n同学A：好，周四我们一起检查。');
  await page.getByRole('button', { name: '校验并预览', exact: true }).click();
  await expect(page.getByText('校验通过', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: '开始分析', exact: true }).click();
  await expect(page.getByRole('heading', { name: deployment.schemaVersion === 2 ? '动态讨论主题' : '隐主题发现', exact: true })).toBeVisible({ timeout: 125_000 });
  if (deployment.schemaVersion === 2) {
    await expect(page.getByRole('heading', { name: '固定风险类别', exact: true })).toBeVisible();
    await expect(page.getByText('SCCD会话迁移参考', { exact: true })).toBeVisible();
    await expect(page.getByText('最高消息风险线索强度', { exact: true }).first()).toBeVisible();
  }
  await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Top-3 候选证据', exact: true })).toBeVisible();
  await page.locator('.evidence-card').first().click();
  await expect(page.locator('.conversation-message.focused')).toBeVisible();
  await page.getByRole('radio', { name: '未见风险', exact: true }).check();
  const note = '已查看完整上下文，内容是正常的小组作业分工。';
  await page.getByLabel('核查备注', { exact: true }).fill(note);
  await page.getByRole('button', { name: '保存核查记录', exact: true }).click();
  await expect(page.getByText('有尚未保存的修改', { exact: true })).toHaveCount(0);
  await capture(page, '02-分析结果与人工核查.png');
  const jsonDownloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'JSON', exact: true }).click();
  const jsonDownload = await jsonDownloadPromise;
  expect(jsonDownload.suggestedFilename()).toMatch(/\.json$/);
  await jsonDownload.saveAs(resolve(screenshotDir, '示例结果.json'));
  const csvDownloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: '导出 CSV', exact: true }).click();
  const csvDownload = await csvDownloadPromise;
  expect(csvDownload.suggestedFilename()).toMatch(/\.csv$/);
  await csvDownload.saveAs(resolve(screenshotDir, '示例消息.csv'));
  await page.getByRole('button', { name: '历史记录', exact: true }).click();
  await expect(page.getByRole('button', { name: title, exact: true })).toBeVisible();
  await capture(page, '03-历史记录.png');
  await page.getByRole('button', { name: title, exact: true }).click();
  await expect(page.getByLabel('核查备注', { exact: true })).toHaveValue(note);
  await page.reload();
  await expect(page.getByLabel('核查备注', { exact: true })).toHaveValue(note);
  await page.getByRole('button', { name: '算法对比', exact: true }).click();
  await expect(page.getByRole('heading', { name: '宏平均 F1', exact: true })).toBeVisible();
  await expect(page.getByRole('rowheader', { name: 'CST-MIL', exact: false }).first()).toBeVisible();
  await capture(page, '04-SCCD算法对比.png');
  await page.getByRole('tab', { name: '候选证据定位', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Top-3 命中率', exact: true })).toBeVisible();
  await page.getByRole('tab', { name: '计算与资源成本', exact: true }).click();
  await capture(page, '05-计算成本.png');
  expect(errors).toEqual([]);
});

test('schema2多语言、混合内容及反引用原文与身份可查看', async ({ page, request }) => {
  test.skip(deployment.schemaVersion !== 2, 'Schema2 presentation only');
  const messages = [
    { id: 'a', text: 'Vamos estudar juntos amanhã na biblioteca.', order: 0 },
    { id: 'b', text: 'We can review the project results together.', parentId: 'a', order: 1 },
    { id: 'c', text: '有人说“你每次都拖后腿”，这种针对个人的说法不合适。', order: 2 },
    { id: 'd', text: '同意，请停止人身攻击，讨论具体分工。', parentId: 'c', order: 3 },
    { id: 'e', text: '不要支付来历不明的奖学金保证金，请向学校核实。', order: 4 },
  ];
  await expect.poll(async () => (await (await request.get('/api/health')).json()).worker.status, { timeout: 95_000 }).toBe('ready');
  const created = await request.post('/api/sessions', { data: { sessions: [{ title: '多语言与上下文呈现', source: 'manual', messages }] } });
  expect(created.status()).toBe(201);
  const session = (await created.json()).sessions[0];
  const queued = await request.post('/api/analyses', { data: { sessionIds: [session.id] } });
  expect(queued.status()).toBe(202);
  const id = (await queued.json()).analyses[0].id;
  await expect.poll(async () => (await (await request.get(`/api/analyses/${id}`)).json()).analysis.status, { timeout: 125_000 }).toBe('succeeded');
  const detail = await (await request.get(`/api/analyses/${id}`)).json();
  expect(detail.analysis.modelHash).toBe(deployment.modelSha256);
  expect(detail.analysis.modelVersion).toBe(deployment.modelVersion);
  await page.goto(`/#analysis/${id}`);
  await expect(page.getByRole('heading', { name: '固定风险类别', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: '动态讨论主题', exact: true })).toBeVisible();
  await expect(page.locator('.category-card')).toHaveCount(3);
  for (const message of messages) await expect(page.locator('.conversation-message').filter({ hasText: message.text })).toBeVisible();
  await page.locator('.model-details summary').click();
  await expect(page.locator('.model-details')).toContainText(deployment.modelVersion);
  await expect(page.locator('.model-details')).toContainText(deployment.modelSha256);
  await expect(page.locator('.model-details')).toContainText(deployment.deploymentHash);
  await capture(page, '09-多语言原文与实际模型身份.png');
});

test('上传预览、校验错误、示例与1024像素布局', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto('/');
  await page.getByRole('tab', { name: '上传文件', exact: true }).click();
  await page.locator('input[type=file]').setInputFiles({ name: '重复编号.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ title: '无效示例', messages: [{ id: 'same', text: '第一句' }, { id: 'same', text: '第二句' }] })) });
  await page.getByRole('button', { name: '校验并预览', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('重复');
  await page.locator('input[type=file]').setInputFiles(resolve('examples/校园评论示例.csv'));
  await page.getByRole('button', { name: '校验并预览', exact: true }).click();
  await expect(page.getByText('校验通过', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: /查看会话/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await capture(page, '06-1024文件预览.png');
  await page.getByRole('tab', { name: '校园示例', exact: true }).click();
  await expect(page.locator('.example-card').first()).toBeVisible();
  await page.locator('.example-card').first().click();
  await page.getByRole('button', { name: '校验并预览', exact: true }).click();
  await expect(page.getByText('校验通过', { exact: true })).toBeVisible();
  await capture(page, '07-校园示例.png');
});

test('真实模型未知词输入仍可查看和人工核查', async ({ page, request }) => {
  await expect.poll(async () => (await (await request.get('/api/health')).json()).worker.status, { timeout: 95_000 }).toBe('ready');
  await page.goto('/');
  await page.getByLabel('会话名称', { exact: false }).fill('零主题边界验证');
  await page.getByLabel('评论内容', { exact: true }).fill('zzzxqvjk');
  await page.getByRole('button', { name: '校验并预览', exact: true }).click();
  await page.getByRole('button', { name: '开始分析', exact: true }).click();
  await expect(page.getByRole('heading', { name: '人工核查', exact: true })).toBeVisible({ timeout: 125_000 });
  await expect(page.getByLabel('核查备注', { exact: true })).toBeVisible();
  const id = page.url().split('/').at(-1)!;
  const detail = await (await request.get(`/api/analyses/${id}`)).json();
  expect(detail.analysis.status).toBe('succeeded');
  if (deployment.schemaVersion === 1) expect(detail.result.topics).toEqual([]);
  else {
    expect(detail.result.schemaVersion).toBe(2); expect(detail.result.riskCategories).toHaveLength(3);
    if (detail.result.topics.length === 0) await expect(page.getByRole('heading', { name: '未形成清晰主题', exact: true })).toBeVisible();
    await expect(page.locator('.conversation-message').filter({ hasText: 'zzzxqvjk' })).toBeVisible();
    await expect(page.locator('.message-category-scores').first()).not.toBeEmpty();
    await expect(page.getByRole('heading', { name: '固定风险类别', exact: true })).toBeVisible();
  }
  await capture(page, '08-零主题结果.png');
});

test('schema2编写校园围攻挑战显示真实话题与原文支持', async ({ page, request }) => {
  test.skip(deployment.schemaVersion !== 2, 'Schema2 presentation only');
  const latestFull = deployment.modelClass === 'FollowupPackage';
  const caseBytes = await readFile(latestFull ? resolve('tests/fixtures/latest-display-cases.json') : resolve(dirname(deployment.manifestPath), 'qa/cases.json'));
  const cases = JSON.parse(caseBytes.toString('utf8'));
  const selected = cases.cases.find((item: { id?: string; case_id?: string }) => (item.id ?? item.case_id) === 'campus:competition');
  expect(selected.kind).toBe('authored_challenge');
  const messages = (latestFull ? selected.session.messages.map((message: { id: string; text: string; order: number; parentId: string | null }) => ({ comment_id: message.id, raw_text: message.text, order: message.order, parent_id: message.parentId })) : selected.messages).map((message: { comment_id: string; raw_text: string; order: number; parent_id: string | null }) => ({ id: message.comment_id, text: message.raw_text, order: message.order, parentId: message.parent_id }));
  await expect.poll(async () => (await (await request.get('/api/health')).json()).worker.status, { timeout: 95_000 }).toBe('ready');
  const created = await request.post('/api/sessions', { data: { sessions: [{ title: '校园比赛围攻（编写挑战，非校园金标）', source: 'demo', messages }] } });
  expect(created.status()).toBe(201);
  const session = (await created.json()).sessions[0];
  const queued = await request.post('/api/analyses', { data: { sessionIds: [session.id] } });
  expect(queued.status()).toBe(202);
  const id = (await queued.json()).analyses[0].id;
  await expect.poll(async () => (await (await request.get(`/api/analyses/${id}`)).json()).analysis.status, { timeout: 125_000 }).toBe('succeeded');
  const detail = await (await request.get(`/api/analyses/${id}`)).json();
  expect(detail.analysis.modelHash).toBe(deployment.modelSha256);
  expect(detail.analysis.deploymentHash).toBe(deployment.deploymentHash);
  expect(detail.result.topics).toHaveLength(1);
  const topic = detail.result.topics[0];
  expect(topic.title).toBe(latestFull ? selected.expected_topic_titles[0] : '别再出来丢人 / 垃圾赶紧滚出');
  expect(topic.message_ids).toEqual(expect.arrayContaining(['m2', 'm6']));
  expect(topic.evidence.map((item: { comment_id: string }) => item.comment_id)).toEqual(expect.arrayContaining(['m2', 'm6']));
  await page.goto(`/#analysis/${id}`);
  await expect(page.locator('.topic-card h3')).toHaveText(topic.title);
  await page.locator('.topic-card .text-button').click();
  for (const mid of ['m2', 'm6']) await expect(page.locator(`#message-${mid}`)).toContainText(messages.find((message: { id: string }) => message.id === mid).text);
  await page.locator('.model-details summary').click();
  await capture(page, '10-校园围攻主题与原文支持.png');
  await writeFile(resolve(screenshotDir, '10-authored-competition-result.json'), JSON.stringify({
    caseId: selected.id ?? selected.case_id, kind: selected.kind, note: '编写功能挑战，不是真实校园金标或独立准确率样本',
    qaCaseFileSha256: createHash('sha256').update(caseBytes).digest('hex'),
    modelVersion: detail.analysis.modelVersion, modelHash: detail.analysis.modelHash,
    deploymentHash: detail.analysis.deploymentHash, schemaVersion: detail.analysis.schemaVersion,
    actualTopicTitles: detail.result.topics.map((item: { title: string }) => item.title),
    session: detail.session, analysis: detail.analysis, result: detail.result,
  }, null, 2), 'utf8');
});
