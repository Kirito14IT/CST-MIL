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

test('real model supports analysis, review, exports, history and research information', async ({ page, request }) => {
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
  await expect(page.getByRole('heading', { name: 'Explore topics and risk clues' })).toBeVisible();
  await capture(page, '01-new-analysis.png');
  const title = `校园作业核查-${Date.now()}`;
  await page.getByLabel('Conversation title', { exact: false }).fill(title);
  await page.getByLabel('Comments', { exact: true }).fill('Alex：这周的小组作业我们怎么分工？\nBlair：我来整理数据，周三之前发给大家。\n同学C：那我负责把结果画成图。\nAlex：好，周四我们一起检查。');
  await page.getByRole('button', { name: 'Validate and preview', exact: true }).click();
  await expect(page.getByText('Validated', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Start analysis', exact: true }).click();
  await expect(page.getByRole('heading', { name: deployment.schemaVersion === 2 ? 'Discussion topics' : 'Topic discovery', exact: true })).toBeVisible({ timeout: 125_000 });
  if (deployment.schemaVersion === 2) {
    await expect(page.getByRole('heading', { name: 'Fixed risk categories', exact: true })).toBeVisible();
    await expect(page.getByText('SCCD conversation reference', { exact: true })).toBeVisible();
    await expect(page.getByText('Highest message clue score', { exact: true }).first()).toBeVisible();
  }
  await expect(page.getByRole('heading', { name: title, exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Top-3 candidate evidence', exact: true })).toBeVisible();
  await page.locator('.evidence-card').first().click();
  await expect(page.locator('.conversation-message.focused')).toBeVisible();
  await page.getByRole('radio', { name: 'No risk identified', exact: true }).check();
  const note = '已查看完整上下文，内容是正常的小组作业分工。';
  await page.getByLabel('Review notes', { exact: true }).fill(note);
  await page.getByRole('button', { name: 'Save review', exact: true }).click();
  await expect(page.getByText('Unsaved changes', { exact: true })).toHaveCount(0);
  await capture(page, '02-analysis-review.png');
  const jsonDownloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'JSON', exact: true }).click();
  const jsonDownload = await jsonDownloadPromise;
  expect(jsonDownload.suggestedFilename()).toMatch(/\.json$/);
  await jsonDownload.saveAs(resolve(screenshotDir, 'example-result.json'));
  const csvDownloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: 'Export CSV', exact: true }).click();
  const csvDownload = await csvDownloadPromise;
  expect(csvDownload.suggestedFilename()).toMatch(/\.csv$/);
  await csvDownload.saveAs(resolve(screenshotDir, 'example-messages.csv'));
  await page.getByRole('button', { name: 'History', exact: true }).click();
  await expect(page.getByRole('button', { name: title, exact: true })).toBeVisible();
  await capture(page, '03-history.png');
  await page.getByRole('button', { name: title, exact: true }).click();
  await expect(page.getByLabel('Review notes', { exact: true })).toHaveValue(note);
  await page.reload();
  await expect(page.getByLabel('Review notes', { exact: true })).toHaveValue(note);
  await page.getByRole('button', { name: 'Research code', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Experimental results are not distributed', exact: true })).toBeVisible();
  await expect(page.locator('.benchmark-table')).toHaveCount(0);
  await capture(page, '04-research-code.png');
  expect(errors).toEqual([]);
});

test('schema2 preserves multilingual source text and model identity', async ({ page, request }) => {
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
  await expect(page.getByRole('heading', { name: 'Fixed risk categories', exact: true })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Discussion topics', exact: true })).toBeVisible();
  await expect(page.locator('.category-card')).toHaveCount(3);
  for (const message of messages) await expect(page.locator('.conversation-message').filter({ hasText: message.text })).toBeVisible();
  await page.locator('.model-details summary').click();
  await expect(page.locator('.model-details')).toContainText(deployment.modelVersion);
  await expect(page.locator('.model-details')).toContainText(deployment.modelSha256);
  await expect(page.locator('.model-details')).toContainText(deployment.deploymentHash);
  await capture(page, '09-multilingual-source-identity.png');
});

test('upload preview, validation errors, examples and 1024-pixel layout', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 768 });
  await page.goto('/');
  await page.getByRole('tab', { name: 'Upload file', exact: true }).click();
  await page.locator('input[type=file]').setInputFiles({ name: 'duplicate-ids.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ title: '无效示例', messages: [{ id: 'same', text: '第一句' }, { id: 'same', text: '第二句' }] })) });
  await page.getByRole('button', { name: 'Validate and preview', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('duplicate');
  await page.locator('input[type=file]').setInputFiles(resolve('examples/campus-comments.csv'));
  await page.getByRole('button', { name: 'Validate and preview', exact: true }).click();
  await expect(page.getByText('Validated', { exact: true })).toBeVisible();
  await expect(page.getByRole('combobox', { name: /Select conversation/ })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await capture(page, '06-import-preview-1024.png');
  await page.getByRole('tab', { name: 'Examples', exact: true }).click();
  await expect(page.locator('.example-card').first()).toBeVisible();
  await page.locator('.example-card').first().click();
  await page.getByRole('button', { name: 'Validate and preview', exact: true }).click();
  await expect(page.getByText('Validated', { exact: true })).toBeVisible();
  await capture(page, '07-examples.png');
});

test('unknown-word input supports inspection and human review', async ({ page, request }) => {
  await expect.poll(async () => (await (await request.get('/api/health')).json()).worker.status, { timeout: 95_000 }).toBe('ready');
  await page.goto('/');
  await page.getByLabel('Conversation title', { exact: false }).fill('零主题边界验证');
  await page.getByLabel('Comments', { exact: true }).fill('zzzxqvjk');
  await page.getByRole('button', { name: 'Validate and preview', exact: true }).click();
  await page.getByRole('button', { name: 'Start analysis', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Human review', exact: true })).toBeVisible({ timeout: 125_000 });
  await expect(page.getByLabel('Review notes', { exact: true })).toBeVisible();
  const id = page.url().split('/').at(-1)!;
  const detail = await (await request.get(`/api/analyses/${id}`)).json();
  expect(detail.analysis.status).toBe('succeeded');
  if (deployment.schemaVersion === 1) expect(detail.result.topics).toEqual([]);
  else {
    expect(detail.result.schemaVersion).toBe(2); expect(detail.result.riskCategories).toHaveLength(3);
    if (detail.result.topics.length === 0) await expect(page.getByRole('heading', { name: 'No clear topics found', exact: true })).toBeVisible();
    await expect(page.locator('.conversation-message').filter({ hasText: 'zzzxqvjk' })).toBeVisible();
    await expect(page.locator('.message-category-scores').first()).not.toBeEmpty();
    await expect(page.getByRole('heading', { name: 'Fixed risk categories', exact: true })).toBeVisible();
  }
  await capture(page, '08-no-topic.png');
});

test('schema2 authored Chinese challenge retains grounded topic support', async ({ page, request }) => {
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
  await capture(page, '10-authored-source-support.png');
  await writeFile(resolve(screenshotDir, '10-authored-competition-result.json'), JSON.stringify({
    caseId: selected.id ?? selected.case_id, kind: selected.kind, note: '编写功能挑战，不是真实校园金标或独立准确率样本',
    qaCaseFileSha256: createHash('sha256').update(caseBytes).digest('hex'),
    modelVersion: detail.analysis.modelVersion, modelHash: detail.analysis.modelHash,
    deploymentHash: detail.analysis.deploymentHash, schemaVersion: detail.analysis.schemaVersion,
    actualTopicTitles: detail.result.topics.map((item: { title: string }) => item.title),
    session: detail.session, analysis: detail.analysis, result: detail.result,
  }, null, 2), 'utf8');
});
