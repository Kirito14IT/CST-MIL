import { createReadStream, existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { Example, MessageInput } from '../shared/types.ts';
import { RESEARCH_ROOT } from './paths.ts';
import recommendedSessions from '../examples/校园推荐演示-20261004.json';
import { validateSessions } from './validation.ts';

const definitions = [
  { title: '课程小组的分工讨论', description: '同学围绕作业分工交换意见，适合熟悉操作流程。', tags: ['课程学习', '合作讨论'], lines: ['我们周五交报告，今天先把任务分好吧。', '我整理问卷数据，明晚发给大家。', '我负责图表和页面，文字部分谁来检查？', '我来校对，引用的来源也一起核对。', '辛苦大家，有问题就在群里说。'] },
  { title: '宿舍生活中的意见分歧', description: '对晚间作息有不同看法，观察模型如何处理抱怨与协商。', tags: ['宿舍生活', '意见分歧'], lines: ['最近晚上十二点还有键盘声，我有点睡不好。', '抱歉，项目没做完，今晚我去公共自习区。', '我们能不能约定十一点以后戴耳机、放轻动作？', '可以，我也注意闹钟音量。', '谢谢，大家商量一下就好。'] },
  { title: '比赛失利后的不当评论', description: '虚构的攻击性发言与制止发言混在一起，便于查看候选证据。', tags: ['社团活动', '人工核查'], lines: ['今天比赛输了，大家复盘一下配合问题吧。', '那个队员真是废物，别再让他上场了。', '说他废物就过分了，每个人都有失误。', '他就是拖后腿，我们以后都别理他。', '可以讨论技术，别针对同学本人。', '先把训练录像整理出来，下次一起改进。'] },
  { title: '食堂建议与社团招募', description: '同一评论区出现生活建议和活动信息，观察主题与消息对应关系。', tags: ['多话题', '校园生活'], lines: ['食堂二楼中午排队太久了，希望多开一个窗口。', '我也遇到了，可以把高峰时段记录给食堂。', '周末摄影社有活动，想拍校园秋景的同学可以参加。', '摄影活动几点集合，需要自己带相机吗？', '食堂建议我已经写好了，主要是排队和座位问题。', '摄影活动下午两点，用手机也可以。'] },
  { title: '引用攻击性用语并劝阻', description: '发言中引用了不当用语，但真实意图是劝阻；提醒结合上下文核查。', tags: ['上下文', '易混淆案例'], lines: ['有人在评论里说“你就是个废物”，这样讲话不合适。', '对，不能因为一次作业没做好就骂同学。', '我已经提醒大家停止人身攻击了。', '讨论具体问题就好，别再传播那句话。'] },
  { title: '简短的活动确认', description: '只有两条消息的短会话，检查少量消息时的主题和证据展示。', tags: ['短会话', '活动安排'], lines: ['明天下午的读书分享还在图书馆吗？', '是的，三点在二楼讨论室见。'] },
];

export const campusExamples: Example[] = definitions.map((item, index) => ({
  id: `campus-${index + 1}`, title: item.title,
  description: `校园虚构示例。${item.description} 示例不提供预设模型判断。`, tags: ['虚构演示', ...item.tags],
  session: { title: item.title, source: 'demo', includePost: false, postText: '', messages: item.lines.map((line, i) => ({ id: `m${i + 1}`, text: line, order: i, speaker: `同学${String.fromCharCode(65 + i % 4)}`, parentId: i > 1 ? 'm1' : null })) },
}));

const recommendedMetadata = [
  { id: 'campus-forum-competition', description: '推荐从这一例开始：校园比赛评论区含针对队员的贬损与排斥，也有制止评论。2026-10-04 用真实模型复核后选入，适合查看风险候选及完整上下文。', tags: ['推荐演示', '独立评论', '攻击与劝阻'] },
  { id: 'campus-forum-rumor', description: '评论区含未核实的指控和针对个人的围攻。2026-10-04 复核后选入；候选可能包含提出质疑或制止攻击的人，需要逐条核查。', tags: ['推荐演示', '传言与围攻', '人工核查'] },
  { id: 'campus-forum-normal', description: '与比赛围攻场景配对的正常建议对照：讨论传球、防守和复盘，帮助理解正常交流也会有主题和候选消息。', tags: ['正常对照', '独立评论', '具体批评'] },
  { id: 'campus-threat-boundary', description: '明确含有对发言人的威胁，但本次复核没有达到模型阈值。用于观察漏检与人工核查；低分不能排除文本中的问题。', tags: ['边界观察', '本次漏检', '短会话'] },
];
export const recommendedCampusExamples: Example[] = validateSessions(recommendedSessions).map((session, index) => ({
  ...recommendedMetadata[index], title: session.title, session,
  description: `校园虚构示例。${recommendedMetadata[index].description} 按当前模型表现筛选，仅用于功能演示，不构成校园效果评测。`,
  tags: ['虚构演示', ...recommendedMetadata[index].tags],
}));

let cached: Promise<Example[]> | null = null;
export function getExamples(): Promise<Example[]> {
  cached ??= loadSccdExamples().then(sccd => [...recommendedCampusExamples, ...campusExamples, ...sccd]).catch(error => {
    console.error('[SCCD examples]', error instanceof Error ? error.message : '本地数据无法读取');
    return [...recommendedCampusExamples, ...campusExamples];
  });
  return cached;
}

async function loadSccdExamples(): Promise<Example[]> {
  const folder = path.join(RESEARCH_ROOT, 'benchmark', 'data', 'full677');
  const sessionsPath = path.join(folder, 'sessions.jsonl');
  const messagesPath = path.join(folder, 'messages.jsonl');
  if (!existsSync(sessionsPath) || !existsSync(messagesPath)) return [];
  type DatasetSession = { post_id: string; split: string; message_count: number; post_raw_text: string };
  type DatasetMessage = { post_id: string; comment_id: string; raw_text: string; order: number; parent_id?: string | null; speaker_id?: string | null; comment_time?: string | null };
  const sessions = (await readFile(sessionsPath, 'utf8')).split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line) as DatasetSession)
    .filter(s => s.split === 'test' && s.message_count >= 3 && s.message_count <= 30).sort((a, b) => a.post_id.localeCompare(b.post_id)).slice(0, 2);
  const wanted = new Set(sessions.map(s => s.post_id));
  const grouped = new Map<string, DatasetMessage[]>();
  const stream = createReadStream(messagesPath, { encoding: 'utf8' });
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    const message = JSON.parse(line) as DatasetMessage;
    if (wanted.has(message.post_id)) {
      const group = grouped.get(message.post_id) ?? [];
      group.push(message);
      grouped.set(message.post_id, group);
    }
  }
  return sessions.map((session, index) => {
    const rows = (grouped.get(session.post_id) ?? []).sort((a, b) => a.order - b.order);
    const ids = new Map(rows.map((row, i) => [row.comment_id, `m${i + 1}`]));
    const speakers = new Map<string, string>();
    const messages: MessageInput[] = rows.map((row, i) => {
      const speakerKey = row.speaker_id || `missing-${i}`;
      if (!speakers.has(speakerKey)) speakers.set(speakerKey, `发言人 ${speakers.size + 1}`);
      return { id: ids.get(row.comment_id)!, text: row.raw_text, order: row.order, speaker: speakers.get(speakerKey), parentId: row.parent_id ? ids.get(row.parent_id) ?? null : null, time: row.comment_time ?? null };
    });
    return { id: `sccd-${index + 1}`, title: `SCCD 微博会话 ${index + 1}`, description: '动态读取本机 SCCD 固定测试集，保留原评论文本，发言人和消息编号作展示匿名化。此为微博数据，不是校园采集数据；原始标签不提供给模型。', tags: ['SCCD', '本地研究数据', '非校园数据'], session: { title: `SCCD 微博会话 ${index + 1}`, source: 'sccd', postText: session.post_raw_text, includePost: false, messages } };
  });
}
