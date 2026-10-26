import { createReadStream, existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import path from 'node:path';
import type { Example, MessageInput } from '../shared/types.ts';
import { RESEARCH_ROOT } from './paths.ts';
import recommendedSessions from '../examples/campus-recommended-demo.json';
import { validateSessions } from './validation.ts';

const definitions = [
  {
    "title": "Campus competition discussion",
    "description": "A fictional competition thread discusses passing practice and a Friday review meeting alongside personal attacks.",
    "tags": [
      "Competition",
      "Context",
      "Human review"
    ],
    "lines": [
      "Our team needs more passing practice before the next match.",
      "We need more passing practice before our next match.",
      "More passing practice will help our team prepare for the next match.",
      "That player is a useless idiot and should leave the team.",
      "He is a useless idiot who should leave our team.",
      "That useless idiot should leave the team after this match.",
      "The match review meeting is on Friday in the library.",
      "We will review the match on Friday in the library.",
      "Our Friday meeting in the library will review the match.",
      "Please discuss the actual decisions instead of attacking a teammate."
    ]
  },
  {
    "title": "Planning a group assignment",
    "description": "An ordinary collaboration example for learning the input, analysis and review workflow.",
    "tags": [
      "Coursework",
      "Collaboration"
    ],
    "lines": [
      "The report is due on Friday. Let us divide the work today.",
      "I can organize the survey data and share it tomorrow evening.",
      "I will prepare the charts. Who can review the writing?",
      "I can proofread it and check the references.",
      "Thanks everyone. Post any questions here."
    ]
  },
  {
    "title": "Dormitory quiet hours",
    "description": "Roommates discuss noise, explain their constraints and agree on practical changes.",
    "tags": [
      "Campus life",
      "Disagreement"
    ],
    "lines": [
      "The keyboard noise after midnight has been keeping me awake.",
      "Sorry about that. I will finish my project in the study room tonight.",
      "Could we use headphones and keep things quiet after eleven?",
      "Agreed. I will also turn down my morning alarm.",
      "Thanks for talking it through."
    ]
  },
  {
    "title": "Cafeteria feedback and photography club",
    "description": "Two discussion subjects appear in one thread. Inspect how messages connect to topics.",
    "tags": [
      "Multiple topics",
      "Campus life"
    ],
    "lines": [
      "The cafeteria queue is too long at lunchtime. Another service counter would help.",
      "I agree. We can record the busiest times and send feedback.",
      "The photography club has a campus walk this weekend.",
      "What time is the photography walk? Do we need a camera?",
      "I have written the cafeteria feedback about queues and seating.",
      "The photography walk starts at two. A phone camera is fine."
    ]
  },
  {
    "title": "Quoting an insult to discourage it",
    "description": "The speaker quotes offensive wording while asking others to stop. Read the surrounding context.",
    "tags": [
      "Context",
      "Quotation"
    ],
    "lines": [
      "Someone wrote \"you are a useless idiot\" in the comments. That is not acceptable.",
      "Right. A mistake on one assignment is no reason to insult a classmate.",
      "I have asked everyone to stop the personal attacks.",
      "Please discuss the actual problem instead of repeating that insult."
    ]
  },
  {
    "title": "Confirming a reading group",
    "description": "A two-message conversation illustrates the interface with a small amount of text.",
    "tags": [
      "Short conversation",
      "Scheduling"
    ],
    "lines": [
      "Is the reading group still meeting at the library tomorrow?",
      "Yes. See you at three in the second-floor discussion room."
    ]
  }
];

export const campusExamples: Example[] = definitions.map((item, index) => ({
  id: `campus-english-${index + 1}`, title: item.title,
  description: `Fictional demonstration. ${item.description} No model outcome is prescribed.`, tags: ['Fictional demo', ...item.tags],
  session: { title: item.title, source: 'demo', includePost: false, postText: '', messages: item.lines.map((line, i) => ({ id: `m${i + 1}`, text: line, order: i, speaker: `Student ${String.fromCharCode(65 + i % 4)}`, parentId: index === 0 ? null : i > 1 ? 'm1' : null })) },
}));

const recommendedMetadata = [
  { id: 'campus-forum-competition', title: 'Competition comments (Chinese)', description: 'A fictional Chinese thread with personal attacks and requests to stop.', tags: ['Chinese', 'Competition'] },
  { id: 'campus-forum-rumor', title: 'Rumors and personal criticism (Chinese)', description: 'A fictional Chinese thread with unverified allegations and contextual ambiguity.', tags: ['Chinese', 'Context'] },
  { id: 'campus-forum-normal', title: 'Constructive match review (Chinese)', description: 'A fictional Chinese thread about passing, defense and reviewing a match.', tags: ['Chinese', 'Constructive discussion'] },
  { id: 'campus-threat-boundary', title: 'Threatening language (Chinese)', description: 'A short fictional Chinese conversation requiring careful human review.', tags: ['Chinese', 'Short conversation'] },
];
export const recommendedCampusExamples: Example[] = validateSessions(recommendedSessions).map((session, index) => ({
  ...recommendedMetadata[index], session: { ...session, title: recommendedMetadata[index].title },
  description: `Fictional demonstration. ${recommendedMetadata[index].description} This example is not an evaluation sample.`,
  tags: ['Fictional demo', ...recommendedMetadata[index].tags],
}));

let cached: Promise<Example[]> | null = null;
export function getExamples(): Promise<Example[]> {
  cached ??= loadSccdExamples().then(sccd => [...campusExamples, ...recommendedCampusExamples, ...sccd]).catch(error => {
    console.error('[SCCD examples]', error instanceof Error ? error.message : 'Local source data could not be read');
    return [...campusExamples, ...recommendedCampusExamples];
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
      if (!speakers.has(speakerKey)) speakers.set(speakerKey, `Speaker ${speakers.size + 1}`);
      return { id: ids.get(row.comment_id)!, text: row.raw_text, order: row.order, speaker: speakers.get(speakerKey), parentId: row.parent_id ? ids.get(row.parent_id) ?? null : null, time: row.comment_time ?? null };
    });
    return { id: `sccd-${index + 1}`, title: `SCCD Weibo conversation ${index + 1}`, description: 'Loaded from the local SCCD source dataset. Original text is preserved; speaker and message IDs are anonymized for display. This is Weibo data, not campus data. Original labels are excluded from inference.', tags: ['SCCD', 'Local source data', 'Non-campus data'], session: { title: `SCCD Weibo conversation ${index + 1}`, source: 'sccd', postText: session.post_raw_text, includePost: false, messages } };
  });
}
