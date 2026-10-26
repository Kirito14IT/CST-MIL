import { ArrowRight, ChevronDown, CornerDownRight, Quote } from 'lucide-react';
import type { KeywordSupport, Topic } from '../shared/types';

export interface SourceHighlight { messageId: string; start: number; end: number; excerpt: string }
type TraceableTopic = Topic & { basic_message_ids?: string[]; context_association_message_ids?: string[] };

/** Python source offsets count Unicode code points, while JS string slicing counts UTF-16 units. */
export function sourceParts(text: string, highlight?: SourceHighlight | null): { before: string; match: string; after: string } | null {
  if (!highlight) return null;
  const points = Array.from(text);
  const { start, end, excerpt } = highlight;
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > points.length) return null;
  const match = points.slice(start, end).join('');
  if (match !== excerpt) return null;
  return { before: points.slice(0, start).join(''), match, after: points.slice(end).join('') };
}

export function TopicCard({ topic, index, onFilter, onSupport, onMessage }: {
  topic: TraceableTopic; index: number; onFilter: () => void;
  onSupport: (support: KeywordSupport) => void; onMessage: (messageId: string) => void;
}) {
  const associations = topic.context_association_message_ids ?? [];
  const basic = topic.basic_message_ids ?? topic.message_ids.filter(id => !associations.includes(id));
  const hasMemberKinds = Array.isArray(topic.basic_message_ids) || Array.isArray(topic.context_association_message_ids);
  const supports = topic.keywords.flatMap(keyword => (topic.keyword_supports?.[keyword] ?? []).map(support => ({ keyword, support })));
  return <article className="topic-card" data-topic-id={topic.topic_id}>
    <div className="topic-heading"><span className={`topic-number color-${index % 5}`}>Topic {index + 1}</span><span>{hasMemberKinds ? `${basic.length} core messages` : `${topic.message_ids.length} associated messages`}{associations.length > 0 && ` · ${associations.length} linked replies`}</span></div>
    <h3>{topic.title || topic.keywords.slice(0, 3).join(' · ') || `Topic ${index + 1}`}</h3>
    <div className="keyword-list">{topic.keywords.map((keyword, i) => {
      const support = topic.keyword_supports?.[keyword]?.[0];
      return support ? <button key={`${keyword}-${i}`} className="keyword-link" title={`Locate source: ${support.excerpt}`} onClick={() => onSupport(support)}>{keyword}<Quote size={12} /></button> : <span key={`${keyword}-${i}`}>{keyword}</span>;
    })}</div>
    {supports.length > 0 && <details className="topic-source-details"><summary><Quote size={14} />Keyword source evidence<ChevronDown size={14} /></summary><div className="topic-source-list">{supports.map(({ keyword, support }, i) => <button key={`${keyword}-${support.message_id}-${support.start}-${i}`} className="source-support" onClick={() => onSupport(support)}><span><b>{keyword}</b><small title="Zero-based Unicode codepoint span; end is exclusive">Message {support.message_id} · Span [{support.start}, {support.end})</small></span><q>{support.excerpt}</q><ArrowRight size={14} /></button>)}</div></details>}
    {associations.length > 0 && <div className="context-associations"><span><CornerDownRight size={14} />Linked through a direct parent</span><div>{associations.map(messageId => <button key={messageId} onClick={() => onMessage(messageId)}>Message {messageId}<ArrowRight size={12} /></button>)}</div><p>Linked replies do not affect keyword statistics. Their risk scores are computed independently.</p></div>}
    <div className="topic-bottom"><span title="Model topic grouping strength, not accuracy">Group strength <b>{Number.isFinite(topic.topic_strength) ? topic.topic_strength.toFixed(3) : '—'}</b></span><button className="text-button" onClick={onFilter}>View {topic.message_ids.length} associated messages<ArrowRight size={14} /></button></div>
  </article>;
}
