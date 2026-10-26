import type { Prediction, ModelIdentity, WorkerHealth, RiskCategoryId, SessionRecord } from '../shared/types.ts';
import type { LoadedDeployment } from '../config/deployment.mjs';

const categories: RiskCategoryId[] = ['attack', 'hate', 'spam_scam'];
const probability = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
const object = (value: unknown): Record<string, unknown> => { if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Model output schema validation failed'); return value as Record<string, unknown>; };
const sameNumber = (a: number, b: number) => Math.abs(a - b) <= 1e-12;
const nonnegative = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const integer = (value: unknown): value is number => nonnegative(value) && Number.isInteger(value);

export function workerIdentity(health: WorkerHealth): ModelIdentity {
  if (!health.modelVersion || !health.modelHash || !health.deploymentHash || !health.taskName || ![1, 2].includes(health.schemaVersion ?? 0)) throw new Error('The inference worker returned incomplete model identity. Result not saved as successful');
  return { modelVersion: health.modelVersion, modelHash: health.modelHash, deploymentHash: health.deploymentHash, schemaVersion: health.schemaVersion as 1 | 2, taskName: health.taskName };
}

export function validatePrediction(value: unknown, deployment: LoadedDeployment, postId: string, input?: SessionRecord): Prediction {
  const result = object(value);
  if (result.label !== null || result.post_id !== postId || ![0, 1].includes(Number(result.prediction)) || typeof result.prediction !== 'number' || !Array.isArray(result.topics) || !Array.isArray(result.message_scores) || !result.message_scores.length || !Array.isArray(result.evidence)) throw new Error('Model output base fields failed validation');
  const messages = result.message_scores.map(object);
  const originals = new Map(messages.map(message => [message.comment_id, message.raw_text]));
  const byId = new Map(messages.map(message => [message.comment_id, message]));
  if (originals.size !== messages.length || messages.some(message => typeof message.comment_id !== 'string' || typeof message.raw_text !== 'string' || !probability(message.risk_score) || !Array.isArray(message.topic_ids) || message.topic_ids.some(id => !integer(id)) || !Array.isArray(message.topic_strengths) || message.topic_strengths.some(strength => !nonnegative(strength)) || typeof message.assigned !== 'boolean' || message.assigned !== (message.topic_ids.length > 0) || !integer(message.rank) || message.rank < 1)) throw new Error('Model message output failed validation');
  if (input) {
    const expected = new Map(input.messages.map(message => [message.id, message.text]));
    const expectedParents = new Map(input.messages.map(message => [message.id, message.parentId ?? null]));
    if (input.includePost) { let rootId = '__post__', suffix = 0; while (expected.has(rootId)) rootId = `__post___${++suffix}`; expected.set(rootId, input.postText ?? ''); expectedParents.set(rootId, null); }
    if (expected.size !== originals.size || [...expected].some(([id, text]) => originals.get(id) !== text || byId.get(id)?.parent_id !== expectedParents.get(id))) throw new Error('Model output did not preserve the input messages, source text and parent links');
  }
  const checkLocations = (locations: unknown) => {
    if (!Array.isArray(locations)) throw new Error('Invalid model source-location format');
    for (const item of locations.map(object)) if (!originals.has(item.comment_id) || originals.get(item.comment_id) !== item.raw_text) throw new Error('Model evidence lacks its complete source text');
  };
  const checkEvidence = (evidence: unknown, categoryId?: RiskCategoryId) => {
    if (!Array.isArray(evidence)) throw new Error('Invalid model evidence format');
    checkLocations(evidence);
    for (const item of evidence.map(object)) {
      const source = byId.get(item.comment_id)!;
      const expected = categoryId ? object(source.riskCategoryScores)[categoryId] : source.risk_score;
      if (!probability(item.risk_score) || !sameNumber(item.risk_score, Number(expected))) throw new Error('Evidence score does not match the source message');
    }
  };
  checkEvidence(result.evidence);
  const topics = result.topics.map(object);
  const topicIds = new Set(topics.map(topic => topic.topic_id));
  if (topicIds.size !== topics.length || topics.some(topic => !integer(topic.topic_id) || typeof topic.title !== 'string' || !Array.isArray(topic.keywords) || topic.keywords.some(word => typeof word !== 'string') || !nonnegative(topic.topic_strength) || !probability(topic.risk_score) || !Array.isArray(topic.message_ids) || topic.message_ids.some(id => !originals.has(id)) || !Array.isArray(topic.segments))) throw new Error('Invalid topic output schema');
  for (const topic of topics) {
    checkLocations(topic.support_messages); checkEvidence(topic.evidence);
    for (const segment of (topic.segments as unknown[]).map(object)) if (!integer(segment.start_position) || !integer(segment.end_position) || segment.end_position < segment.start_position || !originals.has(segment.start_message_id) || !originals.has(segment.end_message_id) || !Array.isArray(segment.message_ids) || segment.message_ids.some(id => !originals.has(id))) throw new Error('Invalid topic message-segment schema');
  }
  if (messages.some(message => (message.topic_ids as number[]).some(id => !topicIds.has(id)))) throw new Error('Message references a missing topic');
  const coverage = object(result.coverage_details);
  const assigned = messages.filter(message => message.assigned).length;
  const unassigned = messages.filter(message => !message.assigned).map(message => message.comment_id).sort().join('\0');
  if (!probability(result.coverage) || !integer(coverage.assigned_messages) || coverage.assigned_messages !== assigned || coverage.total_messages !== messages.length || !probability(coverage.fraction) || !sameNumber(coverage.fraction, assigned / messages.length) || !sameNumber(result.coverage, coverage.fraction) || !integer(coverage.zero_topic_messages) || coverage.zero_topic_messages > messages.length || !Array.isArray(coverage.unassigned_message_ids) || [...coverage.unassigned_message_ids].sort().join('\0') !== unassigned) throw new Error('Topic coverage fields are invalid or inconsistent with messages');
  if (deployment.schemaVersion === 1) {
    if ((result.schemaVersion !== undefined && result.schemaVersion !== 1) || !probability(result.risk_probability) || !probability(result.threshold) || !sameNumber(result.threshold, deployment.thresholds.attack)) throw new Error('Legacy model score or threshold failed validation');
  } else {
    if (result.schemaVersion !== 2 || result.modelVersion !== deployment.modelVersion || result.scoreType !== 'max_message_category_score' || 'risk_probability' in result || 'threshold' in result || !probability(result.overallEvidenceScore) || !Array.isArray(result.riskCategories)) throw new Error('Semantic model output schema failed validation');
    const memberships = new Map<string, number>();
    for (const topic of topics) {
      const keywords = topic.keywords as string[];
      const members = topic.message_ids as string[];
      const memberIds = new Set(members);
      if (!keywords.length || keywords.some(word => !word.trim()) || new Set(keywords).size !== keywords.length || topic.title !== keywords.slice(0, 2).join(' / ') || !members.length || memberIds.size !== members.length) throw new Error('Topic names must use non-empty distinct source keywords');
      for (const location of [...topic.support_messages as unknown[], ...topic.evidence as unknown[]].map(object)) if (!memberIds.has(String(location.comment_id))) throw new Error('Supporting message is not a member of this topic');
      const supports = object(topic.keyword_supports);
      let basicIds: Set<string> | undefined;
      if (deployment.modelClass === 'FollowupPackage' || topic.basic_message_ids !== undefined || topic.context_association_message_ids !== undefined) {
        if (!Array.isArray(topic.basic_message_ids) || topic.basic_message_ids.length < 2 || !Array.isArray(topic.context_association_message_ids)) throw new Error('Invalid core or parent-linked topic members');
        const basic = topic.basic_message_ids as string[], linked = topic.context_association_message_ids as string[];
        basicIds = new Set(basic);
        const combined = [...basic, ...linked];
        if (new Set(combined).size !== combined.length || combined.length !== members.length || combined.some(id => !memberIds.has(id))) throw new Error('Core and linked topic members must be complete and disjoint');
        for (const id of combined) {
          const message = byId.get(id);
          if (!message || memberships.has(id) || (message.topic_ids as number[]).length !== 1 || (message.topic_ids as number[])[0] !== topic.topic_id) throw new Error('Message topic membership is inconsistent or duplicated');
          memberships.set(id, topic.topic_id as number);
          if (basicIds.has(id) && message.assignment_provenance !== 'semantic_cluster') throw new Error('Core member lacks semantic-cluster provenance');
          if (!basicIds.has(id) && (typeof message.parent_id !== 'string' || !basicIds.has(message.parent_id) || message.context_parent_message_id !== message.parent_id || message.assignment_provenance !== 'direct_parent_context' || Array.from(String(message.raw_text).replace(/\s/gu, '')).length > 12 || (message.topic_strengths as number[])[topic.topic_id as number] !== 0)) throw new Error('Linked members must use a real direct parent without cascading or fabricated semantic strength');
        }
      }
      if (Object.keys(supports).sort().join('\0') !== [...keywords].sort().join('\0')) throw new Error('Keyword source support is incomplete');
      for (const word of keywords) {
        const entries = supports[word];
        if (!Array.isArray(entries) || !entries.length) throw new Error('Keyword lacks source support');
        for (const support of entries.map(object)) {
          if (typeof support.message_id !== 'string' || !memberIds.has(support.message_id) || !originals.has(support.message_id) || (basicIds && !basicIds.has(support.message_id))) throw new Error('Keyword support is not a core member of this topic');
          const codepoints = Array.from(String(originals.get(support.message_id)));
          if (!integer(support.start) || !integer(support.end) || support.start >= support.end || support.end > codepoints.length || typeof support.excerpt !== 'string' || codepoints.slice(support.start, support.end).join('') !== support.excerpt) throw new Error('Keyword character span or excerpt does not match source text');
        }
      }
    }
    if (deployment.modelClass === 'FollowupPackage' && messages.some(message => message.assigned !== memberships.has(String(message.comment_id)))) throw new Error('Topic coverage does not match the member list');
    const riskCategories = result.riskCategories.map(object);
    if (riskCategories.map(category => category.id).sort().join(',') !== [...categories].sort().join(',')) throw new Error('Risk categories are incomplete or duplicated');
    for (const message of messages) {
      const scores = object(message.riskCategoryScores);
      if (Object.keys(scores).sort().join(',') !== [...categories].sort().join(',') || categories.some(id => !probability(scores[id]))) throw new Error('Invalid message category scores');
      if (!sameNumber(Number(message.risk_score), Math.max(...categories.map(id => Number(scores[id]))))) throw new Error('Message clue score is not the maximum category score');
    }
    for (const category of riskCategories) {
      const id = category.id as RiskCategoryId;
      const maximum = Math.max(...messages.map(message => Number(object(message.riskCategoryScores)[id])));
      if (typeof category.name !== 'string' || !category.name || !probability(category.score) || !probability(category.threshold) || !sameNumber(category.threshold, deployment.thresholds[id]) || !sameNumber(category.score, maximum) || category.detected !== (category.score >= category.threshold) || !Array.isArray(category.message_ids) || category.message_ids.some(messageId => !originals.has(messageId))) throw new Error('Risk category score, threshold or evidence ID is inconsistent');
      checkEvidence(category.evidence, id);
    }
    const maximum = Math.max(...riskCategories.map(category => Number(category.score)));
    if (!sameNumber(result.overallEvidenceScore, maximum) || result.prediction !== Number(riskCategories.some(category => category.detected))) throw new Error('Highest message clue score or overall decision is inconsistent');
    const conversation = object(result.conversationRisk);
    if (!probability(conversation.score) || !probability(conversation.threshold) || !sameNumber(conversation.threshold, deployment.conversationThreshold) || conversation.prediction !== Number(conversation.score >= conversation.threshold) || conversation.domain !== 'SCCD' || typeof conversation.note !== 'string') throw new Error('SCCD conversation reference fields are inconsistent');
  }
  return result as unknown as Prediction;
}
