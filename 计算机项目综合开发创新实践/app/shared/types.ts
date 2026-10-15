export type Source = 'manual' | 'upload' | 'demo' | 'sccd';
export type RiskCategoryId = 'attack' | 'hate' | 'spam_scam';
export interface ModelIdentity { modelVersion: string; modelHash: string; deploymentHash: string; schemaVersion: 1 | 2; taskName: string }
export interface DeploymentManifest { manifestVersion: 1; modelVersion: string; schemaVersion: 1 | 2; taskName: string; modelPath: string; modelSha256: string; predictorModule: 'cst_mil.bench.model' | 'cst_mil.unified_semantic_v1_1.model' | 'cst_mil.demo_semantic_adapter' | 'cst_mil.deployment.adapter'; modelClass: 'BenchmarkModel' | 'UnifiedSemanticModel' | 'FollowupPackage'; predictFunction: 'predict_session'; codeFingerprints: Record<string, string>; assetFingerprints?: Record<string, string>; bundle?: { manifestPath: string; explanationPath: string; seed: 42; architecture: 'full' }; encoder: { path: string; fingerprint: string; manifestPath?: string } | null; thresholds: Record<string, number>; conversationThreshold: number; trainingSources: string[]; portable?: { release: string; catalogSha256: string; publishedManifestPath: string; archiveSha256: string } }
export interface MessageInput { id: string; text: string; speaker?: string | null; parentId?: string | null; time?: string | null; order?: number }
export interface SessionInput { title: string; postText?: string; includePost?: boolean; source: Source; messages: MessageInput[] }
export interface SessionRecord extends SessionInput { id: string; createdAt: string; messageCount: number }
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed';
export type ReviewStatus = 'pending' | 'risk' | 'clear' | 'uncertain';
export interface Review { status: ReviewStatus; note: string; updatedAt: string | null }
export interface AnalysisRecord {
  id: string; sessionId: string; title: string; source: Source; messageCount: number;
  status: JobStatus; createdAt: string; completedAt: string | null;
  prediction: number | null; riskScore: number | null; topicCount: number | null;
  error: string | null; modelHash: string | null; modelVersion: string;
  threshold: number | null; durationMs: number | null; review: Review;
  schemaVersion: 1 | 2; deploymentHash: string | null; scoreType: string | null;
}
export interface MessageLocation { post_id: string; comment_id: string; parent_id: string | null; order: number; position: number; comment_time: string | null; speaker_id: string | null; raw_text: string }
export interface Evidence extends MessageLocation { risk_score: number; rank?: number; topic_strength?: number; evidence_score?: number }
export interface MessageScore extends Evidence { topic_ids: number[]; topic_strengths: number[]; assigned: boolean; rank: number; riskCategoryScores?: Record<RiskCategoryId, number>; assignment_provenance?: 'semantic_cluster' | 'direct_parent_context' | 'unassigned'; context_parent_message_id?: string | null; qualityStatus?: string }
export interface TopicSegment { start_position: number; end_position: number; start_message_id: string; end_message_id: string; start_time: string | null; end_time: string | null; message_ids: string[] }
export interface KeywordSupport { message_id: string; start: number; end: number; excerpt: string }
export interface Topic { topic_id: number; title: string; keywords: string[]; topic_strength: number; risk_score: number; message_ids: string[]; support_messages: MessageLocation[]; segments: TopicSegment[]; evidence: Evidence[]; keyword_supports?: Record<string, KeywordSupport[]>; basic_message_ids?: string[]; context_association_message_ids?: string[]; qualityStatus?: string; contextHint?: string | null }
export interface PredictionBase {
  post_id: string; split: string; label: null; prediction: number; prediction_name: string;
  topics: Topic[]; message_scores: MessageScore[];
  evidence: Evidence[]; coverage: number;
  coverage_details: { assigned_messages: number; total_messages: number; fraction: number; unassigned_message_ids: string[]; zero_topic_messages: number };
}
export interface PredictionV1 extends PredictionBase { schemaVersion?: 1; risk_probability: number; threshold: number }
export interface RiskCategory { id: RiskCategoryId; name: string; score: number; threshold: number; detected: boolean; message_ids: string[]; evidence: Evidence[] }
export interface PredictionV2 extends PredictionBase { schemaVersion: 2; overallEvidenceScore: number; scoreType: 'max_message_category_score'; riskCategories: RiskCategory[]; conversationRisk: { score: number; threshold: number; prediction: number; domain: 'SCCD'; note: string }; modelVersion: string }
export type Prediction = PredictionV1 | PredictionV2;
export const isPredictionV2 = (prediction: Prediction): prediction is PredictionV2 => prediction.schemaVersion === 2;
export const predictionScore = (prediction: Prediction): number => isPredictionV2(prediction) ? prediction.overallEvidenceScore : prediction.risk_probability;
export interface AnalysisDetail { analysis: AnalysisRecord; session: SessionRecord; result: Prediction | null }
export interface Example { id: string; title: string; description: string; tags: string[]; session: SessionInput }
export interface WorkerHealth extends Partial<ModelIdentity> { status: 'starting' | 'ready' | 'error'; error?: string; threshold?: number; thresholds?: Record<string, number>; conversationThreshold?: number; startupDurationMs?: number; pythonStartupSeconds?: number; encoderWarmed?: boolean; firstRequestDurationMs?: number; firstRequestStatus?: 'succeeded' | 'failed' }
export interface Health { app: string; appProtocolVersion: number; worker: WorkerHealth; counts: { sessions: number; analyses: number } }
export interface BenchmarkRow { method: string; label: string; macro_f1: number; recall_cb: number; precision_cb: number; ap: number; roc_auc: number; hit_rate_at_3: number; mean_recall_at_3: number; comment_ap: number; train_seconds: number; inference_test_seconds: number; latency_mean_seconds: number; peak_rss_mib: number; model_mib: number }
export interface Benchmarks { dataset: string; split: string; note: string; sources: string[]; methods: BenchmarkRow[] }
export interface ImportRequest { format: 'paste' | 'csv' | 'json'; content: string; title?: string; postText?: string; includePost?: boolean }
export interface ImportPreview { sessions: SessionInput[]; warnings: string[] }
