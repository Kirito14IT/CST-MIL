import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import type { AnalysisDetail, AnalysisRecord, Prediction, ModelIdentity, Review, ReviewStatus, SessionInput, SessionRecord } from '../shared/types.ts';
import { isPredictionV2, predictionScore } from '../shared/types.ts';
import { HttpError } from './validation.ts';

type Row = Record<string, string | number | null>;
export interface Filters { query?: string; prediction?: number; from?: string; to?: string }
export class Store {
  private db: DatabaseSync;
  private closed = false;
  constructor(dataDir: string) {
    mkdirSync(dataDir, { recursive: true });
    this.db = new DatabaseSync(path.join(dataDir, 'campus.sqlite'));
    try {
      // Connection-owned exclusive locks survive commits and are released by SQLite
      // on close or process death. Acquire before touching recovery/status records.
      this.db.exec('PRAGMA busy_timeout=250; PRAGMA locking_mode=EXCLUSIVE; BEGIN EXCLUSIVE; COMMIT;');
      const existing = this.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='analyses'").get();
      const columns = existing ? (this.db.prepare('PRAGMA table_info(analyses)').all() as Row[]).map(row => String(row.name)) : [];
      if (existing && ['schema_version', 'deployment_hash', 'score_type'].some(column => !columns.includes(column))) {
        // EXCLUSIVE locking mode retains this connection's lock after COMMIT.
        // VACUUM INTO is a SQLite-consistent snapshot, including committed WAL data.
        const backupDir = path.join(dataDir, 'backups');
        mkdirSync(backupDir, { recursive: true });
        const backup = path.join(backupDir, `campus-before-schema2-${Date.now()}-${randomUUID()}.sqlite`);
        this.db.exec(`VACUUM INTO '${backup.replace(/'/g, "''")}'`);
      }
      this.db.exec('PRAGMA journal_mode=WAL; BEGIN EXCLUSIVE;');
      this.db.exec(`
      CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, body TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS analyses(
        id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), title TEXT NOT NULL,
        source TEXT NOT NULL, message_count INTEGER NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
        completed_at TEXT, prediction INTEGER, risk_score REAL, topic_count INTEGER, error TEXT,
        model_hash TEXT, model_version TEXT NOT NULL, threshold REAL, duration_ms REAL, result TEXT,
        review_status TEXT NOT NULL DEFAULT 'pending', review_note TEXT NOT NULL DEFAULT '', review_updated_at TEXT
      ); CREATE INDEX IF NOT EXISTS analyses_created ON analyses(created_at DESC);
      `);
      const current = (this.db.prepare('PRAGMA table_info(analyses)').all() as Row[]).map(row => String(row.name));
      for (const [column, type] of [['schema_version', 'INTEGER'], ['deployment_hash', 'TEXT'], ['score_type', 'TEXT']]) if (!current.includes(column)) this.db.exec(`ALTER TABLE analyses ADD COLUMN ${column} ${type}`);
      this.db.exec('PRAGMA user_version=2; COMMIT; PRAGMA foreign_keys=ON;');
      this.db.prepare("UPDATE analyses SET status='failed', error='The previous service stopped during this task. Please retry', completed_at=? WHERE status IN ('queued','running')").run(new Date().toISOString());
    } catch (error) {
      this.db.close();
      if (error instanceof Error && /locked|busy/i.test(error.message)) throw new HttpError(409, 'Another running service owns this data directory. Open that service or use a separate CAMPUS_DATA_DIR for testing');
      throw error;
    }
  }
  private transaction<T>(fn: () => T): T {
    this.db.exec('BEGIN IMMEDIATE');
    try { const value = fn(); this.db.exec('COMMIT'); return value; }
    catch (error) { this.db.exec('ROLLBACK'); throw error; }
  }
  createSessions(inputs: SessionInput[]): SessionRecord[] {
    return this.transaction(() => inputs.map(input => {
      const record = { ...input, id: randomUUID(), createdAt: new Date().toISOString(), messageCount: input.messages.length + Number(Boolean(input.includePost)) };
      this.db.prepare('INSERT INTO sessions(id,body) VALUES(?,?)').run(record.id, JSON.stringify(record));
      return record;
    }));
  }
  getSession(id: string): SessionRecord {
    const row = this.db.prepare('SELECT body FROM sessions WHERE id=?').get(id) as Row | undefined;
    if (!row) throw new HttpError(404, 'Conversation not found');
    return JSON.parse(String(row.body)) as SessionRecord;
  }
  createAnalyses(sessionIds: string[]): AnalysisRecord[] {
    return this.transaction(() => sessionIds.map(id => {
      const session = this.getSession(id);
      const aid = randomUUID();
      this.db.prepare("INSERT INTO analyses(id,session_id,title,source,message_count,status,created_at,model_version) VALUES(?,?,?,?,?,'queued',?,?)")
        .run(aid, session.id, session.title, session.source, session.messageCount, new Date().toISOString(), 'Model not loaded');
      return this.getAnalysis(aid);
    }));
  }
  private map(row: Row): AnalysisRecord {
    return { id: String(row.id), sessionId: String(row.session_id), title: String(row.title), source: row.source as AnalysisRecord['source'], messageCount: Number(row.message_count), status: row.status as AnalysisRecord['status'], createdAt: String(row.created_at), completedAt: row.completed_at as string | null, prediction: row.prediction as number | null, riskScore: row.risk_score as number | null, topicCount: row.topic_count as number | null, error: row.error as string | null, modelHash: row.model_hash as string | null, modelVersion: String(row.model_version), threshold: row.threshold as number | null, durationMs: row.duration_ms as number | null, schemaVersion: row.schema_version === 2 ? 2 : 1, deploymentHash: row.deployment_hash as string | null, scoreType: row.score_type as string | null, review: { status: row.review_status as ReviewStatus, note: String(row.review_note), updatedAt: row.review_updated_at as string | null } };
  }
  getAnalysis(id: string): AnalysisRecord {
    const row = this.db.prepare('SELECT * FROM analyses WHERE id=?').get(id) as Row | undefined;
    if (!row) throw new HttpError(404, 'Analysis record not found');
    return this.map(row);
  }
  detail(id: string): AnalysisDetail {
    const analysis = this.getAnalysis(id);
    const row = this.db.prepare('SELECT result FROM analyses WHERE id=?').get(id) as Row;
    return { analysis, session: this.getSession(analysis.sessionId), result: row.result ? JSON.parse(String(row.result)) as Prediction : null };
  }
  list(filters: Filters = {}): AnalysisRecord[] {
    const clauses: string[] = [];
    const values: (string | number)[] = [];
    if (filters.query) { clauses.push("(title LIKE ? ESCAPE '\\' OR review_note LIKE ? ESCAPE '\\')"); const query = '%' + filters.query.replace(/[\\%_]/g, '\\$&') + '%'; values.push(query, query); }
    if (filters.prediction !== undefined) { clauses.push('prediction=?'); values.push(filters.prediction); }
    if (filters.from) { clauses.push('created_at>=?'); values.push(new Date(filters.from + 'T00:00:00+08:00').toISOString()); }
    if (filters.to) { clauses.push('created_at<?'); values.push(new Date(new Date(filters.to + 'T00:00:00+08:00').getTime() + 86400000).toISOString()); }
    const rows = this.db.prepare('SELECT * FROM analyses' + (clauses.length ? ' WHERE ' + clauses.join(' AND ') : '') + ' ORDER BY created_at DESC,rowid DESC').all(...values) as Row[];
    return rows.map(row => this.map(row));
  }
  markRunning(id: string) { this.db.prepare("UPDATE analyses SET status='running' WHERE id=? AND status='queued'").run(id); }
  succeed(id: string, result: Prediction, identity: ModelIdentity, durationMs: number) {
    if (identity.schemaVersion !== (isPredictionV2(result) ? 2 : 1) || (isPredictionV2(result) && result.modelVersion !== identity.modelVersion)) throw new Error('Result does not match the actual model identity');
    this.db.prepare("UPDATE analyses SET status='succeeded',completed_at=?,prediction=?,risk_score=?,topic_count=?,model_hash=?,model_version=?,schema_version=?,deployment_hash=?,score_type=?,threshold=?,duration_ms=?,result=?,error=NULL WHERE id=? AND status='running'")
      .run(new Date().toISOString(), result.prediction, predictionScore(result), result.topics.length, identity.modelHash, identity.modelVersion, identity.schemaVersion, identity.deploymentHash, isPredictionV2(result) ? result.scoreType : 'legacy_conversation_score', isPredictionV2(result) ? null : result.threshold, durationMs, JSON.stringify(result), id);
  }
  fail(id: string, error: string) { this.db.prepare("UPDATE analyses SET status='failed',completed_at=?,error=? WHERE id=? AND status IN ('queued','running')").run(new Date().toISOString(), error.slice(0, 1000), id); }
  review(id: string, status: ReviewStatus, note: string): Review {
    this.getAnalysis(id);
    const updatedAt = new Date().toISOString();
    this.db.prepare('UPDATE analyses SET review_status=?,review_note=?,review_updated_at=? WHERE id=?').run(status, note, updatedAt, id);
    return { status, note, updatedAt };
  }
  delete(id: string) {
    const record = this.getAnalysis(id);
    if (record.status === 'queued' || record.status === 'running') throw new HttpError(409, 'Wait until the task finishes before deleting it');
    this.transaction(() => {
      this.db.prepare('DELETE FROM analyses WHERE id=?').run(id);
      this.db.prepare('DELETE FROM sessions WHERE id=? AND NOT EXISTS(SELECT 1 FROM analyses WHERE session_id=?)').run(record.sessionId, record.sessionId);
    });
  }
  counts() {
    return { sessions: Number((this.db.prepare('SELECT count(*) AS n FROM sessions').get() as Row).n), analyses: Number((this.db.prepare('SELECT count(*) AS n FROM analyses').get() as Row).n) };
  }
  close() { if (!this.closed) { this.closed = true; this.db.close(); } }
}
