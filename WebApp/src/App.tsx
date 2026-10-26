import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Activity, ArrowDownToLine, ArrowRight, BarChart3, Check, CheckCircle2, ChevronDown, ChevronRight, CircleHelp, Clock3, FileJson, FileText, FlaskConical, History, Layers3, LoaderCircle, MessageSquareText, Plus, RefreshCw, Search, ShieldCheck, Trash2, Upload, X, TriangleAlert } from 'lucide-react';
import type { AnalysisDetail, AnalysisRecord, Benchmarks, Example, Health, ImportPreview, Prediction, PredictionV2, Review, ReviewStatus, SessionRecord, Source } from '../shared/types';
import { isPredictionV2 } from '../shared/types';
import { errorMessage, post, request } from './api';
import { sourceParts, TopicCard, type SourceHighlight } from './TopicCard';

type Page = 'new' | 'history' | 'benchmarks' | 'analysis';
const sourceNames: Record<Source, string> = { manual: 'Pasted text', upload: 'Imported file', demo: 'Demo example', sccd: 'SCCD data' };
const statusNames = { queued: 'Queued', running: 'Analyzing', succeeded: 'Completed', failed: 'Analysis failed' };
const reviewNames: Record<ReviewStatus, string> = { pending: 'Pending review', risk: 'Risk identified', clear: 'No risk identified', uncertain: 'Uncertain' };
const categoryNames = { attack: 'Attack / offense', hate: 'Hate / discrimination', spam_scam: 'Spam / scam promotion' };
const reviewOptions: ReviewStatus[] = ['pending', 'risk', 'clear', 'uncertain'];
const dateText = (value: string | null) => value ? new Date(value).toLocaleString('en-US', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—';
const score = (value: number | null | undefined, digits = 1) => value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`;
const num = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits });
const riskDisplay = (value: number | null | undefined) => value != null && Number.isFinite(value) && value > 0 && value < 0.001 ? '<0.001' : num(value, 3);
const riskExact = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : String(value);
const thresholdDisplay = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(3);
const duration = (value: number | null) => value == null ? '—' : value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;

function readRoute() {
  const route = window.location.hash.replace(/^#\/?/, '').split('/');
  return { page: (['history', 'benchmarks', 'analysis'].includes(route[0]) ? route[0] : 'new') as Page, id: route[0] === 'analysis' ? route[1] : undefined };
}
function navigate(page: Page, id?: string) { window.location.hash = page === 'analysis' && id ? `analysis/${encodeURIComponent(id)}` : page; }
function InlineError({ message, onClose }: { message: string | null; onClose?: () => void }) {
  if (!message) return null;
  return <div className="notice notice-error" role="alert"><TriangleAlert size={18} /><span>{message}</span>{onClose && <button className="icon-button" aria-label="Dismiss message" onClick={onClose}><X size={17} /></button>}</div>;
}
function Loading({ children = 'Loading…' }: { children?: ReactNode }) { return <div className="loading-state" role="status"><LoaderCircle className="spin" size={23} /><span>{children}</span></div>; }
function Empty({ title, description, children }: { title: string; description: string; children?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon"><MessageSquareText size={28} /></div><h3>{title}</h3><p>{description}</p>{children}</div>;
}
function PageHeading({ number, title, description, action }: { number: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow"><span>{number}</span> CST-MIL WORKSPACE</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}
function Badge({ type, children }: { type?: string; children: ReactNode }) { return <span className={`badge ${type || ''}`}>{children}</span>; }
function ModelStatus({ health, error }: { health: Health | null; error: string | null }) {
  const ready = health?.worker.status === 'ready';
  return <div className="model-status"><span className={`status-dot ${ready ? 'ready' : error || health?.worker.status === 'error' ? 'failed' : ''}`} /><div><strong>{ready ? 'Local model ready' : error ? 'Connection lost' : health?.worker.status === 'error' ? 'Model unavailable' : 'Loading local model'}</strong><span>{ready ? 'Processed on this device' : 'Prepare your conversation'}</span></div></div>;
}

export default function App() {
  const [route, setRoute] = useState(readRoute);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [lastId, setLastId] = useState<string | null>(() => localStorage.getItem('campus-last-analysis'));
  const [toast, setToast] = useState('');
  useEffect(() => { const handle = () => setRoute(readRoute()); window.addEventListener('hashchange', handle); return () => window.removeEventListener('hashchange', handle); }, []);
  useEffect(() => {
    let active = true;
    const check = async () => { try { const value = await request<Health>('/health'); if (active) { setHealth(value); setHealthError(null); } } catch (error) { if (active) setHealthError(errorMessage(error)); } };
    void check(); const timer = window.setInterval(check, 6000); return () => { active = false; window.clearInterval(timer); };
  }, []);
  useEffect(() => { if (route.id) { setLastId(route.id); localStorage.setItem('campus-last-analysis', route.id); } window.scrollTo({ top: 0 }); }, [route]);
  useEffect(() => { if (!toast) return; const timer = window.setTimeout(() => setToast(''), 6000); return () => window.clearTimeout(timer); }, [toast]);
  const openAnalysis = (id: string) => navigate('analysis', id);
  return <div className="app-shell">
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <aside className="sidebar">
      <a className="brand" href="#new" aria-label="CST-MIL home"><span className="brand-mark"><MessageSquareText size={25} /></span><div><strong>CST-MIL</strong><span>Topics and risk evidence</span></div></a>
      <div className="nav-caption">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <button className={route.page === 'new' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('new')}><Plus size={19} /><span>New analysis</span><ChevronRight className="nav-chevron" size={15} /></button>
        <button className={route.page === 'analysis' ? 'nav-item active' : 'nav-item'} onClick={() => lastId ? openAnalysis(lastId) : navigate('analysis')}><Layers3 size={19} /><span>Analysis results</span></button>
        <button className={route.page === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('history')}><History size={19} /><span>History</span></button>
        <button className={route.page === 'benchmarks' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('benchmarks')}><BarChart3 size={19} /><span>Research code</span></button>
      </nav>
      <div className="sidebar-note"><FlaskConical size={18} /><p>Start with a conversation.<br />Explore topics. Review context.</p><span>LOCAL TEXT ANALYSIS</span></div>
      <div className="sidebar-bottom"><ModelStatus health={health} error={healthError} /><span className="local-label">LOCAL · HUMAN REVIEW</span></div>
    </aside>
    <div className="workspace"><header className="topbar"><span>Short-text analysis</span><span className="topbar-right"><ShieldCheck size={15} /> Local workspace</span></header>
      <main id="main-content" className="main-content">
        {healthError && <InlineError message={healthError} />}
        {health?.worker.status === 'error' && <details className="worker-warning"><summary><TriangleAlert size={16} /> Model could not load. Show details</summary><p>{health.worker.error || 'Check the Python environment and model files, then restart.'}</p></details>}
        {route.page === 'new' && <NewAnalysis health={health} onCreated={(id, count) => { if (count > 1) setToast(`Submitted ${count} conversations. View the remaining tasks in History.`); openAnalysis(id); }} />}
        {route.page === 'analysis' && (route.id ? <Results key={route.id} id={route.id} notify={setToast} /> : <><PageHeading number="02" title="Analysis results" description="Review topics and candidate evidence in the context of the full conversation." /><div className="card"><Empty title="Analyze a conversation first" description="Submit a conversation to see model outputs and your review notes here."><button className="button primary" onClick={() => navigate('new')}><Plus size={17} />New analysis</button></Empty></div></>)}
        {route.page === 'history' && <HistoryPage onSelect={openAnalysis} notify={setToast} onDeleted={id => { if (lastId === id) setLastId(null); }} />}
        {route.page === 'benchmarks' && <BenchmarkPage />}
      </main><footer className="workspace-footer"><span>CST-MIL · Local short-text analysis</span><span>Model outputs support understanding. People make the final judgment.</span></footer>
    </div>
    {toast && <div className="toast" role="status"><CheckCircle2 size={19} /><span>{toast}</span><button className="icon-button" onClick={() => setToast('')} aria-label="Dismiss notification"><X size={16} /></button></div>}
  </div>;
}

function NewAnalysis({ health, onCreated }: { health: Health | null; onCreated: (id: string, count: number) => void }) {
  const [mode, setMode] = useState<'paste' | 'upload' | 'examples'>('paste');
  const [title, setTitle] = useState('');
  const [postText, setPostText] = useState('');
  const [includePost, setIncludePost] = useState(false);
  const [content, setContent] = useState('');
  const [format, setFormat] = useState<'csv' | 'json'>('csv');
  const [filename, setFilename] = useState('');
  const [examples, setExamples] = useState<Example[]>([]);
  const [examplesLoading, setExamplesLoading] = useState(true);
  const [selectedExample, setSelectedExample] = useState<Example | null>(null);
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [previewIndex, setPreviewIndex] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<'preview' | 'submit' | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  useEffect(() => { let active = true; request<{ examples: Example[] }>('/examples').then(data => { if (active) setExamples(data.examples); }).catch(e => { if (active) setError(errorMessage(e)); }).finally(() => { if (active) setExamplesLoading(false); }); return () => { active = false; }; }, []);
  const invalidate = () => { setPreview(null); setError(null); };
  const changeMode = (value: typeof mode) => { setMode(value); invalidate(); setContent(''); setFilename(''); setSelectedExample(null); setTitle(''); setPostText(''); setIncludePost(false); };
  const loadFile = async (file?: File) => {
    if (!file) return;
    invalidate();
    if (file.size > 5 * 1024 * 1024) { setError('The file exceeds 5 MB. Split the conversations before importing.'); return; }
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension !== 'csv' && extension !== 'json') { setError('Choose a CSV or JSON file.'); return; }
    try { const text = await file.text(); if (text.includes('\uFFFD')) { setError('The file may not use UTF-8. Save it as UTF-8 CSV or JSON and try again.'); return; } setContent(text); setFilename(file.name); setFormat(extension); } catch { setError('Could not read the file. Select it again.'); }
  };
  const selectExample = (example: Example) => { invalidate(); setSelectedExample(example); setTitle(example.session.title); setPostText(example.session.postText || ''); setIncludePost(Boolean(example.session.includePost)); };
  const makePreview = async () => {
    setBusy('preview'); setError(null); setPreview(null);
    try {
      const exampleSession = selectedExample ? { ...selectedExample.session, title: title.trim() || selectedExample.session.title, postText, includePost } : null;
      const data = await post<ImportPreview>('/import/preview', { format: mode === 'examples' ? 'json' : mode === 'upload' ? format : 'paste', content: mode === 'examples' ? JSON.stringify(exampleSession) : content, ...(title.trim() ? { title: title.trim() } : {}), ...(mode !== 'upload' ? { postText, includePost } : {}) });
      setPreview(data); setPreviewIndex(0); setShowAll(false);
    } catch (e) { setError(errorMessage(e)); } finally { setBusy(null); }
  };
  const submit = async () => {
    if (!preview) return;
    setBusy('submit'); setError(null);
    try {
      const saved = await post<{ sessions: SessionRecord[] }>('/sessions', { sessions: preview.sessions });
      const tasks = await post<{ analyses: AnalysisRecord[] }>('/analyses', { sessionIds: saved.sessions.map(session => session.id) });
      if (!tasks.analyses.length) throw new Error('No analysis task was created. Please try again.');
      onCreated(tasks.analyses[0].id, tasks.analyses.length);
    } catch (e) { setError(errorMessage(e)); } finally { setBusy(null); }
  };
  const downloadTemplate = (kind: 'csv' | 'json') => {
    const text = kind === 'csv' ? '\uFEFFsession_id,title,id,speaker,text,parent_id,order,post_text\nexample,Course discussion,m1,Alex,What time is our group presentation tomorrow?,,1,Course group chat\nexample,Course discussion,m2,Blair,At 2 pm. Let us arrive ten minutes early.,m1,2,Course group chat\n' : JSON.stringify({ title: 'Course discussion', postText: 'Course group chat', includePost: false, messages: [{ id: 'm1', speaker: 'Alex', text: 'What time is our group presentation tomorrow?' }, { id: 'm2', speaker: 'Blair', text: 'At 2 pm. Let us arrive ten minutes early.', parentId: 'm1' }] }, null, 2);
    const url = URL.createObjectURL(new Blob([text], { type: kind === 'csv' ? 'text/csv;charset=utf-8' : 'application/json;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `conversation-template.${kind}`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const current = preview?.sessions[previewIndex];
  const canPreview = mode === 'examples' ? Boolean(selectedExample) : Boolean(content.trim());
  return <>
    <PageHeading number="01" title="Explore topics and risk clues" description="Import a conversation, discover discussion topics, and locate passages for closer review." />
    <div className="workflow-strip" aria-label="Analysis workflow"><span className={!preview ? 'current' : 'complete'}><b>{preview ? <Check size={13} /> : '1'}</b>Prepare input</span><i /><span className={preview ? 'current' : ''}><b>2</b>Confirm input</span><i /><span><b>3</b>Analyze and review</span><div className="workflow-note"><ShieldCheck size={15} /> Local processing on your device</div></div>
    <div className="new-layout">
      <section className="card input-card" data-report-region="input">
        <div className="card-heading"><div><h2>Conversation</h2><p>Choose an input method</p></div><span className="small-label">INPUT</span></div>
        <div className="input-tabs" role="tablist" aria-label="Conversation input method">{([['paste', 'Paste text', MessageSquareText], ['upload', 'Upload file', Upload], ['examples', 'Examples', FlaskConical]] as const).map(([key, label, Icon]) => <button key={key} role="tab" aria-selected={mode === key} className={mode === key ? 'selected' : ''} onClick={() => changeMode(key)} disabled={Boolean(busy)}><Icon size={17} />{label}</button>)}</div>
        <div className="input-panel" role="tabpanel">
          {mode === 'examples' && <div className="example-section">{examplesLoading ? <Loading>Loading examples…</Loading> : <><p className="field-help">Campus examples are fictional demonstrations, not evaluation data. They include ordinary discussion, potential risks and contextual ambiguity. Optional SCCD examples use locally available source data.</p><div className="example-grid">{examples.map((example, index) => <button key={example.id} className={`example-card ${selectedExample?.id === example.id ? 'selected' : ''}`} onClick={() => selectExample(example)} disabled={Boolean(busy)}><div className="example-top"><span className="example-number">{String(index + 1).padStart(2, '0')}</span>{selectedExample?.id === example.id && <CheckCircle2 size={17} />}</div><strong>{example.title}</strong><div className="example-source"><Badge type={example.session.source === 'demo' ? 'demo' : ''}>{sourceNames[example.session.source]}</Badge></div><p>{example.description}</p><div className="example-tags">{example.tags.map(tag => <span key={tag}>{tag}</span>)}</div></button>)}</div></>}</div>}
          {mode === 'upload' ? <>
            <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void loadFile(e.dataTransfer.files[0]); }}><span className="upload-icon"><Upload size={27} /></span><h3>{filename || 'Drop a conversation file here'}</h3><p>{filename ? 'File loaded. Validate and preview it below.' : 'UTF-8 CSV or JSON, up to 5 MB per file'}</p><button className="button secondary" onClick={() => fileRef.current?.click()} disabled={Boolean(busy)}>{filename ? 'Choose another file' : 'Choose file'}</button><input ref={fileRef} type="file" accept=".csv,.json,application/json,text/csv" className="visually-hidden" aria-label="Upload conversation file" onChange={e => void loadFile(e.target.files?.[0])} /></div>
            <div className="template-row"><span>Need a starting point?</span><button className="text-button" onClick={() => downloadTemplate('csv')}><ArrowDownToLine size={14} /> CSV template</button><button className="text-button" onClick={() => downloadTemplate('json')}><ArrowDownToLine size={14} /> JSON template</button></div>
            <details className="format-details"><summary>File format guide <ChevronDown size={15} /></summary><p>CSV requires a text column. Optional fields: id, speaker, parent_id, time, order, session_id, title, post_text. Use session_id for multiple conversations. JSON uses title and messages (with id and text); postText holds the original post and includePost controls its inclusion in analysis.</p><p>Import up to 100 conversations, with up to 2,000 messages each. Preview checks duplicate IDs, empty text and missing reply targets.</p></details>
          </> : (mode === 'paste' || selectedExample) && <>
            <label className="field-label" htmlFor="session-title">Conversation title <span>Make it easy to find later</span></label><input id="session-title" value={title} onChange={e => { setTitle(e.target.value); invalidate(); }} placeholder="Example: Student club event planning" maxLength={200} disabled={Boolean(busy)} />
            {mode === 'paste' ? <><div className="field-row"><label className="field-label" htmlFor="conversation">Comments</label><span>{content.split('\n').filter(line => line.trim()).length} non-empty lines</span></div><textarea id="conversation" className="conversation-input" value={content} onChange={e => { setContent(e.target.value); invalidate(); }} placeholder={'One comment per line. Optionally use Speaker: comment.\n\nAlex: How is our group report coming along?\nBlair: The charts are ready. Let us check them tonight.'} disabled={Boolean(busy)} /><p className="field-help">Keep the original wording and order. Empty lines are ignored. Import a file to preserve reply links.</p></> : <div className="example-selected"><CheckCircle2 size={16} /><span>Selected: {selectedExample?.title} · {selectedExample?.session.messages.length} messages. Validate to preview the complete conversation.</span></div>}
            <details className="post-details" open={postText ? true : undefined}><summary>Original post and context <span>Optional</span><ChevronDown size={15} /></summary><label className="visually-hidden" htmlFor="post-text">Original post</label><textarea id="post-text" rows={3} value={postText} onChange={e => { setPostText(e.target.value); invalidate(); }} placeholder="Add the original post or background for this conversation." disabled={Boolean(busy)} /><label className="checkbox-label"><input type="checkbox" checked={includePost} onChange={e => { setIncludePost(e.target.checked); invalidate(); }} disabled={!postText.trim() || Boolean(busy)} /><span>Include the original post as the first message</span></label><p className="field-help">By default, the post provides context and only comments are analyzed.</p></details>
          </>}
        </div>
        <div className="input-actions"><span>Check your input before analysis</span><button className="button primary" onClick={() => void makePreview()} disabled={!canPreview || Boolean(busy)}>{busy === 'preview' ? <LoaderCircle className="spin" size={17} /> : <CheckCircle2 size={17} />}{busy === 'preview' ? 'Validating…' : 'Validate and preview'}</button></div>
      </section>
      <aside className="guidance-column"><section className="guidance-card"><span className="section-kicker">FROM TEXT TO EVIDENCE</span><h2>One conversation.<br />Three perspectives.</h2><div className="guidance-item"><Layers3 size={20} /><div><h3>What is being discussed?</h3><p>Find topic keywords and the messages that support them.</p></div></div><div className="guidance-item"><Activity size={20} /><div><h3>What needs closer review?</h3><p>Inspect risk categories and candidate evidence in context.</p></div></div><div className="guidance-item"><ShieldCheck size={20} /><div><h3>Record your review</h3><p>Save review decisions and notes, then export the analysis.</p></div></div></section><div className="care-note"><CircleHelp size={17} /><p>Review outputs in context. Scores alone cannot distinguish jokes, quotations and disagreement. Risk categories are separate from discussion topics.</p></div><div className="method-stamp"><span>CST-MIL</span><p>Lightweight topic discovery<br />and conversation risk analysis</p></div></aside>
    </div>
    <InlineError message={error} onClose={() => setError(null)} />
    {preview && current && <section className="card preview-card" data-report-region="input-preview"><div className="card-heading"><div><div className="section-kicker">Confirm input</div><h2>Conversation preview <span className="heading-count">{preview.sessions.length} conversations · {preview.sessions.reduce((total, session) => total + session.messages.length, 0)} comments</span></h2></div><Badge type="good"><Check size={13} />Validated</Badge></div>
      {preview.warnings.map((warning, i) => <div className="notice notice-warn" key={i}><TriangleAlert size={16} /><span>{warning}</span></div>)}
      {preview.sessions.length > 1 && <label className="preview-picker">Select conversation<select value={previewIndex} onChange={e => { setPreviewIndex(Number(e.target.value)); setShowAll(false); }}>{preview.sessions.map((session, index) => <option key={index} value={index}>{index + 1}. {session.title}</option>)}</select></label>}
      <div className="preview-title"><h3>{current.title}</h3><Badge>{sourceNames[current.source]}</Badge></div>
      {current.postText && <div className="post-context"><strong>Original post <small>{current.includePost ? 'Included in analysis' : 'Context only'}</small></strong><p>{current.postText}</p></div>}
      <div className="preview-messages">{current.messages.slice(0, showAll ? undefined : 4).map((message, index) => <div className="preview-message" key={message.id}><span className="message-index">{String(index + 1).padStart(2, '0')}</span><div><div className="message-meta"><strong>{message.speaker || 'Unspecified speaker'}</strong>{message.parentId && <span>Reply to #{current.messages.findIndex(item => item.id === message.parentId) + 1}</span>}</div><p>{message.text}</p></div></div>)}</div>
      {current.messages.length > 4 && <button className="text-button show-more" onClick={() => setShowAll(!showAll)}>{showAll ? 'Show first 4 messages' : `Showing first 4 · Show all ${current.messages.length} messages`}<ChevronDown size={15} /></button>}
      <div className="preview-footer"><p>{health?.worker.status === 'ready' ? 'Once confirmed, the local model will analyze this conversation.' : health?.worker.status === 'error' ? 'The model is unavailable. Resubmitting will attempt to restart it.' : 'The model is loading. Keep this preview and start when it is ready.'}</p><button className="button primary" disabled={Boolean(busy) || !health || health.worker.status === 'starting'} onClick={() => void submit()}>{busy === 'submit' ? <LoaderCircle className="spin" size={17} /> : <Activity size={17} />}{busy === 'submit' ? 'Submitting…' : health?.worker.status === 'error' ? 'Retry analysis' : `Start analysis${preview.sessions.length > 1 ? ` ${preview.sessions.length} conversations` : ''}`}<ArrowRight size={17} /></button></div>
    </section>}
  </>;
}

function RiskSummary({ result }: { result: Prediction }) {
  if (isPredictionV2(result)) return <section className={`card risk-summary ${result.prediction === 1 ? 'elevated' : ''}`}><div className="stat-label"><Activity size={17} />Highest message clue score</div><div className="risk-value" title={`Exact highest message clue score: ${riskExact(result.overallEvidenceScore)}`}>{riskDisplay(result.overallEvidenceScore)}<span>{result.prediction === 1 ? 'At least one category reached its threshold' : 'No category reached its threshold'}</span></div><div className="risk-track"><div style={{ width: `${result.overallEvidenceScore * 100}%` }} /></div><div className="risk-scale"><span>0</span><span>Maximum across messages and categories</span><span>1</span></div></section>;
  return <section className={`card risk-summary ${result.prediction === 1 ? 'elevated' : ''}`}><div className="stat-label"><Activity size={17} />Conversation risk score</div><div className="risk-value" title={`Exact conversation score: ${riskExact(result.risk_probability)}`}>{riskDisplay(result.risk_probability)}<span>{result.prediction === 1 ? 'Potential cyberbullying risk' : 'Below risk threshold'}</span></div><div className="risk-track"><div style={{ width: `${Math.max(0, Math.min(100, result.risk_probability * 100))}%` }} /><span style={{ left: `${Math.max(0, Math.min(100, result.threshold * 100))}%` }} /></div><div className="risk-scale"><span>Low</span><span title={`Exact decision threshold: ${riskExact(result.threshold)}`}>Decision threshold {thresholdDisplay(result.threshold)}</span><span>High</span></div></section>;
}

function CategoryPanel({ result, focusMessage }: { result: PredictionV2; focusMessage: (id: string) => void }) {
  return <>
    <section className="card categories-section" data-report-region="risk-categories"><div className="card-heading"><div><h2>Fixed risk categories</h2><p>Each category uses its highest message score and its own development-set threshold.</p></div><ShieldCheck size={21} className="muted" /></div><div className="category-grid">{result.riskCategories.map(category => <article className={`category-card ${category.detected ? 'detected' : ''}`} key={category.id}><div className="category-heading"><h3>{categoryNames[category.id]}</h3><Badge type={category.detected ? 'candidate' : ''}>{category.detected ? 'Threshold reached' : 'Below threshold'}</Badge></div><div className="category-value"><strong title={`Highest category score: ${riskExact(category.score)}`}>{riskDisplay(category.score)}</strong><span title={`Exact development threshold: ${riskExact(category.threshold)}`}>Threshold {thresholdDisplay(category.threshold)}</span></div>{category.message_ids.length > 0 ? <button className="text-button" onClick={() => focusMessage(category.evidence[0]?.comment_id ?? category.message_ids[0])}>Locate source evidence<ArrowRight size={14} /></button> : <span className="muted">No messages above this threshold</span>}</article>)}</div></section>
    <details className="card conversation-reference"><summary><span><MessageSquareText size={17} /><strong>SCCD conversation reference</strong><small>Separate conversation classifier</small></span><ChevronDown size={16} /></summary><div className="reference-values"><strong title={`Exact SCCD reference score: ${riskExact(result.conversationRisk.score)}`}>{riskDisplay(result.conversationRisk.score)}</strong><span title={`Exact conversation threshold: ${riskExact(result.conversationRisk.threshold)}`}>Reference threshold {thresholdDisplay(result.conversationRisk.threshold)}</span><Badge>{result.conversationRisk.prediction === 1 ? 'Conversation threshold reached' : 'Below conversation threshold'}</Badge></div><p>This separate classifier was trained on SCCD conversations. Its score is a reference and is not a calibrated probability for arbitrary domains.</p></details>
  </>;
}

function Results({ id, notify }: { id: string; notify: (message: string) => void }) {
  const [detail, setDetail] = useState<AnalysisDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [selectedMessage, setSelectedMessage] = useState<string | null>(null);
  const [selectedSpan, setSelectedSpan] = useState<SourceHighlight | null>(null);
  const [topicFilter, setTopicFilter] = useState<number | null>(null);
  const [showAllMessages, setShowAllMessages] = useState(false);
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus>('pending');
  const [reviewNote, setReviewNote] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const reviewLoaded = useRef(false);
  useEffect(() => {
    let active = true; let timer: number | undefined;
    async function load() {
      try {
        const data = await request<AnalysisDetail>(`/analyses/${encodeURIComponent(id)}`);
        if (!active) return;
        setDetail(data); setError(null);
        if (!reviewLoaded.current) { setReviewStatus(data.analysis.review.status); setReviewNote(data.analysis.review.note); reviewLoaded.current = true; }
        if (data.analysis.status === 'queued' || data.analysis.status === 'running') timer = window.setTimeout(load, 1000);
      } catch (e) { if (active) setError(errorMessage(e)); }
    }
    void load(); return () => { active = false; window.clearTimeout(timer); };
  }, [id, reload]);
  const focusMessage = (messageId: string, highlight?: SourceHighlight) => {
    setTopicFilter(null); setShowAllMessages(true); setSelectedMessage(messageId);
    setSelectedSpan(highlight ?? null);
    window.setTimeout(() => { const target = document.getElementById(`message-${messageId}`); target?.scrollIntoView({ behavior: 'smooth', block: 'center' }); target?.focus({ preventScroll: true }); }, 70);
  };
  const saveReview = async () => {
    setSaving(true); setSaveError(null);
    try {
      const data = await request<{ review: Review }>(`/analyses/${encodeURIComponent(id)}/review`, { method: 'PATCH', body: JSON.stringify({ status: reviewStatus, note: reviewNote }) });
      setDetail(current => current ? { ...current, analysis: { ...current.analysis, review: data.review } } : current); notify('Review saved.');
    } catch (e) { setSaveError(errorMessage(e)); } finally { setSaving(false); }
  };
  if (!detail) return <><PageHeading number="02" title="Analysis results" description="Inspect topics and risk candidates alongside the complete conversation." />{error ? <><InlineError message={error} /><button className="button secondary" onClick={() => setReload(value => value + 1)}><RefreshCw size={16} />Reload</button></> : <Loading>Loading analysis…</Loading>}</>;
  const { analysis, session, result } = detail;
  const unified = result && isPredictionV2(result) ? result : null;
  const pending = analysis.status === 'queued' || analysis.status === 'running';
  const dirty = reviewStatus !== analysis.review.status || reviewNote !== analysis.review.note;
  const filteredMessages = result?.message_scores.filter(message => topicFilter == null || message.topic_ids.includes(topicFilter)) || [];
  const messages = filteredMessages.slice(0, showAllMessages ? undefined : 30);
  const topicOrder = new Map(result?.topics.map((topic, index) => [topic.topic_id, index + 1]));
  return <>
    <PageHeading number="02" title="Analysis results" description="Check topics, risk categories and source text, then record your judgment." action={<button className="button secondary" onClick={() => navigate('new')}><Plus size={17} />New analysis</button>} />
    <InlineError message={error} />
    <section className="card result-title-card" data-report-region="result-identity"><div><div className="result-title-line"><Badge type={session.source === 'demo' ? 'demo' : ''}>{sourceNames[session.source]}</Badge><span className="muted">{dateText(analysis.createdAt)}</span><Badge type={analysis.status}>{statusNames[analysis.status]}</Badge></div><h2>{session.title}</h2><p>{analysis.messageCount} analyzed messages <span>·</span> {session.includePost ? 'Original post included' : 'Original post is context only'} {analysis.durationMs != null && <><span>·</span>Duration {duration(analysis.durationMs)}</>}</p></div>{analysis.status === 'succeeded' && <div className="export-buttons"><a className="button secondary small" href={`/api/analyses/${encodeURIComponent(id)}/export?format=csv`} download><ArrowDownToLine size={15} />Export CSV</a><a className="button secondary small" href={`/api/analyses/${encodeURIComponent(id)}/export?format=json`} download><FileJson size={15} />JSON</a></div>}</section>
    {session.source === 'demo' && <div className="notice notice-info compact"><FlaskConical size={16} /><span>This is a fictional demonstration. These are actual model outputs, not an evaluation of performance on campus conversations.</span></div>}
    {pending && <section className="card job-progress"><div className="progress-orbit"><LoaderCircle size={39} className="spin" /></div><h2>{analysis.status === 'queued' ? 'Conversation queued' : 'Analyzing your conversation'}</h2><p>{analysis.status === 'queued' ? 'Tasks run in submission order. This page updates automatically.' : 'The local model is extracting topics and risk clues. Please wait.'}</p><span className="badge">{analysis.status === 'queued' ? 'Queued' : 'Running local inference'}</span><button className="text-button" onClick={() => navigate('history')}>View other tasks in History<ArrowRight size={15} /></button>{error && <button className="button secondary" onClick={() => setReload(value => value + 1)}><RefreshCw size={16} />Resume status updates</button>}</section>}
    {analysis.status === 'failed' && <section className="card"><Empty title="Analysis could not finish" description={analysis.error || 'The model encountered an error.'}><button className="button primary" onClick={() => navigate('new')}>New analysis<ArrowRight size={16} /></button></Empty></section>}
    {result && analysis.status === 'succeeded' && <>
      <div className="summary-grid"><RiskSummary result={result} /><section className="card stat-card"><div className="stat-label"><Layers3 size={17} />Topics discovered</div><strong>{result.topics.length}<small> topics</small></strong><p>{result.topics.length ? (unified ? 'Semantic groups and direct-reply associations' : 'Keywords and associated messages') : 'No clear topics found'}</p></section><section className="card stat-card"><div className="stat-label"><MessageSquareText size={17} />Topic coverage</div><strong>{result.coverage_details.assigned_messages}<small> / {result.coverage_details.total_messages}</small></strong><p>Analyzed messages assigned to a topic</p></section></div>
      <div className="notice interpretation-note"><CircleHelp size={17} /><span>{unified ? "Risk categories and discussion topics are computed separately. Scores indicate review clues; read candidate evidence in context. Expand the SCCD reference for its separate conversation score. " : "Legacy scores range from 0 to 1 and are neither accuracy nor event probabilities. Conversation scores summarize the whole discussion and differ from message scores. Review evidence in context. "}Hover over a score for its full value.</span></div>
      {unified && <CategoryPanel result={unified} focusMessage={focusMessage} />}
      <div className="results-upper" data-report-region="topics-evidence"><section className="card topics-section"><div className="card-heading"><div><h2>{unified ? "Discussion topics" : "Topic discovery"}</h2><p>Select a keyword to inspect its source. Topics are distinct from risk categories.</p></div><Layers3 size={21} className="muted" /></div>{result.topics.length === 0 ? <div className="zero-topics"><Layers3 size={26} /><h3>{unified ? "No clear topics found" : "No clear topics found"}</h3><p>{unified ? "Messages remain ungrouped when no clear topic emerges. Risk categories, candidate evidence and source text remain available." : "This conversation did not meet the topic extraction criteria. Review its score and candidate messages alongside the source text."}</p></div> : <div className="topic-list">{result.topics.map((topic, index) => <TopicCard topic={topic} index={index} key={topic.topic_id} onMessage={focusMessage} onSupport={support => focusMessage(support.message_id, { messageId: support.message_id, start: support.start, end: support.end, excerpt: support.excerpt })} onFilter={() => { setTopicFilter(topic.topic_id); setShowAllMessages(false); document.getElementById('conversation-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} />)}</div>}</section><section className="card evidence-section"><div className="card-heading"><div><h2>Top-3 candidate evidence</h2><p>Select evidence to inspect its surrounding context.</p></div><Search size={20} className="muted" /></div>{result.evidence.length ? <div className="evidence-list">{result.evidence.slice(0, 3).map((item, index) => <button className="evidence-card" key={`${item.comment_id}-${index}`} onClick={() => focusMessage(item.comment_id)}><div className="evidence-meta"><span className="evidence-rank">{String(index + 1).padStart(2, '0')}</span><strong>{item.speaker_id || 'Unspecified speaker'}</strong><ArrowRight size={15} /></div><p>{item.raw_text}</p><span className="evidence-score" title={`Exact message score: ${riskExact(item.risk_score)}`}>{unified ? "Message clue score" : "Message risk score"} {riskDisplay(item.risk_score)}</span></button>)}</div> : <div className="zero-topics"><p>No candidate evidence was returned.</p></div>}<p className="evidence-footnote">Candidates are ranked to help navigation. A higher rank does not confirm risk.</p></section></div>
      <section className="card conversation-section" id="conversation-panel" data-report-region="conversation"><div className="card-heading"><div><h2>Full conversation <span className="heading-count">{result.message_scores.length} messages</span></h2><p>Original message IDs, order and reply relationships are preserved.</p></div><label className="compact-select"><span className="visually-hidden">Filter messages by topic</span><select value={topicFilter ?? ''} onChange={e => { setTopicFilter(e.target.value === '' ? null : Number(e.target.value)); setShowAllMessages(false); }}><option value="">All messages</option>{result.topics.map((topic, index) => <option value={topic.topic_id} key={topic.topic_id}>Topic {index + 1} · {topic.keywords.slice(0, 2).join(' / ')}</option>)}</select></label></div>
        {session.postText && <div className="post-context"><strong>Original post <small>{session.includePost ? 'Included in analysis' : 'Context only'}</small></strong><p>{session.postText}</p></div>}
        {topicFilter != null && <div className="filter-indicator">Viewing topic {topicOrder.get(topicFilter)}: {filteredMessages.length} associated messages<button className="text-button" onClick={() => setTopicFilter(null)}>Show all<X size={13} /></button></div>}
        {selectedMessage && <div className="source-location" role="status"><Search size={14} /><span>Source location</span><code>Message {selectedMessage}</code>{selectedSpan && <span title="Zero-based Unicode codepoint span; end is exclusive">Character span [{selectedSpan.start}, {selectedSpan.end})</span>}<button className="text-button" onClick={() => { setSelectedMessage(null); setSelectedSpan(null); }}>Clear location<X size={13} /></button></div>}
        <div className="message-list">{messages.map((message, index) => { const evidenceIndex = result.evidence.findIndex(item => item.comment_id === message.comment_id); const highlighted = sourceParts(message.raw_text, selectedSpan?.messageId === message.comment_id ? selectedSpan : null); return <article className={`conversation-message ${selectedMessage === message.comment_id ? 'focused' : ''}`} id={`message-${message.comment_id}`} key={message.comment_id} tabIndex={-1}><span className="message-index">{String(result.message_scores.findIndex(item => item.comment_id === message.comment_id) + 1 || index + 1).padStart(2, '0')}</span><div className="message-body"><div className="message-meta"><strong>{message.speaker_id || 'Unspecified speaker'}</strong><code className="message-id">#{message.comment_id}</code>{message.comment_time && <time>{message.comment_time}</time>}{message.parent_id && result.message_scores.some(item => item.comment_id === message.parent_id) && <button className="reply-link" onClick={() => focusMessage(message.parent_id!)}>Reply to message {message.parent_id}</button>}{evidenceIndex >= 0 && evidenceIndex < 3 && <Badge type="candidate">Candidate {evidenceIndex + 1}</Badge>}</div><p>{highlighted ? <>{highlighted.before}<mark className="source-span">{highlighted.match}</mark>{highlighted.after}</> : message.raw_text}</p><div className="message-category-scores">{unified && message.riskCategoryScores && unified.riskCategories.map(category => <span key={category.id} title={`Exact category score: ${riskExact(message.riskCategoryScores?.[category.id])}`}>{categoryNames[category.id]} {riskDisplay(message.riskCategoryScores?.[category.id])}</span>)}</div><div className="message-topics">{message.topic_ids.length ? message.topic_ids.map(topicId => <button key={topicId} className={`topic-chip color-${((topicOrder.get(topicId) || 1) - 1) % 5}`} onClick={() => setTopicFilter(topicId)}>Topic {topicOrder.get(topicId) || topicId}{result.topics.find(topic => topic.topic_id === topicId)?.context_association_message_ids?.includes(message.comment_id) ? " · Reply association" : ""}</button>) : <span className="no-topic">No topic assigned</span>}</div></div><span className="message-risk" title={`Exact message score: ${riskExact(message.risk_score)}`}>{riskDisplay(message.risk_score)}</span></article>; })}</div>
        {filteredMessages.length > 30 && <button className="text-button show-more" onClick={() => setShowAllMessages(value => !value)}>{showAllMessages ? 'Collapse conversation' : `Show all ${filteredMessages.length} messages`}<ChevronDown size={15} /></button>}
      </section>
      <section className="card review-section" data-report-region="review"><div className="card-heading"><div><h2>Human review</h2><p>Record your contextual judgment separately from model outputs.</p></div><ShieldCheck size={22} className="muted" /></div><div className="review-options" role="radiogroup" aria-label="Review decision">{reviewOptions.map(status => <label key={status} className={reviewStatus === status ? `selected review-${status}` : ''}><input type="radio" name="review-status" checked={reviewStatus === status} onChange={() => setReviewStatus(status)} /><span>{reviewNames[status]}</span></label>)}</div><label className="field-label" htmlFor="review-note">Review notes <span>{reviewNote.length} / 5,000 characters</span></label><textarea id="review-note" aria-label="Review notes" rows={3} value={reviewNote} onChange={e => setReviewNote(e.target.value)} placeholder="Example: This candidate quotes someone else. Check the preceding reply for context." maxLength={5000} /><InlineError message={saveError} /><div className="review-footer"><span>{dirty ? 'Unsaved changes' : analysis.review.updatedAt ? `Last saved ${dateText(analysis.review.updatedAt)}` : 'No review saved yet'}</span><button className="button primary" onClick={() => void saveReview()} disabled={saving || !dirty}>{saving ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}{saving ? 'Saving…' : 'Save review'}</button></div></section>
      <details className="model-details"><summary><FileText size={15} />Analysis details<ChevronDown size={15} /></summary><dl><div><dt>Model version</dt><dd>{analysis.modelVersion}</dd></div><div><dt>{unified ? 'Highest message clue score' : 'Conversation score (full value)'}</dt><dd>{riskExact(unified ? unified.overallEvidenceScore : !isPredictionV2(result) ? result.risk_probability : null)}</dd></div>{!isPredictionV2(result) && <div><dt>Decision threshold (full value)</dt><dd>{riskExact(result.threshold)}</dd></div>}<div><dt>Result schema version</dt><dd>{analysis.schemaVersion}</dd></div><div><dt>Completed at</dt><dd>{dateText(analysis.completedAt)}</dd></div><div><dt>Model identity</dt><dd>{analysis.modelHash || 'Not recorded'}</dd></div><div><dt>Deployment identity</dt><dd>{analysis.deploymentHash || 'Not recorded for this legacy entry'}</dd></div></dl><p>{unified ? 'Risk categories and extractive topics are computed separately. SCCD reference scores are not guaranteed to remain calibrated on campus inputs. Examples demonstrate application features.' : 'Topic strength measures grouping, not accuracy. This local legacy record preserves its original SCCD model output and review notes.'}</p></details>
    </>}
  </>;
}

function HistoryPage({ onSelect, notify, onDeleted }: { onSelect: (id: string) => void; notify: (message: string) => void; onDeleted: (id: string) => void }) {
  const [items, setItems] = useState<AnalysisRecord[]>([]);
  const [query, setQuery] = useState('');
  const [prediction, setPrediction] = useState('');
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [deleting, setDeleting] = useState<string | null>(null);
  useEffect(() => {
    let active = true; let poll: number | undefined;
    if (from && to && from > to) { setError('Start date must not be later than end date.'); setLoading(false); return; }
    const load = async () => {
      try {
        const params = new URLSearchParams(); if (query.trim()) params.set('query', query.trim()); if (prediction) params.set('prediction', prediction); if (from) params.set('from', from); if (to) params.set('to', to);
        const data = await request<{ analyses: AnalysisRecord[] }>(`/analyses?${params}`);
        if (!active) return; setItems(data.analyses); setError(null); setLoading(false);
        if (data.analyses.some(item => item.status === 'running' || item.status === 'queued')) poll = window.setTimeout(load, 1500);
      } catch (e) { if (active) { setError(errorMessage(e)); setLoading(false); } }
    };
    setLoading(true); const debounce = window.setTimeout(load, 250); return () => { active = false; window.clearTimeout(debounce); window.clearTimeout(poll); };
  }, [query, prediction, from, to, refresh]);
  const remove = async (item: AnalysisRecord) => {
    if (!window.confirm(`Delete the analysis and review for ${item.title}? This cannot be undone.`)) return;
    setDeleting(item.id);
    try { await request(`/analyses/${encodeURIComponent(item.id)}`, { method: 'DELETE' }); setRefresh(value => value + 1); notify('Analysis deleted.'); onDeleted(item.id); if (localStorage.getItem('campus-last-analysis') === item.id) localStorage.removeItem('campus-last-analysis'); } catch (e) { setError(errorMessage(e)); } finally { setDeleting(null); }
  };
  const hasFilter = Boolean(query || prediction || from || to);
  return <><PageHeading number="03" title="Analysis and review history" description="Revisit conversations, continue reviewing, or export your analysis." action={<button className="button primary" onClick={() => navigate('new')}><Plus size={17} />New analysis</button>} />
    <section className="card history-card" data-report-region="history"><div className="history-toolbar"><div className="search-field"><Search size={17} /><input aria-label="Search conversations" value={query} onChange={e => setQuery(e.target.value)} placeholder="Search titles or text" />{query && <button className="icon-button" aria-label="Clear search" onClick={() => setQuery('')}><X size={15} /></button>}</div><label><span>Model signal</span><select value={prediction} onChange={e => setPrediction(e.target.value)}><option value="">All results</option><option value="1">Risk clues present</option><option value="0">Below risk threshold</option></select></label><button className="icon-button refresh-button" onClick={() => setRefresh(value => value + 1)} aria-label="Refresh history"><RefreshCw size={18} className={loading ? 'spin' : ''} /></button></div><div className="date-filter"><Clock3 size={15} /><span>Analysis date</span><input type="date" aria-label="Start date" value={from} onChange={e => setFrom(e.target.value)} /><span>to</span><input type="date" aria-label="End date" value={to} min={from} onChange={e => setTo(e.target.value)} />{hasFilter && <button className="text-button" onClick={() => { setQuery(''); setPrediction(''); setFrom(''); setTo(''); }}>Reset filters</button>}<span className="history-count">{loading ? 'Searching…' : `${items.length} records`}</span></div>
      <InlineError message={error} />
      {loading && !items.length ? <Loading>Loading history…</Loading> : !items.length ? <Empty title={hasFilter ? 'No matching records' : 'No analyses yet'} description={hasFilter ? 'Try another search term or date range.' : 'Analyses are saved locally and remain available after restart.'}>{!hasFilter && <button className="button secondary" onClick={() => navigate('new')}>Create your first analysis<ArrowRight size={16} /></button>}</Empty> : <div className="table-scroll"><table className="history-table"><thead><tr><th>Conversation</th><th>Status</th><th>Clue / conversation score</th><th>Human review</th><th>Time</th><th><span className="visually-hidden">Actions</span></th></tr></thead><tbody>{items.map(item => <tr key={item.id}><td><button className="record-title" onClick={() => onSelect(item.id)}>{item.title}<ChevronRight size={15} /></button><div className="record-subtitle">{sourceNames[item.source]}<span>·</span>{item.messageCount} messages{item.topicCount != null && <><span>·</span>{item.topicCount} topics</>}</div></td><td><Badge type={item.status}>{(item.status === 'queued' || item.status === 'running') && <LoaderCircle size={12} className="spin" />}{statusNames[item.status]}</Badge></td><td><span className={`table-score ${item.prediction === 1 ? 'risk' : ''}`} title={`${item.schemaVersion === 2 ? "Highest message clue score" : "Legacy conversation score"}：${riskExact(item.riskScore)}`}>{riskDisplay(item.riskScore)}</span>{item.prediction != null && <small className="prediction-label">{item.schemaVersion === 2 ? 'Message category clues' : 'Legacy conversation score'} · {item.prediction === 1 ? 'Potential risk' : 'Below threshold'}</small>}</td><td><span className={`review-state review-${item.review.status}`}><i />{reviewNames[item.review.status]}</span></td><td className="history-date">{dateText(item.createdAt)}</td><td><div className="table-actions"><button className="text-button" onClick={() => onSelect(item.id)}>View</button><button className="icon-button delete-button" title={item.status === 'queued' || item.status === 'running' ? 'Available after the task finishes' : 'Delete record'} aria-label={`Delete ${item.title}`} disabled={item.status === 'queued' || item.status === 'running' || deleting === item.id} onClick={() => void remove(item)}>{deleting === item.id ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}</button></div></td></tr>)}</tbody></table></div>}
    </section></>;
}

function BenchmarkPage() {
  const [data, setData] = useState<Benchmarks | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { let active = true; request<Benchmarks>('/benchmarks').then(value => { if (active) setData(value); }).catch(e => { if (active) setError(errorMessage(e)); }); return () => { active = false; }; }, []);
  return <><PageHeading number="04" title="Research code" description="Explore the methods and run your own experiments with locally prepared data." />
    <InlineError message={error} />
    {!data && !error && <Loading>Loading research information…</Loading>}
    {data && <section className="card research-info"><div className="card-heading"><div><h2>Experimental results are not distributed</h2><p>{data.note}</p></div><FlaskConical size={24} /></div>
      <p>The public repository includes CST-MIL and adapters for LDA, BTM, GSDMM, NMF and BERTopic. Obtain source datasets under their original terms and save new runs locally.</p>
      <p>The demonstration analyzes your input on this device. Its outputs illustrate application features and are not a benchmark evaluation.</p>
      <a className="button secondary" href="https://github.com/Kirito14IT/CST-MIL#research-code-and-reproducibility" target="_blank" rel="noreferrer">Research documentation<ArrowRight size={16} /></a>
    </section>}
  </>;
}
