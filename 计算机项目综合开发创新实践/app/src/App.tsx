import { useEffect, useRef, useState, type ReactNode } from 'react';
import { Activity, ArrowDownToLine, ArrowRight, BarChart3, Check, CheckCircle2, ChevronDown, ChevronRight, CircleHelp, Clock3, FileJson, FileText, FlaskConical, History, Layers3, LoaderCircle, MessageSquareText, Plus, RefreshCw, Search, ShieldCheck, Trash2, Upload, X, TriangleAlert } from 'lucide-react';
import type { AnalysisDetail, AnalysisRecord, BenchmarkRow, Benchmarks, Example, Health, ImportPreview, Prediction, PredictionV2, Review, ReviewStatus, SessionRecord, Source } from '../shared/types';
import { isPredictionV2 } from '../shared/types';
import { errorMessage, post, request } from './api';
import { sourceParts, TopicCard, type SourceHighlight } from './TopicCard';

type Page = 'new' | 'history' | 'benchmarks' | 'analysis';
const sourceNames: Record<Source, string> = { manual: '手动输入', upload: '文件导入', demo: '演示样例', sccd: 'SCCD 数据' };
const statusNames = { queued: '等待分析', running: '分析中', succeeded: '已完成', failed: '分析失败' };
const reviewNames: Record<ReviewStatus, string> = { pending: '待核查', risk: '存在风险', clear: '未见风险', uncertain: '暂不能确定' };
const reviewOptions: ReviewStatus[] = ['pending', 'risk', 'clear', 'uncertain'];
const dateText = (value: string | null) => value ? new Date(value).toLocaleString('zh-CN', { hour12: false, month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '—';
const score = (value: number | null | undefined, digits = 1) => value == null || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`;
const num = (value: number | null | undefined, digits = 2) => value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('zh-CN', { maximumFractionDigits: digits });
const riskDisplay = (value: number | null | undefined) => value != null && Number.isFinite(value) && value > 0 && value < 0.001 ? '<0.001' : num(value, 3);
const riskExact = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : String(value);
const thresholdDisplay = (value: number | null | undefined) => value == null || !Number.isFinite(value) ? '—' : value.toFixed(3);
const duration = (value: number | null) => value == null ? '—' : value < 1000 ? `${Math.round(value)} 毫秒` : `${(value / 1000).toFixed(2)} 秒`;

function readRoute() {
  const route = window.location.hash.replace(/^#\/?/, '').split('/');
  return { page: (['history', 'benchmarks', 'analysis'].includes(route[0]) ? route[0] : 'new') as Page, id: route[0] === 'analysis' ? route[1] : undefined };
}
function navigate(page: Page, id?: string) { window.location.hash = page === 'analysis' && id ? `analysis/${encodeURIComponent(id)}` : page; }
function InlineError({ message, onClose }: { message: string | null; onClose?: () => void }) {
  if (!message) return null;
  return <div className="notice notice-error" role="alert"><TriangleAlert size={18} /><span>{message}</span>{onClose && <button className="icon-button" aria-label="关闭提示" onClick={onClose}><X size={17} /></button>}</div>;
}
function Loading({ children = '正在读取…' }: { children?: ReactNode }) { return <div className="loading-state" role="status"><LoaderCircle className="spin" size={23} /><span>{children}</span></div>; }
function Empty({ title, description, children }: { title: string; description: string; children?: ReactNode }) {
  return <div className="empty-state"><div className="empty-icon"><MessageSquareText size={28} /></div><h3>{title}</h3><p>{description}</p>{children}</div>;
}
function PageHeading({ number, title, description, action }: { number: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow"><span>{number}</span> 校园文本分析工作台</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}
function Badge({ type, children }: { type?: string; children: ReactNode }) { return <span className={`badge ${type || ''}`}>{children}</span>; }
function ModelStatus({ health, error }: { health: Health | null; error: string | null }) {
  const ready = health?.worker.status === 'ready';
  return <div className="model-status"><span className={`status-dot ${ready ? 'ready' : error || health?.worker.status === 'error' ? 'failed' : ''}`} /><div><strong>{ready ? '本地模型已就绪' : error ? '服务连接中断' : health?.worker.status === 'error' ? '模型暂不可用' : '正在加载本地模型'}</strong><span>{ready ? '数据仅在本机处理' : '可先准备会话内容'}</span></div></div>;
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
    <a className="skip-link" href="#main-content">跳到主要内容</a>
    <aside className="sidebar">
      <a className="brand" href="#new" aria-label="校园文本分析首页"><span className="brand-mark"><MessageSquareText size={25} /></span><div><strong>校园文本分析</strong><span>轻量级风险分析系统</span></div></a>
      <div className="nav-caption">分析工作台</div>
      <nav aria-label="主要导航">
        <button className={route.page === 'new' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('new')}><Plus size={19} /><span>新建分析</span><ChevronRight className="nav-chevron" size={15} /></button>
        <button className={route.page === 'analysis' ? 'nav-item active' : 'nav-item'} onClick={() => lastId ? openAnalysis(lastId) : navigate('analysis')}><Layers3 size={19} /><span>分析结果</span></button>
        <button className={route.page === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('history')}><History size={19} /><span>历史记录</span></button>
        <button className={route.page === 'benchmarks' ? 'nav-item active' : 'nav-item'} onClick={() => navigate('benchmarks')}><BarChart3 size={19} /><span>算法对比</span></button>
      </nav>
      <div className="sidebar-note"><FlaskConical size={18} /><p>从一段对话开始，<br />理解主题，审慎判断。</p><span>课程项目 · CST-MIL</span></div>
      <div className="sidebar-bottom"><ModelStatus health={health} error={healthError} /><span className="local-label">本地运行 · 人工核查辅助工具</span></div>
    </aside>
    <div className="workspace"><header className="topbar"><span>计算机项目综合开发创新实践</span><span className="topbar-right"><ShieldCheck size={15} /> 本机工作空间</span></header>
      <main id="main-content" className="main-content">
        {healthError && <InlineError message={healthError} />}
        {health?.worker.status === 'error' && <details className="worker-warning"><summary><TriangleAlert size={16} /> 模型未能加载，展开查看原因</summary><p>{health.worker.error || '请检查算法环境和模型文件后重新启动。'}</p></details>}
        {route.page === 'new' && <NewAnalysis health={health} onCreated={(id, count) => { if (count > 1) setToast(`已提交 ${count} 个会话，其余任务可在历史记录中查看。`); openAnalysis(id); }} />}
        {route.page === 'analysis' && (route.id ? <Results key={route.id} id={route.id} notify={setToast} /> : <><PageHeading number="02" title="分析结果" description="主题、候选证据和完整对话，共同提供核查依据。" /><div className="card"><Empty title="先分析一段校园会话" description="提交会话后，这里会展示真实模型输出及人工核查记录。"><button className="button primary" onClick={() => navigate('new')}><Plus size={17} />新建分析</button></Empty></div></>)}
        {route.page === 'history' && <HistoryPage onSelect={openAnalysis} notify={setToast} onDeleted={id => { if (lastId === id) setLastId(null); }} />}
        {route.page === 'benchmarks' && <BenchmarkPage />}
      </main><footer className="workspace-footer"><span>校园文本场景的轻量级风险分析系统</span><span>模型分析用于辅助理解，结论由人工核查。</span></footer>
    </div>
    {toast && <div className="toast" role="status"><CheckCircle2 size={19} /><span>{toast}</span><button className="icon-button" onClick={() => setToast('')} aria-label="关闭通知"><X size={16} /></button></div>}
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
    if (file.size > 5 * 1024 * 1024) { setError('文件超过 5 MB，请拆分会话后再导入。'); return; }
    const extension = file.name.split('.').pop()?.toLowerCase();
    if (extension !== 'csv' && extension !== 'json') { setError('请选择 CSV 或 JSON 文件。'); return; }
    try { const text = await file.text(); if (text.includes('\uFFFD')) { setError('文件可能不是 UTF-8 编码。请另存为 UTF-8 CSV 或 JSON 后重试。'); return; } setContent(text); setFilename(file.name); setFormat(extension); } catch { setError('文件读取失败，请重新选择文件。'); }
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
      if (!tasks.analyses.length) throw new Error('尚未创建分析任务，请重试。');
      onCreated(tasks.analyses[0].id, tasks.analyses.length);
    } catch (e) { setError(errorMessage(e)); } finally { setBusy(null); }
  };
  const downloadTemplate = (kind: 'csv' | 'json') => {
    const text = kind === 'csv' ? '\uFEFFsession_id,title,id,speaker,text,parent_id,order,post_text\nexample,课程讨论,m1,同学A,明天的小组汇报几点开始？,,1,课程小组交流\nexample,课程讨论,m2,同学B,下午两点，我们提前十分钟到。,m1,2,课程小组交流\n' : JSON.stringify({ title: '课程讨论', postText: '课程小组交流', includePost: false, messages: [{ id: 'm1', speaker: '同学A', text: '明天的小组汇报几点开始？' }, { id: 'm2', speaker: '同学B', text: '下午两点，我们提前十分钟到。', parentId: 'm1' }] }, null, 2);
    const url = URL.createObjectURL(new Blob([text], { type: kind === 'csv' ? 'text/csv;charset=utf-8' : 'application/json;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = `校园会话导入模板.${kind}`; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const current = preview?.sessions[previewIndex];
  const canPreview = mode === 'examples' ? Boolean(selectedExample) : Boolean(content.trim());
  return <>
    <PageHeading number="01" title="短文本主题与风险分析" description="导入一段校园会话，发现讨论主题，定位需要进一步核查的内容。" />
    <div className="workflow-strip" aria-label="分析流程"><span className={!preview ? 'current' : 'complete'}><b>{preview ? <Check size={13} /> : '1'}</b>准备会话</span><i /><span className={preview ? 'current' : ''}><b>2</b>确认输入</span><i /><span><b>3</b>查看分析与核查</span><div className="workflow-note"><ShieldCheck size={15} /> 本地处理，无需上传云端</div></div>
    <div className="new-layout">
      <section className="card input-card" data-report-region="input">
        <div className="card-heading"><div><h2>会话内容</h2><p>选择一种方式开始</p></div><span className="small-label">输入</span></div>
        <div className="input-tabs" role="tablist" aria-label="会话输入方式">{([['paste', '粘贴文本', MessageSquareText], ['upload', '上传文件', Upload], ['examples', '校园示例', FlaskConical]] as const).map(([key, label, Icon]) => <button key={key} role="tab" aria-selected={mode === key} className={mode === key ? 'selected' : ''} onClick={() => changeMode(key)} disabled={Boolean(busy)}><Icon size={17} />{label}</button>)}</div>
        <div className="input-panel" role="tabpanel">
          {mode === 'examples' && <div className="example-section">{examplesLoading ? <Loading>正在读取校园示例…</Loading> : <><p className="field-help">校园样例为虚构演示，包含正常对照、风险候选和漏检观察，并非每例都应超过阈值。部分推荐案例曾按旧模型表现筛选，仅用于功能演示；SCCD 样例来自已有公开数据。</p><div className="example-grid">{examples.map((example, index) => <button key={example.id} className={`example-card ${selectedExample?.id === example.id ? 'selected' : ''}`} onClick={() => selectExample(example)} disabled={Boolean(busy)}><div className="example-top"><span className="example-number">{String(index + 1).padStart(2, '0')}</span>{selectedExample?.id === example.id && <CheckCircle2 size={17} />}</div><strong>{example.title}</strong><div className="example-source"><Badge type={example.session.source === 'demo' ? 'demo' : ''}>{sourceNames[example.session.source]}</Badge></div><p>{example.description}</p><div className="example-tags">{example.tags.map(tag => <span key={tag}>{tag}</span>)}</div></button>)}</div></>}</div>}
          {mode === 'upload' ? <>
            <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); void loadFile(e.dataTransfer.files[0]); }}><span className="upload-icon"><Upload size={27} /></span><h3>{filename || '把会话文件拖到这里'}</h3><p>{filename ? '文件已读取，点击下方按钮校验并预览。' : '支持 UTF-8 编码的 CSV、JSON，单文件不超过 5 MB'}</p><button className="button secondary" onClick={() => fileRef.current?.click()} disabled={Boolean(busy)}>{filename ? '重新选择文件' : '选择文件'}</button><input ref={fileRef} type="file" accept=".csv,.json,application/json,text/csv" className="visually-hidden" aria-label="上传会话文件" onChange={e => void loadFile(e.target.files?.[0])} /></div>
            <div className="template-row"><span>还没有准备好的文件？</span><button className="text-button" onClick={() => downloadTemplate('csv')}><ArrowDownToLine size={14} /> CSV 模板</button><button className="text-button" onClick={() => downloadTemplate('json')}><ArrowDownToLine size={14} /> JSON 模板</button></div>
            <details className="format-details"><summary>查看文件字段说明 <ChevronDown size={15} /></summary><p>CSV 至少包含 text 列。可选 id、speaker、parent_id、time、order、session_id、title、post_text；多个会话用 session_id 区分。JSON 使用 title、messages（包含 id 和 text），主帖使用 postText，是否参与分析使用 includePost。</p><p>单次最多 100 个会话，每会话最多 2,000 条消息。导入预览会检查重复编号、空内容及不存在的回复对象。</p></details>
          </> : (mode === 'paste' || selectedExample) && <>
            <label className="field-label" htmlFor="session-title">会话名称 <span>便于之后查找</span></label><input id="session-title" value={title} onChange={e => { setTitle(e.target.value); invalidate(); }} placeholder="例如：社团活动报名讨论" maxLength={200} disabled={Boolean(busy)} />
            {mode === 'paste' ? <><div className="field-row"><label className="field-label" htmlFor="conversation">评论内容</label><span>{content.split('\n').filter(line => line.trim()).length} 条非空行</span></div><textarea id="conversation" className="conversation-input" value={content} onChange={e => { setContent(e.target.value); invalidate(); }} placeholder={'每行一条评论，可写成“同学A：评论内容”。\n\n同学A：大家的小组报告准备得怎么样了？\n同学B：图表已经整理好了，今晚一起核对一下吧。'} disabled={Boolean(busy)} /><p className="field-help">保留原始表达和发言顺序。空行会被忽略，回复关系可通过文件导入保留。</p></> : <div className="example-selected"><CheckCircle2 size={16} /><span>已选“{selectedExample?.title}”，包含 {selectedExample?.session.messages.length} 条消息。校验后可查看完整内容。</span></div>}
            <details className="post-details" open={postText ? true : undefined}><summary>主帖与背景 <span>选填</span><ChevronDown size={15} /></summary><label className="visually-hidden" htmlFor="post-text">主帖内容</label><textarea id="post-text" rows={3} value={postText} onChange={e => { setPostText(e.target.value); invalidate(); }} placeholder="记录这段讨论的主帖或背景信息。" disabled={Boolean(busy)} /><label className="checkbox-label"><input type="checkbox" checked={includePost} onChange={e => { setIncludePost(e.target.checked); invalidate(); }} disabled={!postText.trim() || Boolean(busy)} /><span>将主帖作为第一条消息参与分析</span></label><p className="field-help">默认仅展示主帖背景，模型分析评论列表。</p></details>
          </>}
        </div>
        <div className="input-actions"><span>先检查输入，再开始分析</span><button className="button primary" onClick={() => void makePreview()} disabled={!canPreview || Boolean(busy)}>{busy === 'preview' ? <LoaderCircle className="spin" size={17} /> : <CheckCircle2 size={17} />}{busy === 'preview' ? '正在校验…' : '校验并预览'}</button></div>
      </section>
      <aside className="guidance-column"><section className="guidance-card"><span className="section-kicker">从内容到线索</span><h2>一段会话，<br />三个观察角度。</h2><div className="guidance-item"><Layers3 size={20} /><div><h3>大家在讨论什么</h3><p>从短文本中提取主题关键词，并找到关联消息。</p></div></div><div className="guidance-item"><Activity size={20} /><div><h3>哪些内容值得关注</h3><p>查看风险类别与候选证据，回到上下文理解。</p></div></div><div className="guidance-item"><ShieldCheck size={20} /><div><h3>留下人工核查依据</h3><p>记录核查结论和备注，保存或导出分析结果。</p></div></div></section><div className="care-note"><CircleHelp size={17} /><p>分析输出需结合语境核查。玩笑、引用和分歧不能只凭分数定性；风险类别也不等同于讨论话题。</p></div><div className="method-stamp"><span>CST-MIL</span><p>轻量级隐主题发现<br />与会话风险分析</p></div></aside>
    </div>
    <InlineError message={error} onClose={() => setError(null)} />
    {preview && current && <section className="card preview-card" data-report-region="input-preview"><div className="card-heading"><div><div className="section-kicker">确认输入</div><h2>会话预览 <span className="heading-count">{preview.sessions.length} 个会话 · {preview.sessions.reduce((total, session) => total + session.messages.length, 0)} 条评论</span></h2></div><Badge type="good"><Check size={13} />校验通过</Badge></div>
      {preview.warnings.map((warning, i) => <div className="notice notice-warn" key={i}><TriangleAlert size={16} /><span>{warning}</span></div>)}
      {preview.sessions.length > 1 && <label className="preview-picker">查看会话<select value={previewIndex} onChange={e => { setPreviewIndex(Number(e.target.value)); setShowAll(false); }}>{preview.sessions.map((session, index) => <option key={index} value={index}>{index + 1}. {session.title}</option>)}</select></label>}
      <div className="preview-title"><h3>{current.title}</h3><Badge>{sourceNames[current.source]}</Badge></div>
      {current.postText && <div className="post-context"><strong>主帖背景 <small>{current.includePost ? '将参与分析' : '仅展示背景'}</small></strong><p>{current.postText}</p></div>}
      <div className="preview-messages">{current.messages.slice(0, showAll ? undefined : 4).map((message, index) => <div className="preview-message" key={message.id}><span className="message-index">{String(index + 1).padStart(2, '0')}</span><div><div className="message-meta"><strong>{message.speaker || '未标注发言人'}</strong>{message.parentId && <span>回复 #{current.messages.findIndex(item => item.id === message.parentId) + 1}</span>}</div><p>{message.text}</p></div></div>)}</div>
      {current.messages.length > 4 && <button className="text-button show-more" onClick={() => setShowAll(!showAll)}>{showAll ? '收起至前 4 条' : `当前预览前 4 条 · 展开全部 ${current.messages.length} 条消息`}<ChevronDown size={15} /></button>}
      <div className="preview-footer"><p>{health?.worker.status === 'ready' ? '内容确认后，将调用本机模型进行分析。' : health?.worker.status === 'error' ? '模型暂不可用。重新提交时，系统将尝试恢复模型进程。' : '模型尚未就绪。可保留预览，待模型加载完成后开始。'}</p><button className="button primary" disabled={Boolean(busy) || !health || health.worker.status === 'starting'} onClick={() => void submit()}>{busy === 'submit' ? <LoaderCircle className="spin" size={17} /> : <Activity size={17} />}{busy === 'submit' ? '正在提交…' : health?.worker.status === 'error' ? '重试并分析' : `开始分析${preview.sessions.length > 1 ? ` ${preview.sessions.length} 个会话` : ''}`}<ArrowRight size={17} /></button></div>
    </section>}
  </>;
}

function RiskSummary({ result }: { result: Prediction }) {
  if (isPredictionV2(result)) return <section className={`card risk-summary ${result.prediction === 1 ? 'elevated' : ''}`}><div className="stat-label"><Activity size={17} />最高消息风险线索强度</div><div className="risk-value" title={`最高消息风险线索强度完整数值：${riskExact(result.overallEvidenceScore)}`}>{riskDisplay(result.overallEvidenceScore)}<span>{result.prediction === 1 ? '有类别达到核查阈值' : '各类别未达核查阈值'}</span></div><div className="risk-track"><div style={{ width: `${result.overallEvidenceScore * 100}%` }} /></div><div className="risk-scale"><span>0</span><span>所有消息、所有类别中的最高分</span><span>1</span></div></section>;
  return <section className={`card risk-summary ${result.prediction === 1 ? 'elevated' : ''}`}><div className="stat-label"><Activity size={17} />会话风险分数</div><div className="risk-value" title={`会话风险分数完整数值：${riskExact(result.risk_probability)}`}>{riskDisplay(result.risk_probability)}<span>{result.prediction === 1 ? '疑似网络欺凌风险' : '未达到风险阈值'}</span></div><div className="risk-track"><div style={{ width: `${Math.max(0, Math.min(100, result.risk_probability * 100))}%` }} /><span style={{ left: `${Math.max(0, Math.min(100, result.threshold * 100))}%` }} /></div><div className="risk-scale"><span>低</span><span title={`判定阈值完整数值：${riskExact(result.threshold)}`}>判定阈值 {thresholdDisplay(result.threshold)}</span><span>高</span></div></section>;
}

function CategoryPanel({ result, focusMessage }: { result: PredictionV2; focusMessage: (id: string) => void }) {
  return <>
    <section className="card categories-section" data-report-region="risk-categories"><div className="card-heading"><div><h2>固定风险类别</h2><p>各类别取最高消息分，分别使用验证集确定的阈值</p></div><ShieldCheck size={21} className="muted" /></div><div className="category-grid">{result.riskCategories.map(category => <article className={`category-card ${category.detected ? 'detected' : ''}`} key={category.id}><div className="category-heading"><h3>{category.name}</h3><Badge type={category.detected ? 'candidate' : ''}>{category.detected ? '达到核查阈值' : '未达阈值'}</Badge></div><div className="category-value"><strong title={`类别最高分：${riskExact(category.score)}`}>{riskDisplay(category.score)}</strong><span title={`验证阈值完整数值：${riskExact(category.threshold)}`}>阈值 {thresholdDisplay(category.threshold)}</span></div>{category.message_ids.length > 0 ? <button className="text-button" onClick={() => focusMessage(category.evidence[0]?.comment_id ?? category.message_ids[0])}>查看原文线索<ArrowRight size={14} /></button> : <span className="muted">未返回超过阈值的消息</span>}</article>)}</div></section>
    <details className="card conversation-reference"><summary><span><MessageSquareText size={17} /><strong>SCCD会话迁移参考</strong><small>会话分类器独立输出</small></span><ChevronDown size={16} /></summary><div className="reference-values"><strong title={`SCCD会话参考完整数值：${riskExact(result.conversationRisk.score)}`}>{riskDisplay(result.conversationRisk.score)}</strong><span title={`会话参考阈值完整数值：${riskExact(result.conversationRisk.threshold)}`}>参考阈值 {thresholdDisplay(result.conversationRisk.threshold)}</span><Badge>{result.conversationRisk.prediction === 1 ? '达到会话参考阈值' : '未达会话参考阈值'}</Badge></div><p>{result.conversationRisk.note}</p></details>
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
      setDetail(current => current ? { ...current, analysis: { ...current.analysis, review: data.review } } : current); notify('人工核查记录已保存。');
    } catch (e) { setSaveError(errorMessage(e)); } finally { setSaving(false); }
  };
  if (!detail) return <><PageHeading number="02" title="分析结果" description="查看会话主题与风险候选，结合完整上下文进行核查。" />{error ? <><InlineError message={error} /><button className="button secondary" onClick={() => setReload(value => value + 1)}><RefreshCw size={16} />重新读取</button></> : <Loading>正在读取分析任务…</Loading>}</>;
  const { analysis, session, result } = detail;
  const unified = result && isPredictionV2(result) ? result : null;
  const pending = analysis.status === 'queued' || analysis.status === 'running';
  const dirty = reviewStatus !== analysis.review.status || reviewNote !== analysis.review.note;
  const filteredMessages = result?.message_scores.filter(message => topicFilter == null || message.topic_ids.includes(topicFilter)) || [];
  const messages = filteredMessages.slice(0, showAllMessages ? undefined : 30);
  const topicOrder = new Map(result?.topics.map((topic, index) => [topic.topic_id, index + 1]));
  return <>
    <PageHeading number="02" title="分析结果" description="核对讨论主题、风险类别与对应原文，再记录人工判断。" action={<button className="button secondary" onClick={() => navigate('new')}><Plus size={17} />新建分析</button>} />
    <InlineError message={error} />
    <section className="card result-title-card" data-report-region="result-identity"><div><div className="result-title-line"><Badge type={session.source === 'demo' ? 'demo' : ''}>{sourceNames[session.source]}</Badge><span className="muted">{dateText(analysis.createdAt)}</span><Badge type={analysis.status}>{statusNames[analysis.status]}</Badge></div><h2>{session.title}</h2><p>{analysis.messageCount} 条分析消息 <span>·</span> {session.includePost ? '包含主帖' : '主帖仅作背景'} {analysis.durationMs != null && <><span>·</span>耗时 {duration(analysis.durationMs)}</>}</p></div>{analysis.status === 'succeeded' && <div className="export-buttons"><a className="button secondary small" href={`/api/analyses/${encodeURIComponent(id)}/export?format=csv`} download><ArrowDownToLine size={15} />导出 CSV</a><a className="button secondary small" href={`/api/analyses/${encodeURIComponent(id)}/export?format=json`} download><FileJson size={15} />JSON</a></div>}</section>
    {session.source === 'demo' && <div className="notice notice-info compact"><FlaskConical size={16} /><span>本会话为课程演示样例。以下为模型实际输出，不代表校园场景的整体检测效果。</span></div>}
    {pending && <section className="card job-progress"><div className="progress-orbit"><LoaderCircle size={39} className="spin" /></div><h2>{analysis.status === 'queued' ? '会话已加入分析队列' : '正在理解这段会话'}</h2><p>{analysis.status === 'queued' ? '任务按提交顺序处理，开始后会自动更新。' : '本机模型正在提取话题与风险线索，请稍候。'}</p><span className="badge">{analysis.status === 'queued' ? '等待中' : '真实模型推理中'}</span><button className="text-button" onClick={() => navigate('history')}>去历史记录查看其他任务<ArrowRight size={15} /></button>{error && <button className="button secondary" onClick={() => setReload(value => value + 1)}><RefreshCw size={16} />恢复状态查询</button>}</section>}
    {analysis.status === 'failed' && <section className="card"><Empty title="本次分析未能完成" description={analysis.error || '模型运行过程中发生异常。'}><button className="button primary" onClick={() => navigate('new')}>返回新建分析<ArrowRight size={16} /></button></Empty></section>}
    {result && analysis.status === 'succeeded' && <>
      <div className="summary-grid"><RiskSummary result={result} /><section className="card stat-card"><div className="stat-label"><Layers3 size={17} />发现主题</div><strong>{result.topics.length}<small> 个</small></strong><p>{result.topics.length ? (unified ? '基本语义分组与直接回复关联' : '关键词与消息关联') : '暂未形成明确主题'}</p></section><section className="card stat-card"><div className="stat-label"><MessageSquareText size={17} />主题覆盖</div><strong>{result.coverage_details.assigned_messages}<small> / {result.coverage_details.total_messages}</small></strong><p>有主题归属的分析消息</p></section></div>
      <div className="notice interpretation-note"><CircleHelp size={17} /><span>{unified ? "风险类别和动态主题分别计算。分数表示核查线索强度，候选证据需结合上下文判断。SCCD会话参考可展开查看。" : "风险分数范围为0–1，表示旧模型判断倾向，不是准确率或事件发生概率。会话分数综合整段对话计算，与单条消息分数不同；候选证据仍需结合上下文核查。"}悬停分数可查看完整数值。</span></div>
      {unified && <CategoryPanel result={unified} focusMessage={focusMessage} />}
      <div className="results-upper" data-report-region="topics-evidence"><section className="card topics-section"><div className="card-heading"><div><h2>{unified ? "动态讨论主题" : "隐主题发现"}</h2><p>点击关键词可核对原文依据，主题不等同于风险类别</p></div><Layers3 size={21} className="muted" /></div>{result.topics.length === 0 ? <div className="zero-topics"><Layers3 size={26} /><h3>{unified ? "未形成清晰主题" : "暂未形成明确主题"}</h3><p>{unified ? "本次保留未分组消息，不强制命名。风险类别、候选线索和完整原文仍可查看。" : "这段会话未满足主题提取条件。仍可查看会话风险与候选消息，并结合原文进行判断。"}</p></div> : <div className="topic-list">{result.topics.map((topic, index) => <TopicCard topic={topic} index={index} key={topic.topic_id} onMessage={focusMessage} onSupport={support => focusMessage(support.message_id, { messageId: support.message_id, start: support.start, end: support.end, excerpt: support.excerpt })} onFilter={() => { setTopicFilter(topic.topic_id); setShowAllMessages(false); document.getElementById('conversation-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} />)}</div>}</section><section className="card evidence-section"><div className="card-heading"><div><h2>Top-3 候选证据</h2><p>点击回到原文，检查前后语境</p></div><Search size={20} className="muted" /></div>{result.evidence.length ? <div className="evidence-list">{result.evidence.slice(0, 3).map((item, index) => <button className="evidence-card" key={`${item.comment_id}-${index}`} onClick={() => focusMessage(item.comment_id)}><div className="evidence-meta"><span className="evidence-rank">{String(index + 1).padStart(2, '0')}</span><strong>{item.speaker_id || '未标注发言人'}</strong><ArrowRight size={15} /></div><p>{item.raw_text}</p><span className="evidence-score" title={`消息风险分数完整数值：${riskExact(item.risk_score)}`}>{unified ? "消息线索强度" : "消息风险分数"} {riskDisplay(item.risk_score)}</span></button>)}</div> : <div className="zero-topics"><p>本次分析没有返回候选证据。</p></div>}<p className="evidence-footnote">候选按算法排序，仅供核查定位；较高排名不代表已确认存在风险。</p></section></div>
      <section className="card conversation-section" id="conversation-panel" data-report-region="conversation"><div className="card-heading"><div><h2>完整会话 <span className="heading-count">{result.message_scores.length} 条消息</span></h2><p>保留消息编号、原文顺序与真实回复关系</p></div><label className="compact-select"><span className="visually-hidden">按主题筛选消息</span><select value={topicFilter ?? ''} onChange={e => { setTopicFilter(e.target.value === '' ? null : Number(e.target.value)); setShowAllMessages(false); }}><option value="">全部消息</option>{result.topics.map((topic, index) => <option value={topic.topic_id} key={topic.topic_id}>主题 {index + 1} · {topic.keywords.slice(0, 2).join(' / ')}</option>)}</select></label></div>
        {session.postText && <div className="post-context"><strong>主帖背景 <small>{session.includePost ? '已计入分析' : '未参与分析'}</small></strong><p>{session.postText}</p></div>}
        {topicFilter != null && <div className="filter-indicator">正在查看主题 {topicOrder.get(topicFilter)} 的 {filteredMessages.length} 条关联消息<button className="text-button" onClick={() => setTopicFilter(null)}>查看全部<X size={13} /></button></div>}
        {selectedMessage && <div className="source-location" role="status"><Search size={14} /><span>原文定位</span><code>消息 {selectedMessage}</code>{selectedSpan && <span title="从0开始、右端不包含的Unicode字符区间">字符区间 [{selectedSpan.start}, {selectedSpan.end})</span>}<button className="text-button" onClick={() => { setSelectedMessage(null); setSelectedSpan(null); }}>清除定位<X size={13} /></button></div>}
        <div className="message-list">{messages.map((message, index) => { const evidenceIndex = result.evidence.findIndex(item => item.comment_id === message.comment_id); const highlighted = sourceParts(message.raw_text, selectedSpan?.messageId === message.comment_id ? selectedSpan : null); return <article className={`conversation-message ${selectedMessage === message.comment_id ? 'focused' : ''}`} id={`message-${message.comment_id}`} key={message.comment_id} tabIndex={-1}><span className="message-index">{String(result.message_scores.findIndex(item => item.comment_id === message.comment_id) + 1 || index + 1).padStart(2, '0')}</span><div className="message-body"><div className="message-meta"><strong>{message.speaker_id || '未标注发言人'}</strong><code className="message-id">#{message.comment_id}</code>{message.comment_time && <time>{message.comment_time}</time>}{message.parent_id && result.message_scores.some(item => item.comment_id === message.parent_id) && <button className="reply-link" onClick={() => focusMessage(message.parent_id!)}>回复消息 {message.parent_id}</button>}{evidenceIndex >= 0 && evidenceIndex < 3 && <Badge type="candidate">候选 {evidenceIndex + 1}</Badge>}</div><p>{highlighted ? <>{highlighted.before}<mark className="source-span">{highlighted.match}</mark>{highlighted.after}</> : message.raw_text}</p><div className="message-category-scores">{unified && message.riskCategoryScores && unified.riskCategories.map(category => <span key={category.id} title={`完整类别分数：${riskExact(message.riskCategoryScores?.[category.id])}`}>{category.name} {riskDisplay(message.riskCategoryScores?.[category.id])}</span>)}</div><div className="message-topics">{message.topic_ids.length ? message.topic_ids.map(topicId => <button key={topicId} className={`topic-chip color-${((topicOrder.get(topicId) || 1) - 1) % 5}`} onClick={() => setTopicFilter(topicId)}>主题 {topicOrder.get(topicId) || topicId}{result.topics.find(topic => topic.topic_id === topicId)?.context_association_message_ids?.includes(message.comment_id) ? " · 回复关联" : ""}</button>) : <span className="no-topic">暂无主题归属</span>}</div></div><span className="message-risk" title={`消息风险分数完整数值：${riskExact(message.risk_score)}`}>{riskDisplay(message.risk_score)}</span></article>; })}</div>
        {filteredMessages.length > 30 && <button className="text-button show-more" onClick={() => setShowAllMessages(value => !value)}>{showAllMessages ? '收起长会话' : `展开全部 ${filteredMessages.length} 条消息`}<ChevronDown size={15} /></button>}
      </section>
      <section className="card review-section" data-report-region="review"><div className="card-heading"><div><h2>人工核查</h2><p>结合上下文记录你的判断，与模型输出分开保存</p></div><ShieldCheck size={22} className="muted" /></div><div className="review-options" role="radiogroup" aria-label="人工核查结论">{reviewOptions.map(status => <label key={status} className={reviewStatus === status ? `selected review-${status}` : ''}><input type="radio" name="review-status" checked={reviewStatus === status} onChange={() => setReviewStatus(status)} /><span>{reviewNames[status]}</span></label>)}</div><label className="field-label" htmlFor="review-note">核查备注 <span>{reviewNote.length} / 5,000 字符</span></label><textarea id="review-note" aria-label="核查备注" rows={3} value={reviewNote} onChange={e => setReviewNote(e.target.value)} placeholder="例如：候选消息是在引用其他人的话，需要结合上一条回复确认。" maxLength={5000} /><InlineError message={saveError} /><div className="review-footer"><span>{dirty ? '有尚未保存的修改' : analysis.review.updatedAt ? `上次保存于 ${dateText(analysis.review.updatedAt)}` : '尚未保存人工核查记录'}</span><button className="button primary" onClick={() => void saveReview()} disabled={saving || !dirty}>{saving ? <LoaderCircle size={16} className="spin" /> : <Check size={16} />}{saving ? '正在保存…' : '保存核查记录'}</button></div></section>
      <details className="model-details"><summary><FileText size={15} />本次分析信息<ChevronDown size={15} /></summary><dl><div><dt>算法版本</dt><dd>{analysis.modelVersion}</dd></div><div><dt>{unified ? '最高消息风险线索强度' : '会话风险分数（完整值）'}</dt><dd>{riskExact(unified ? unified.overallEvidenceScore : !isPredictionV2(result) ? result.risk_probability : null)}</dd></div>{!isPredictionV2(result) && <div><dt>判定阈值（完整值）</dt><dd>{riskExact(result.threshold)}</dd></div>}<div><dt>结果接口版本</dt><dd>{analysis.schemaVersion}</dd></div><div><dt>完成时间</dt><dd>{dateText(analysis.completedAt)}</dd></div><div><dt>模型标识</dt><dd>{analysis.modelHash || '未记录'}</dd></div><div><dt>部署清单标识</dt><dd>{analysis.deploymentHash || '旧记录未保存清单身份'}</dd></div></dl><p>{unified ? '风险类别与抽取式话题分开计算。SCCD会话参考迁移到校园输入时不保证校准或原域表现；校园样例用于演示应用流程。' : '主题强度用于表示主题聚合程度，不是准确率。该历史结果使用旧SCCD模型，保留原输出与人工备注。'}</p></details>
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
    if (from && to && from > to) { setError('开始日期不能晚于结束日期。'); setLoading(false); return; }
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
    if (!window.confirm(`确定删除“${item.title}”的分析和核查记录吗？此操作无法恢复。`)) return;
    setDeleting(item.id);
    try { await request(`/analyses/${encodeURIComponent(item.id)}`, { method: 'DELETE' }); setRefresh(value => value + 1); notify('分析记录已删除。'); onDeleted(item.id); if (localStorage.getItem('campus-last-analysis') === item.id) localStorage.removeItem('campus-last-analysis'); } catch (e) { setError(errorMessage(e)); } finally { setDeleting(null); }
  };
  const hasFilter = Boolean(query || prediction || from || to);
  return <><PageHeading number="03" title="分析与核查历史" description="回看会话、继续核查，或导出已经整理好的分析结果。" action={<button className="button primary" onClick={() => navigate('new')}><Plus size={17} />新建分析</button>} />
    <section className="card history-card" data-report-region="history"><div className="history-toolbar"><div className="search-field"><Search size={17} /><input aria-label="搜索会话关键词" value={query} onChange={e => setQuery(e.target.value)} placeholder="搜索会话名称或内容" />{query && <button className="icon-button" aria-label="清除搜索" onClick={() => setQuery('')}><X size={15} /></button>}</div><label><span>模型判断</span><select value={prediction} onChange={e => setPrediction(e.target.value)}><option value="">全部结果</option><option value="1">存在风险线索</option><option value="0">未达到风险阈值</option></select></label><button className="icon-button refresh-button" onClick={() => setRefresh(value => value + 1)} aria-label="刷新历史记录"><RefreshCw size={18} className={loading ? 'spin' : ''} /></button></div><div className="date-filter"><Clock3 size={15} /><span>分析时间</span><input type="date" aria-label="开始日期" value={from} onChange={e => setFrom(e.target.value)} /><span>至</span><input type="date" aria-label="结束日期" value={to} min={from} onChange={e => setTo(e.target.value)} />{hasFilter && <button className="text-button" onClick={() => { setQuery(''); setPrediction(''); setFrom(''); setTo(''); }}>重置筛选</button>}<span className="history-count">{loading ? '正在查询…' : `共 ${items.length} 条记录`}</span></div>
      <InlineError message={error} />
      {loading && !items.length ? <Loading>正在读取历史记录…</Loading> : !items.length ? <Empty title={hasFilter ? '没有找到匹配的记录' : '还没有分析记录'} description={hasFilter ? '试着调整关键词或日期范围。' : '分析结果会保存在本机，关闭并重新启动后仍可查看。'}>{!hasFilter && <button className="button secondary" onClick={() => navigate('new')}>开始第一次分析<ArrowRight size={16} /></button>}</Empty> : <div className="table-scroll"><table className="history-table"><thead><tr><th>会话</th><th>分析状态</th><th>线索 / 会话分数</th><th>人工核查</th><th>时间</th><th><span className="visually-hidden">操作</span></th></tr></thead><tbody>{items.map(item => <tr key={item.id}><td><button className="record-title" onClick={() => onSelect(item.id)}>{item.title}<ChevronRight size={15} /></button><div className="record-subtitle">{sourceNames[item.source]}<span>·</span>{item.messageCount} 条消息{item.topicCount != null && <><span>·</span>{item.topicCount} 个主题</>}</div></td><td><Badge type={item.status}>{(item.status === 'queued' || item.status === 'running') && <LoaderCircle size={12} className="spin" />}{statusNames[item.status]}</Badge></td><td><span className={`table-score ${item.prediction === 1 ? 'risk' : ''}`} title={`${item.schemaVersion === 2 ? "最高消息风险线索强度" : "旧会话风险分数"}：${riskExact(item.riskScore)}`}>{riskDisplay(item.riskScore)}</span>{item.prediction != null && <small className="prediction-label">{item.schemaVersion === 2 ? '消息类别线索' : '旧会话分数'} · {item.prediction === 1 ? '疑似风险' : '未达阈值'}</small>}</td><td><span className={`review-state review-${item.review.status}`}><i />{reviewNames[item.review.status]}</span></td><td className="history-date">{dateText(item.createdAt)}</td><td><div className="table-actions"><button className="text-button" onClick={() => onSelect(item.id)}>查看</button><button className="icon-button delete-button" title={item.status === 'queued' || item.status === 'running' ? '任务完成后可删除' : '删除记录'} aria-label={`删除 ${item.title}`} disabled={item.status === 'queued' || item.status === 'running' || deleting === item.id} onClick={() => void remove(item)}>{deleting === item.id ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}</button></div></td></tr>)}</tbody></table></div>}
    </section></>;
}

function MetricChart({ title, description, rows, field, percent = true, unit = '', lower = false }: { title: string; description: string; rows: BenchmarkRow[]; field: keyof BenchmarkRow; percent?: boolean; unit?: string; lower?: boolean }) {
  const values = rows.map(row => typeof row[field] === 'number' ? row[field] as number : 0);
  const max = percent ? 1 : Math.max(...values, 0.001);
  return <section className="card chart-card"><div className="card-heading"><div><h2>{title}</h2><p>{description}</p></div><span className="chart-direction">{lower ? '越低越好 ↓' : '越高越好 ↑'}</span></div><div className="bar-chart">{rows.map((row, index) => <div className={`bar-row ${row.method.toLowerCase().replace(/[-_]/g, '').includes('cstmil') ? 'ours' : ''}`} key={row.method}><div className="bar-label">{row.label}</div><div className="bar-track"><div style={{ width: `${Math.max(0, Math.min(100, values[index] / max * 100))}%` }} /></div><span className="bar-value">{percent ? score(values[index]) : num(values[index], 3)}{unit}</span></div>)}</div></section>;
}

function BenchmarkPage() {
  const [data, setData] = useState<Benchmarks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refresh, setRefresh] = useState(0);
  const [tab, setTab] = useState<'performance' | 'evidence' | 'cost'>('performance');
  useEffect(() => { let active = true; request<Benchmarks>('/benchmarks').then(value => { if (active) { setData(value); setError(null); } }).catch(e => { if (active) setError(errorMessage(e)); }); return () => { active = false; }; }, [refresh]);
  return <><PageHeading number="04" title="在已有实验中，理解方法表现" description="从识别效果、证据定位和计算成本三个角度，查看六种方法的实测结果。" /><div className="notice notice-info"><FlaskConical size={18} /><span><strong>历史SCCD冻结实验</strong> · 本页对应旧模型的六方法比较，不代表当前五来源全量语义模型的性能，也不代表校园示例准确率。</span></div>
    {error && <><InlineError message={error} /><button className="button secondary" onClick={() => setRefresh(value => value + 1)}><RefreshCw size={16} />重试</button></>}
    {!data && !error && <Loading>正在读取实验结果…</Loading>}
    {data && <><section className="benchmark-context"><div><span>实验数据</span><strong>{data.dataset}</strong></div><div><span>数据划分</span><strong>{data.split}</strong></div><div><span>对比方法</span><strong>{data.methods.length} 种</strong></div><div className="benchmark-note"><p>{data.note}</p></div></section>
      <div className="benchmark-tabs" role="tablist" aria-label="实验指标类别">{([['performance', '会话识别效果'], ['evidence', '候选证据定位'], ['cost', '计算与资源成本']] as const).map(([key, label]) => <button key={key} role="tab" aria-selected={tab === key} onClick={() => setTab(key)} className={tab === key ? 'active' : ''}>{label}</button>)}</div>
      {tab === 'performance' && <><div className="charts-grid"><MetricChart title="宏平均 F1" description="同时关注两类会话，平衡精确率与召回率" rows={data.methods} field="macro_f1" /><MetricChart title="风险会话召回率" description="实际网络欺凌会话中，被模型识别出的比例" rows={data.methods} field="recall_cb" /></div><BenchmarkTable rows={data.methods} columns={[['macro_f1', '宏平均 F1', true], ['recall_cb', '风险召回率', true], ['precision_cb', '风险精确率', true], ['ap', 'AP', true], ['roc_auc', 'ROC AUC', true]]} /><div className="metric-explanation"><CircleHelp size={18} /><p><strong>怎样读这些指标：</strong>召回率关注漏掉了多少风险会话，精确率关注风险判断有多可靠。AP 和 ROC AUC 用于观察不同阈值下的整体区分能力，不能替代具体场景中的人工核查。</p></div></>}
      {tab === 'evidence' && <><div className="charts-grid"><MetricChart title="Top-3 命中率" description="前三条候选中，至少包含一条标注证据的比例" rows={data.methods} field="hit_rate_at_3" /><MetricChart title="Top-3 平均召回率" description="前三条候选对已标注证据的平均覆盖程度" rows={data.methods} field="mean_recall_at_3" /></div><BenchmarkTable rows={data.methods} columns={[['hit_rate_at_3', 'Top-3 命中率', true], ['mean_recall_at_3', 'Top-3 平均召回率', true], ['comment_ap', '评论级 AP', true]]} /><div className="metric-explanation"><CircleHelp size={18} /><p>证据定位在测试集中 101 个含正例评论的会话上评估，不是全部 136 个测试会话。候选排名靠前，只表示值得优先阅读；核查仍需回到完整会话。</p></div></>}
      {tab === 'cost' && <><div className="charts-grid"><MetricChart title="单会话平均推理时间" description="同一记录口径下的实测耗时（秒）" rows={data.methods} field="latency_mean_seconds" percent={false} lower /><MetricChart title="峰值内存" description="实验记录的峰值驻留内存（MiB）" rows={data.methods} field="peak_rss_mib" percent={false} lower /></div><BenchmarkTable rows={data.methods} columns={[['train_seconds', '训练时间 / 秒', false], ['inference_test_seconds', '测试推理 / 秒', false], ['latency_mean_seconds', '平均延迟 / 秒', false], ['peak_rss_mib', '峰值内存 / MiB', false], ['model_mib', '模型体积 / MiB', false]]} /><div className="metric-explanation"><CircleHelp size={18} /><p>耗时、内存和模型体积受运行环境及统计口径影响。这里展示已有实验记录，不能直接等同于当前电脑的响应速度。</p></div></>}
      <details className="model-details"><summary><FileText size={15} />指标口径与实验来源<ChevronDown size={15} /></summary><p>{data.note}</p><p>本页读取保存的实验结果，不进行模型训练或外部模型调用。CB 表示数据集中的网络欺凌类别。百分比仅用于展示比例指标。</p><ul>{data.sources.map(source => <li key={source}>{source}</li>)}</ul></details>
    </>}
  </>;
}

function BenchmarkTable({ rows, columns }: { rows: BenchmarkRow[]; columns: [keyof BenchmarkRow, string, boolean][] }) {
  return <div className="card table-scroll benchmark-table-wrap"><table className="benchmark-table"><thead><tr><th>方法</th>{columns.map(([key, label]) => <th key={key}>{label}</th>)}</tr></thead><tbody>{rows.map(row => <tr key={row.method} className={row.method.toLowerCase().replace(/[-_]/g, '').includes('cstmil') ? 'ours-row' : ''}><th>{row.label}{row.method.toLowerCase().replace(/[-_]/g, '').includes('cstmil') && <span className="our-label">历史CST</span>}</th>{columns.map(([key, , percent]) => <td key={key}>{percent ? score(row[key] as number, 2) : num(row[key] as number, 4)}</td>)}</tr>)}</tbody></table></div>;
}
