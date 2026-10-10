'use client';

import { useEffect, useRef, useState } from 'react';
import { ArrowRight, BookOpen, Check, ChevronDown, Clock3, Code2, History, Plus, RefreshCw } from 'lucide-react';
import s from './problem-history.module.css';

export type HistoryProblem = {
  id: number; task_id: string; title: string; difficulty: string;
  success: boolean; passed: number; total: number; created_at: string;
  hint_used: boolean; failed_tests: string[];
  attempts: number; solved: boolean; needs_review: boolean;
};
type Submission = {
  id: number; success: boolean; passed: number; total: number;
  created_at: string; hint_used: boolean; has_code: boolean;
};
type SubmissionPage = { submissions: Submission[]; next_before_id: number | null };
type Props = {
  items: HistoryProblem[]; user: string; busyReview: string;
  onSolution: (taskId: string) => void;
  onReview: (taskId: string, needsReview: boolean) => void;
};

function dateLabel(value: string) {
  return new Date(value).toLocaleString(undefined, { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit' });
}

async function getSubmissions(user: string, task: string, signal: AbortSignal, cursor?: number) {
  const query = new URLSearchParams({ user_id: user, task_id: task, limit: '20' });
  if (cursor !== undefined) query.set('before_id', String(cursor));
  const response = await fetch(`http://localhost:8000/api/practice/submissions?${query}`, { signal });
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Couldn’t load these submissions.');
  return body as SubmissionPage;
}

function ProblemRow({ item, user, busyReview, onSolution, onReview }: Omit<Props, 'items'> & { item: HistoryProblem }) {
  const [expanded, setExpanded] = useState(false);
  const [rows, setRows] = useState<Submission[]>([]);
  const [cursor, setCursor] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const moreController = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!expanded) return;
    const controller = new AbortController();
    getSubmissions(user, item.task_id, controller.signal)
      .then(page => {
        if (controller.signal.aborted) return;
        setRows(page.submissions); setCursor(page.next_before_id); setLoadingMore(false); setError('');
      })
      .catch(err => { if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Couldn’t load these submissions.'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => { controller.abort(); moreController.current?.abort(); };
  }, [expanded, item.task_id, item.id, user, retry]);

  async function loadMore() {
    if (cursor === null || loadingMore) return;
    const controller = new AbortController();
    moreController.current?.abort(); moreController.current = controller;
    setLoadingMore(true); setError('');
    try {
      const page = await getSubmissions(user, item.task_id, controller.signal, cursor);
      if (controller.signal.aborted) return;
      setRows(previous => [...previous, ...page.submissions]); setCursor(page.next_before_id);
    } catch (err) {
      if (!controller.signal.aborted) setError(err instanceof Error ? err.message : 'Couldn’t load older submissions.');
    } finally {
      if (!controller.signal.aborted) setLoadingMore(false);
    }
  }

  const panelId = `problem-submissions-${item.task_id}`;
  const status = item.needs_review ? 'Needs review' : item.solved ? 'Solved' : 'In progress';
  return <article className={s.row} aria-label={`${item.title} practice history`}>
    <div className={s.summary}>
      <div className={s.problem}><span className={item.solved ? s.solvedIcon : s.workingIcon}>{item.solved ? <Check size={18} /> : <Code2 size={18} />}</span><div><h3>{item.title}</h3><p>{item.difficulty}<span>·</span><time dateTime={item.created_at}>Last tried {new Date(item.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</time></p></div></div>
      <div className={s.result}><span className={item.needs_review ? s.reviewBadge : item.solved ? s.solvedBadge : s.workingBadge}>{status}</span><small>Latest: {item.passed}/{item.total} tests passed</small></div>
      <div className={s.actions}><a href={`http://localhost:3000/?task=${encodeURIComponent(item.task_id)}&user=${encodeURIComponent(user)}`} target="_blank" rel="noreferrer">Practice<ArrowRight size={13} /></a><button onClick={() => onSolution(item.task_id)} aria-label={`Solution for ${item.task_id}`}><BookOpen size={13} />Solution</button></div>
    </div>
    <div className={s.rowFooter}><button className={s.expandButton} aria-expanded={expanded} aria-controls={panelId} aria-label={`${expanded ? 'Hide' : 'Show'} ${item.attempts} ${item.attempts === 1 ? 'submission' : 'submissions'} for ${item.title}`} onClick={() => { if (!expanded) { setLoading(true); setLoadingMore(false); setError(''); } setExpanded(value => !value); }}><History size={13} />{item.attempts} {item.attempts === 1 ? 'submission' : 'submissions'}<ChevronDown size={13} className={expanded ? s.expandedArrow : ''} /></button><button className={s.reviewButton} disabled={busyReview === item.task_id} onClick={() => onReview(item.task_id, !item.needs_review)}>{item.needs_review ? <Check size={12} /> : <Plus size={12} />}{item.needs_review ? 'In review' : 'Review later'}</button></div>
    {expanded && <section id={panelId} className={s.submissions} aria-label={`Submissions for ${item.title}`}>
      {loading ? <p className={s.loading} role="status"><RefreshCw size={14} />Loading your submissions…</p> : <>
        <ol>{rows.map(row => <li key={row.id}><span className={row.success ? s.accepted : s.retry}><Check size={13} />{row.success ? 'Passed' : row.total === 0 ? 'Execution failed' : 'Keep practicing'}</span><strong>{row.passed}/{row.total}<small>tests passed</small></strong><span className={s.submissionDate}><Clock3 size={12} /><time dateTime={row.created_at}>{dateLabel(row.created_at)}</time>{row.hint_used && <b>Hint used</b>}</span><span className={s.submissionId}>#{row.id}</span></li>)}</ol>
        {cursor !== null && <button className={s.moreButton} onClick={() => void loadMore()} disabled={loadingMore}>{loadingMore ? 'Loading…' : 'Load older submissions'}<ChevronDown size={13} /></button>}
        <p className={s.note}>To revisit the code, open Practice and choose Submission history in the editor.</p>
      </>}
      {error && <div className={s.error} role="alert"><p>{error}</p><button onClick={() => { setLoading(true); setError(''); setRetry(value => value + 1); }}>Try again<RefreshCw size={12} /></button></div>}
    </section>}
  </article>;
}

export default function ProblemHistory({ items, ...props }: Props) {
  return <div className={s.list}>{items.map(item => <ProblemRow key={`${props.user}:${item.task_id}`} item={item} {...props} />)}</div>;
}
