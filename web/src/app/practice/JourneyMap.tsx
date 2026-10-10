'use client';

import { useEffect, useId, useRef, useState, type CSSProperties, type RefObject } from 'react';
import { ArrowRight, BookOpen, Check, ChevronRight, Compass, Flag, RefreshCw, Sparkles, Star, Trophy } from 'lucide-react';
import s from './journey.module.css';

export type JourneyWeek = { week: number; focus: string; task_ids: string[]; solved: number; total: number };
export type JourneyTask = { id: string; title: string; difficulty: string };
export type JourneyProgress = { id: string; status: string; needs_review: boolean; attempts: number; last_attempt_at?: string | null };
export type JourneyHistory = { id: number; task_id: string; title: string; difficulty: string; success: boolean; passed: number; total: number; created_at: string; hint_used: boolean; failed_tests: string[]; attempts: number; solved: boolean; needs_review: boolean };
export type JourneyMapProps = {
  weeks: JourneyWeek[];
  tasks: JourneyTask[];
  progress: JourneyProgress[];
  history: JourneyHistory[];
  user: string;
  onSolution: (id: string) => void;
  onReview?: (id: string, needsReview: boolean) => void;
  busyReview?: string;
};

const WORLDS = [
  { name: 'The little forest', caption: 'Plant your first building blocks', theme: s.forest },
  { name: 'Attention islands', caption: 'Connect the dots across tokens', theme: s.islands },
  { name: 'Builder’s highlands', caption: 'Bring the bigger picture to life', theme: s.highlands },
  { name: 'The summit', caption: 'Make every prediction count', theme: s.summit },
];

type LessonState = 'Solved' | 'Needs review' | 'In progress' | 'Not started';

function WorldArt({ world, large = false }: { world: number; large?: boolean }) {
  return <svg className={large ? s.largeWorldArt : s.worldArt} viewBox="0 0 240 110" fill="none" aria-hidden="true">
    <path d="M27 26h26m-13-8h35M172 18h20m-7 8h28" stroke="currentColor" strokeWidth="4" strokeLinecap="round" opacity=".16" />
    <circle cx="186" cy="35" r="12" fill="var(--sun)" />
    <ellipse cx="121" cy="97" rx="88" ry="8" fill="currentColor" opacity=".07" />
    {world === 0 && <>
      <path d="m38 79 27-19 34 5 18-13 38 5 43 20-17 17-49 6-64-8Z" fill="var(--land-shadow)" />
      <path d="m38 72 27-19 34 5 18-13 38 5 43 20-17 17-49 6-64-8Z" fill="var(--land)" />
      <path d="M104 80c8-12 24-8 27-16s-5-13 9-15" stroke="var(--trail)" strokeWidth="9" strokeLinecap="round" />
      <path d="M72 67V41m82 27V35m-58 5v-15" stroke="var(--tree-trunk)" strokeWidth="5" strokeLinecap="round" />
      <path d="m57 49 15-25 15 25Zm82-7 15-28 15 28ZM85 30l11-19 11 19Z" fill="var(--tree)" />
      <path d="m60 39 12-21 12 21m57-7 13-23 13 23" fill="var(--tree-light)" />
      <path d="M123 62V37" stroke="var(--tree-trunk)" strokeWidth="3" /><path d="m125 37 19 4-19 9Z" fill="var(--flag)" />
      <circle cx="58" cy="70" r="3" fill="var(--flower)" /><circle cx="169" cy="73" r="3" fill="var(--flower)" /><circle cx="79" cy="79" r="2" fill="var(--flower)" />
    </>}
    {world === 1 && <>
      <path d="M25 77c21-9 45 7 68 0s42-10 68-2 35-2 53-6M35 87c30-10 45 6 68-1s41-5 65 1 37-2 42-5" stroke="var(--water)" strokeWidth="4" strokeLinecap="round" />
      <path d="m35 64 30-15 35 10-12 16-33 5-21-6Zm93-13 23-13 33 13-8 16-36 1Z" fill="var(--land-shadow)" />
      <path d="m35 58 30-15 35 10-12 16-33 5-21-6Zm93-13 23-13 33 13-8 16-36 1Z" fill="var(--land)" />
      <path d="M69 55c3-14 0-24-4-29" stroke="var(--tree-trunk)" strokeWidth="5" strokeLinecap="round" />
      <path d="M65 27c-14-9-24-3-26 3 12-2 18 0 26 2-6-15 4-20 10-21-3 6-5 9-10 16 11-4 22 0 24 8-12-5-16-3-24-4" fill="var(--tree)" />
      <path d="m98 60 29-11" stroke="var(--tree-trunk)" strokeWidth="4" strokeDasharray="3 5" />
      <path d="M157 51V23" stroke="var(--tree-trunk)" strokeWidth="3" /><path d="m159 23 21 5-21 9Z" fill="var(--flag)" />
      <path d="m113 76 11-13 11 13-11 5Z" fill="var(--flag)" /><path d="M124 63V50l10 9h-10" fill="#fff8e4" />
    </>}
    {world === 2 && <>
      <path d="m30 76 37-34 30 12 25-32 31 21 35-2 26 37-55 18-82-6Z" fill="var(--land-shadow)" />
      <path d="m30 69 37-34 30 12 25-32 31 21 35-2 26 37-55 18-82-6Z" fill="var(--land)" />
      <path d="m108 33 14-18 17 12-17-4Z" fill="#fff9ed" /><path d="m53 48 14-13 14 5-13 1Z" fill="var(--tree-light)" />
      <path d="M86 67h47V43h-8v-8h-9v8h-9v-8h-9v8H86Z" fill="var(--tree-light)" stroke="var(--tree)" strokeWidth="2" />
      <path d="M106 67V55a5 5 0 0 1 10 0v12" fill="var(--tree)" />
      <path d="M109 37V18" stroke="var(--tree-trunk)" strokeWidth="3" /><path d="m111 18 19 4-19 8Z" fill="var(--flag)" />
      <path d="M118 87c-14-8-28-5-25-12s23-3 18-8" stroke="var(--trail)" strokeWidth="7" strokeLinecap="round" />
      <path d="m164 60 10-19 10 19Zm-3 12 13-23 13 23Z" fill="var(--tree)" />
    </>}
    {world === 3 && <>
      <path d="m29 83 42-46 22 19 34-45 35 36 23-15 31 52-75 15Z" fill="var(--land-shadow)" />
      <path d="m29 76 42-46 22 19 34-45 35 36 23-15 31 52-75 15Z" fill="var(--land)" />
      <path d="m105 33 22-29 27 28-17-8-10 6-8-5Z" fill="#fffdfa" /><path d="m57 45 14-15 16 15-14-6Z" fill="var(--tree-light)" /><path d="m170 40 15-15 12 20-12-6Z" fill="var(--tree-light)" />
      <path d="M120 81c-15-9 31-12 23-23s-32-8-22-19" stroke="var(--trail)" strokeWidth="7" strokeLinecap="round" />
      <path d="M128 11V-1" stroke="var(--tree-trunk)" strokeWidth="3" /><path d="m130 0 18 4-18 8Z" fill="var(--flag)" />
      <path d="m51 75 9-15 9 15Zm125 0 9-15 9 15Z" fill="var(--tree)" />
      <path d="m99 68 3-5 3 5-3 5Z" fill="var(--flower)" /><path d="m156 66 3-5 3 5-3 5Z" fill="var(--flower)" />
    </>}
    <path d="m205 43 2 5 5 2-5 2-2 5-2-5-5-2 5-2Z" fill="var(--flower)" />
  </svg>;
}

function trailPath(points: { x: number; y: number }[], width: number) {
  return points.reduce((path, point, index) => {
    if (!index) return `M${point.x} ${point.y}`;
    const previous = points[index - 1];
    if (previous.y === point.y) {
      const middle = (previous.x + point.x) / 2;
      return `${path} C${middle} ${point.y - 28},${middle} ${point.y + 28},${point.x} ${point.y}`;
    }
    const turn = point.x > width / 2 ? width - 2 : 2;
    return `${path} C${turn} ${previous.y},${turn} ${point.y},${point.x} ${point.y}`;
  }, '');
}

function Trail({ areaRef, lessonKey }: { areaRef: RefObject<HTMLDivElement | null>; lessonKey: string }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const area = areaRef.current;
    const svg = svgRef.current;
    if (!area || !svg) return;
    const nodes = Array.from(area.querySelectorAll<HTMLElement>('[data-journey-node]'));
    // Fit the trail to the rendered nodes, including rows enlarged by wrapped
    // titles, responsive column changes, and late-loading fonts.
    const draw = () => {
      const bounds = area.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const points = nodes.map(node => {
        const button = node.parentElement!;
        const rect = button.getBoundingClientRect();
        // Offsets ignore the medallion's decorative hover translation.
        return { x: rect.left - bounds.left + node.offsetLeft + node.offsetWidth / 2, y: rect.top - bounds.top + node.offsetTop + node.offsetHeight / 2 };
      });
      svg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
      const path = trailPath(points, bounds.width);
      svg.querySelectorAll('path').forEach(element => element.setAttribute('d', path));
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(area);
    nodes.forEach(node => { if (node.parentElement) observer.observe(node.parentElement); });
    return () => observer.disconnect();
  }, [areaRef, lessonKey]);

  return <svg ref={svgRef} className={s.trail} preserveAspectRatio="none" aria-hidden="true">
    <path fill="none" stroke="var(--trail-edge)" strokeWidth="15" strokeLinecap="round" />
    <path fill="none" stroke="var(--trail)" strokeWidth="11" strokeLinecap="round" />
    <path fill="none" stroke="var(--trail-dash)" strokeWidth="2" strokeDasharray="1 9" strokeLinecap="round" />
  </svg>;
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function JourneyForUser({ weeks, tasks, progress, history, user, onSolution, onReview, busyReview }: JourneyMapProps) {
  const detailId = useId();
  const pathAreaRef = useRef<HTMLDivElement>(null);
  const taskById = new Map(tasks.map(task => [task.id, task]));
  const progressById = new Map(progress.map(item => [item.id, item]));
  const historyById = new Map(history.map(item => [item.task_id, item]));
  const isSolved = (id: string) => progressById.get(id)?.status === 'solved' || historyById.get(id)?.solved === true;
  const needsReview = (id: string) => progressById.get(id)?.needs_review ?? historyById.get(id)?.needs_review ?? false;
  const attemptsFor = (id: string) => Math.max(progressById.get(id)?.attempts || 0, historyById.get(id)?.attempts || 0);
  const stateFor = (id: string): LessonState => needsReview(id) ? 'Needs review' : isSolved(id) ? 'Solved' : attemptsFor(id) > 0 || progressById.get(id)?.status === 'attempted' ? 'In progress' : 'Not started';
  const nextId = weeks.flatMap(week => week.task_ids).find(id => !isSolved(id));
  const initialWeek = weeks.find(week => week.task_ids.includes(nextId || '')) || weeks[0];
  const [selection, setSelection] = useState({ week: initialWeek?.week || 1, id: nextId || initialWeek?.task_ids[0] || '' });
  const selectedWeek = weeks.find(week => week.week === selection.week) || initialWeek;
  const selectedId = selectedWeek?.task_ids.includes(selection.id) ? selection.id : selectedWeek?.task_ids.find(id => !isSolved(id)) || selectedWeek?.task_ids[0];
  const selectedTask = selectedId ? taskById.get(selectedId) || { id: selectedId, title: selectedId.replaceAll('_', ' '), difficulty: 'Practice' } : undefined;
  const selectedHistory = selectedId ? historyById.get(selectedId) : undefined;
  const selectedProgress = selectedId ? progressById.get(selectedId) : undefined;
  const selectedState = selectedId ? stateFor(selectedId) : 'Not started';
  const total = weeks.reduce((sum, week) => sum + week.total, 0);
  const solved = weeks.reduce((sum, week) => sum + week.solved, 0);
  const worldIndex = Math.max(0, Math.min(3, (selectedWeek?.week || 1) - 1));
  const world = WORLDS[worldIndex];
  const selectedNumber = selectedWeek && selectedId ? selectedWeek.task_ids.indexOf(selectedId) + 1 : 1;
  const lastAttempt = selectedHistory?.created_at || selectedProgress?.last_attempt_at;

  function selectWeek(week: JourneyWeek) {
    if (week.week === selectedWeek?.week) return;
    setSelection({ week: week.week, id: week.task_ids.find(id => !isSolved(id)) || week.task_ids[0] || '' });
  }

  return <section className={s.journey} aria-label="Your learning journey">
    <div className={s.journeyHeader}>
      <div><span className={s.eyebrow}><Compass size={13} /> A LITTLE FURTHER, EVERY DAY</span><h1>Your journey<span aria-hidden="true">✦</span></h1><p>Four worlds to explore. {total} problems to make your own.</p></div>
      <div className={s.courseProgress}><span className={s.courseProgressIcon}><Trophy size={22} /></span><div><strong>{solved}<span> / {total}</span></strong><small>problems solved</small></div></div>
    </div>

    <div className={s.worlds} aria-label="Choose a study week">
      {weeks.map(week => {
        const index = Math.max(0, Math.min(3, week.week - 1));
        const region = WORLDS[index];
        const complete = week.total > 0 && week.solved >= week.total;
        const current = selectedWeek?.week === week.week;
        return <button key={week.week} className={`${s.world} ${region.theme} ${current ? s.selectedWorld : ''}`} onClick={() => selectWeek(week)} aria-pressed={current} aria-label={`Week ${week.week}: ${week.focus}, ${week.solved} of ${week.total} solved`}>
          <div className={s.worldTop}><span>WEEK {String(week.week).padStart(2, '0')}</span><span className={s.worldStar}>{complete ? <Star size={16} fill="currentColor" /> : <Flag size={13} />}</span></div>
          <WorldArt world={index} />
          <div className={s.worldText}><strong>{week.focus}</strong><span>{region.name}</span></div>
          <div className={s.worldProgress}><progress value={week.solved} max={Math.max(1, week.total)} aria-label={`Week ${week.week} completion`} /><span>{week.solved}/{week.total}<Check size={11} /></span></div>
          {current && <span className={s.worldSelected}>EXPLORING THIS WEEK</span>}
        </button>;
      })}
    </div>

    {selectedWeek && selectedTask && <div className={`${s.expedition} ${world.theme}`}>
      <div className={s.mapBoard}>
        <div className={s.mapHeading}><div><span>WEEK {String(selectedWeek.week).padStart(2, '0')} <span aria-hidden="true">/</span> {world.name}</span><h2>{selectedWeek.focus}</h2><p>{world.caption}.</p></div><span className={s.weekFraction}>{selectedWeek.solved}<small>/{selectedWeek.total}</small><Star size={14} fill="currentColor" /></span></div>
        <div className={s.mapLegend} aria-label="Lesson status legend"><span><i className={s.legendSolved} />Solved</span><span><i className={s.legendReview} />Needs review</span><span><i className={s.legendNew} />Ready to explore</span></div>
        <div ref={pathAreaRef} className={s.pathArea}>
          <div className={s.mapDecorations} aria-hidden="true"><span className={s.flowerOne}>✦</span><span className={s.flowerTwo}>✦</span><svg className={s.tinyTree} viewBox="0 0 40 55"><path d="M20 36v15" stroke="var(--tree-trunk)" strokeWidth="5" strokeLinecap="round" /><path d="m4 36 16-26 16 26Z" fill="var(--tree)" /><path d="m8 23 12-21 12 21Z" fill="var(--tree-light)" /></svg></div>
          <Trail areaRef={pathAreaRef} lessonKey={selectedWeek.task_ids.join(',')} />
          <ol className={s.pathGrid}>
            {selectedWeek.task_ids.map((id, index) => {
              const desktopRow = Math.floor(index / 3);
              const mobileRow = Math.floor(index / 2);
              const task = taskById.get(id);
              const state = stateFor(id);
              const chosen = selectedId === id;
              const next = nextId === id;
              const style = { '--desktop-col': desktopRow % 2 ? 3 - index % 3 : index % 3 + 1, '--desktop-row': desktopRow + 1, '--mobile-col': mobileRow % 2 ? 2 - index % 2 : index % 2 + 1, '--mobile-row': mobileRow + 1 } as CSSProperties;
              return <li key={id} className={s.lesson} style={style}>
                <button className={`${s.lessonButton} ${chosen ? s.chosenLesson : ''} ${isSolved(id) ? s.solvedLesson : ''} ${needsReview(id) ? s.reviewLesson : ''}`} onClick={() => setSelection({ week: selectedWeek.week, id })} aria-current={chosen ? 'step' : undefined} aria-controls={detailId} aria-label={`Lesson ${index + 1}: ${task?.title || id}, ${state}${next ? ', recommended next lesson' : ''}`}>
                  <span data-journey-node className={s.lessonMedallion}>{needsReview(id) ? <RefreshCw size={21} /> : isSolved(id) ? <Check size={24} strokeWidth={3} /> : <span>{index + 1}</span>}{isSolved(id) && <Star className={s.lessonStar} size={14} fill="currentColor" />}</span>
                  <span className={s.lessonName}>{task?.title || id.replaceAll('_', ' ')}</span>
                  {next ? <span className={s.nextLesson}><Sparkles size={9} /> UP NEXT</span> : <span className={s.lessonStatus}>{state === 'Not started' ? `LESSON ${String(index + 1).padStart(2, '0')}` : state}</span>}
                </button>
              </li>;
            })}
          </ol>
        </div>
        <div className={s.finishLine}><span className={s.finishIcon}><Flag size={19} /></span><div><strong>{selectedWeek.solved >= selectedWeek.total ? 'A whole world, completed!' : 'Every lesson is a little victory.'}</strong><span>{selectedWeek.solved >= selectedWeek.total ? 'Revisit a favorite or explore the next week.' : `${selectedWeek.total - selectedWeek.solved} more to complete this week. You’ve got this.`}</span></div><Star size={22} className={s.finishStar} fill="currentColor" /></div>
      </div>

      <aside className={s.lessonPanel} id={detailId} aria-label="Selected lesson details">
        <div className={s.panelScene}><WorldArt world={worldIndex} large /><span className={s.panelSceneBadge}>YOUR SELECTED LESSON</span></div>
        <div className={s.panelContent}>
          <div className={s.panelEyebrow}><span>WEEK {selectedWeek.week} <ChevronRight size={10} /> LESSON {String(selectedNumber).padStart(2, '0')}</span><span className={s.difficulty}>{selectedTask.difficulty}</span></div>
          <h3>{selectedTask.title}</h3>
          <span className={`${s.stateBadge} ${selectedState === 'Needs review' ? s.reviewState : selectedState === 'Solved' ? s.solvedState : ''}`}>{selectedState === 'Solved' ? <Check size={13} /> : selectedState === 'Needs review' ? <RefreshCw size={12} /> : <span className={s.stateDot} />}{selectedState}</span>
          <p className={s.panelIntro}>{selectedState === 'Solved' ? 'Look at you go. Keep this skill fresh with another run.' : selectedState === 'Needs review' ? 'A familiar trail worth walking again. Give this one a little extra practice.' : selectedState === 'In progress' ? 'You’ve already taken the first step. Pick up where you left off.' : 'Start with a small idea. Turn it into working PyTorch, one test at a time.'}</p>
          <a className={s.practiceButton} href={`http://localhost:3000/?task=${encodeURIComponent(selectedTask.id)}&user=${encodeURIComponent(user)}`} target="_blank" rel="noreferrer">Practice It!<ArrowRight size={17} /></a>
          <button className={s.solutionButton} onClick={() => onSolution(selectedTask.id)}><BookOpen size={15} /> Solution</button>
          <div className={s.lessonStats}><div><strong>{attemptsFor(selectedTask.id)}</strong><span>{attemptsFor(selectedTask.id) === 1 ? 'attempt' : 'attempts'}</span></div><div><strong>{selectedHistory ? <>{selectedHistory.passed}<small>/{selectedHistory.total}</small></> : '—'}</strong><span>latest tests passed</span></div></div>
          {lastAttempt ? <p className={s.lastAttempt}>Last practiced {formatDate(lastAttempt)}{selectedHistory?.hint_used ? ' · Hint used' : ''}</p> : <p className={s.lastAttempt}>{isSolved(selectedTask.id) || attemptsFor(selectedTask.id) > 0 ? 'Earlier attempt details weren’t recorded.' : 'Your first attempt is waiting for you.'}</p>}
          {!!selectedHistory?.failed_tests?.length && <div className={s.failedTests}><strong>Try these again</strong><ul>{selectedHistory.failed_tests.map((test, index) => <li key={`${test}-${index}`}><span aria-hidden="true">•</span>{test}</li>)}</ul></div>}
          {onReview && <button className={s.reviewButton} disabled={busyReview === selectedTask.id} onClick={() => onReview(selectedTask.id, !needsReview(selectedTask.id))}><RefreshCw size={13} />{busyReview === selectedTask.id ? 'Updating…' : needsReview(selectedTask.id) ? 'Remove from review' : 'Save for review'}</button>}
        </div>
        <div className={s.panelFootnote}><Sparkles size={14} /><span>Choose any lesson. Follow your curiosity.</span></div>
      </aside>
    </div>}
  </section>;
}

export default function JourneyMap(props: JourneyMapProps) {
  return <JourneyForUser key={props.user} {...props} />;
}
