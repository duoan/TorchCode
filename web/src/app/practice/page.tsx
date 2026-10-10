'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowRight, BookOpen, Check, ChevronDown, CircleCheck, Clock3, Code2, Flame, History, LayoutGrid, Leaf, Minus, Plus, RefreshCw, Search, Sparkles, Target, Trophy, Users, X, Zap } from 'lucide-react';
import s from './practice.module.css';
import SolutionDialog from './SolutionDialog';
import JourneyMap from './JourneyMap';
import ProblemHistory, { type HistoryProblem } from './ProblemHistory';

const API = 'http://localhost:8000/api';
type Profile = { id: string; name: string };
type Problem = { id: string; title: string; difficulty: string; topic: string; reason: string; kind: 'new' | 'review'; completed: boolean; estimated_minutes: number; passed: number | null; total: number | null; last_attempt_at: string | null; needs_review: boolean; study_week: number | null; study_focus: string };
type Attempt = { id: number; task_id: string; title: string; difficulty: string; success: boolean; passed: number; total: number; created_at: string; hint_used: boolean; failed_tests: string[] };
type Progress = { id: string; status: string; needs_review: boolean; attempts: number };
type Dashboard = {
  profile: Profile; goal: { target: number; completed: number };
  stats: { solved: number; total: number; streak: number; attempts: number; xp: number; today_completed: number };
  week: { day: string; label: string; count: number; is_today: boolean }[];
  plan: Problem[]; reviews: Problem[]; history: Attempt[]; problem_history: HistoryProblem[]; progress: Progress[];
  curriculum: { week: number; focus: string; task_ids: string[]; solved: number; total: number }[];
};
type Task = { id: string; title: string; difficulty: string };
type Section = 'today' | 'review' | 'history' | 'library';

function localDay() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : 'We couldn’t load your practice. Try again in a moment.');
  }
  return response.json();
}

function Torch({ mini = false }: { mini?: boolean }) {
  return <svg viewBox="0 0 240 250" className={mini ? s.miniTorch : s.torch} aria-hidden="true">
    <ellipse cx="122" cy="229" rx="64" ry="10" fill="#123b2a" opacity=".09" />
    <path d="M76 185q-25 17-37-5" fill="none" stroke="#ef7d28" strokeWidth="13" strokeLinecap="round" />
    <path d="M173 170q27-13 28-35" fill="none" stroke="#ef7d28" strokeWidth="13" strokeLinecap="round" />
    <path d="M89 208l-5 17m70-17 7 17" stroke="#13392f" strokeWidth="13" strokeLinecap="round" />
    <path d="M120 12c13 50 53 46 50 94 13-8 19-20 16-36 49 52 43 108 8 136-36 30-106 27-135-10-26-34-15-85 17-112-2 29 8 36 16 39-6-43 18-51 28-111Z" fill="#ff963e" />
    <path d="M120 12c10 37 34 50 45 67-13-5-34 0-45 16-14 19-15 33-22 49-1-14-4-27-6-21-6-43 18-51 28-111Z" fill="#ffb754" />
    <path d="M97 153c-8-25 39-53 35-25 2 22 33 27 37 46 12 52-73 62-88 18-4-11 1-27 16-39Z" fill="#ffe49b" />
    <ellipse cx="95" cy="151" rx="9" ry="12" fill="#17382e" /><ellipse cx="153" cy="151" rx="9" ry="12" fill="#17382e" />
    <circle cx="98" cy="147" r="3" fill="white" /><circle cx="156" cy="147" r="3" fill="white" />
    <ellipse cx="78" cy="169" rx="10" ry="5" fill="#f77b45" opacity=".7" /><ellipse cx="172" cy="169" rx="10" ry="5" fill="#f77b45" opacity=".7" />
    <path d="M112 173q12 15 25 0" stroke="#17382e" strokeWidth="5" fill="none" strokeLinecap="round" />
    <path d="m200 112 4 9 10 1-8 6 2 10-8-5-8 5 2-10-8-6 10-1Z" fill="#fff3ae" />
  </svg>;
}

function PracticeLink({ id, user, onSolution, children = 'Practice It!', quiet = false }: { id: string; user: string; onSolution: (id: string) => void; children?: React.ReactNode; quiet?: boolean }) {
  return <div className={s.problemActions}><a className={quiet ? s.quietButton : s.practiceButton} href={`http://localhost:3000/?task=${encodeURIComponent(id)}&user=${encodeURIComponent(user)}`} target="_blank" rel="noreferrer">{children}<ArrowRight size={16} /></a><button className={s.solutionButton} onClick={() => onSolution(id)} aria-label={`Solution for ${id}`}><BookOpen size={14} />Solution</button></div>;
}

export default function PracticePage() {
  const [data, setData] = useState<Dashboard | null>(null);
  const [solutionId, setSolutionId] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [user, setUser] = useState('local');
  const [goal, setGoal] = useState(3);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const [section, setSection] = useState<Section>('today');
  const [showProfiles, setShowProfiles] = useState(false);
  const [profileName, setProfileName] = useState('');
  const [savingProfile, setSavingProfile] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');
  const [busyReview, setBusyReview] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const requestVersion = useRef(0);
  const modalRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const initialUser = new URLSearchParams(window.location.search).get('user') || localStorage.getItem('torchcode_practice_user') || 'local';
    const storedGoal = Number(localStorage.getItem(`torchcode_goal_${initialUser}`) || 3);
    setUser(initialUser);
    setGoal(Number.isInteger(storedGoal) && storedGoal >= 1 && storedGoal <= 10 ? storedGoal : 3);
    setReady(true);
  }, []);

  const refresh = useCallback(async () => {
    if (!ready) return;
    const version = ++requestVersion.current;
    setRefreshing(true);
    try {
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const params = new URLSearchParams({ user_id: user, count: String(goal), day: localDay(), timezone });
      const next = await request<Dashboard>(`/practice/dashboard?${params}`);
      if (version === requestVersion.current) { setData(next); setError(''); }
    } catch (err) {
      if (version === requestVersion.current) setError(err instanceof Error ? err.message : 'Something went wrong. Please try again.');
    } finally {
      if (version === requestVersion.current) setRefreshing(false);
    }
  }, [user, goal, ready]);

  useEffect(() => {
    const versionCounter = requestVersion;
    const timer = window.setTimeout(refresh, 0);
    const onFocus = () => { if (!document.hidden) void refresh(); };
    const interval = window.setInterval(onFocus, 15000);
    window.addEventListener('focus', onFocus);
    document.addEventListener('visibilitychange', onFocus);
    return () => { window.clearTimeout(timer); window.clearInterval(interval); window.removeEventListener('focus', onFocus); document.removeEventListener('visibilitychange', onFocus); versionCounter.current++; };
  }, [refresh]);

  useEffect(() => {
    request<{ profiles: Profile[] }>('/practice/profiles').then(result => setProfiles(result.profiles)).catch(() => {});
    request<Task[]>('/tasks').then(setTasks).catch(() => {});
  }, []);

  useEffect(() => {
    if (showProfiles) modalRef.current?.showModal();
    else modalRef.current?.close();
  }, [showProfiles]);

  function changeUser(id: string) {
    requestVersion.current++;
    setUser(id); setData(null); setError(''); setQuery(''); setFilter('all');
    const value = Number(localStorage.getItem(`torchcode_goal_${id}`) || 3);
    setGoal(value >= 1 && value <= 10 ? value : 3);
    localStorage.setItem('torchcode_practice_user', id);
    const url = new URL(window.location.href); url.searchParams.set('user', id); window.history.replaceState({}, '', url);
    setShowProfiles(false);
  }

  function changeGoal(value: number) {
    if (value < 1 || value > 10) return;
    setGoal(value); localStorage.setItem(`torchcode_goal_${user}`, String(value));
  }

  async function createProfile(event: React.FormEvent) {
    event.preventDefault(); if (!profileName.trim()) return;
    setSavingProfile(true);
    try {
      const profile = await request<Profile>('/practice/profiles', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: profileName.trim() }) });
      setProfiles(previous => [...previous, profile]); setProfileName(''); changeUser(profile.id);
    } catch (err) { setError(err instanceof Error ? err.message : 'Couldn’t create your profile.'); }
    finally { setSavingProfile(false); }
  }

  async function flagReview(taskId: string, needsReview: boolean) {
    setBusyReview(taskId);
    try {
      await request('/practice/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ user_id: user, task_id: taskId, needs_review: needsReview }) });
      await refresh();
    } catch (err) { setError(err instanceof Error ? err.message : 'Couldn’t update your review list.'); }
    finally { setBusyReview(''); }
  }

  const progress = new Map(data?.progress.map(item => [item.id, item]));
  const firstTask = data?.plan.find(item => !item.completed);
  const percent = data ? Math.min(100, Math.round(data.goal.completed / Math.max(1, data.goal.target) * 100)) : 0;
  const visibleHistory = (data?.problem_history || []).filter(item => (filter === 'all' || (filter === 'solved' ? item.solved : !item.solved || item.needs_review)) && item.title.toLowerCase().includes(query.toLowerCase()));
  const visibleTasks = tasks.filter(item => (filter === 'all' || item.difficulty.toLowerCase() === filter) && item.title.toLowerCase().includes(query.toLowerCase()));
  const nav = [{ id: 'today' as const, label: 'Today’s practice', icon: LayoutGrid }, { id: 'review' as const, label: 'Review lab', icon: RefreshCw }, { id: 'history' as const, label: 'Your journey', icon: History }, { id: 'library' as const, label: 'Problem library', icon: BookOpen }];

  function navigate(next: Section) { setSection(next); setQuery(''); setFilter('all'); document.getElementById('practice-scroll')?.scrollTo({ top: 0, behavior: 'smooth' }); }

  const reviewSection = <section className={s.reviewSection}>
    <div className={s.sectionHeading}><div><span className={s.eyebrow}>A LITTLE RECALL GOES A LONG WAY</span><h2>Bring it back <span className={s.countPill}>{data?.reviews.length || 0}</span></h2><p>Turn those tricky moments into second nature.</p></div><RefreshCw className={s.reviewHeadingIcon} size={25} /></div>
    {data?.reviews.length ? <div className={s.reviewList}>{data.reviews.map(item => <article className={s.reviewCard} key={item.id}>
      <div className={s.reviewGlyph}><RefreshCw size={21} /></div><div className={s.reviewText}><h3>{item.title}</h3><p>{item.reason}</p>{item.total != null && <span className={s.reviewScore}>{item.passed}/{item.total} tests last attempt</span>}</div><PracticeLink id={item.id} user={user} onSolution={setSolutionId} quiet>Try again</PracticeLink>
    </article>)}</div> : <div className={s.reviewEmpty}><div className={s.reviewGlyph}><Leaf size={25} /></div><div><h3>A fresh start looks good on you.</h3><p>Tricky problems will land here for a little extra practice. Solved problems return when it’s time to refresh.</p></div><span className={s.seedDecoration} aria-hidden="true">✦</span></div>}
  </section>;

  const historySection = <section className={s.historySection}>
    <div className={s.sectionHeading}><div><span className={s.eyebrow}>EVERY TRY IS A STEP FORWARD</span><h2>{section === 'today' ? 'Your recent practice' : 'Practice history'}</h2><p>One problem, all your attempts. Expand a problem to see each submission.</p></div>{section === 'today' && <button className={s.textButton} onClick={() => navigate('history')}>Your journey <ArrowRight size={16} /></button>}</div>
    {section === 'history' && <div className={s.filters}><div className={s.filterTabs}>{['all', 'solved', 'working'].map(value => <button key={value} className={filter === value ? s.selectedFilter : ''} onClick={() => setFilter(value)}>{value === 'all' ? 'All problems' : value === 'solved' ? 'Solved' : 'Still growing'}</button>)}</div><label className={s.search}><Search size={17} /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find a problem…" aria-label="Search history" /></label></div>}
    {data?.problem_history?.length ? <>
      <ProblemHistory items={section === 'today' ? visibleHistory.slice(0, 5) : visibleHistory} user={user} busyReview={busyReview} onSolution={setSolutionId} onReview={(id, needsReview) => void flagReview(id, needsReview)} />
      {!visibleHistory.length && <p className={s.noResults}>No problems match your search.</p>}
    </> : <div className={s.historyEmpty}><div className={s.emptyIcon}><History size={26} /></div><h3>Your story starts with one try.</h3><p>Run tests in the editor and your results will appear here.<br />Every attempt counts, even the messy ones.</p>{data?.plan[0] && <PracticeLink id={data.plan[0].id} user={user} onSolution={setSolutionId}>Make your first move</PracticeLink>}<span className={s.historyNote}>Previously solved problems appear on your map. Their old attempt times weren’t recorded.</span></div>}
  </section>;

  return <div className={s.app}>
    <aside className={s.sidebar}>
      <button className={s.brand} onClick={() => navigate('today')} aria-label="TorchCode home"><span className={s.brandIcon}><Flame size={23} fill="currentColor" /></span><span>torch<span className={s.brandLight}>code</span><small>YOUR DAILY SPARK</small></span></button>
      <div className={s.navLabel}>LET’S BUILD A HABIT</div><nav aria-label="Practice navigation">{nav.map(item => <button key={item.id} onClick={() => navigate(item.id)} className={`${s.navItem} ${section === item.id ? s.activeNav : ''}`}><item.icon size={20} /><span>{item.label}</span>{item.id === 'review' && !!data?.reviews.length && <b>{data.reviews.length}</b>}</button>)}</nav>
      <div className={s.sidebarBottom}><div className={s.smallCoach}><Torch mini /><h3>Little by little.<br />You’ve got this.</h3><p>Consistency is your<br />superpower.</p></div><a href={`http://localhost:3000/?user=${encodeURIComponent(user)}`} target="_blank" rel="noreferrer" className={s.editorLink}><Code2 size={18} /> Open the editor <ArrowRight size={15} /></a><button className={s.sidebarProfile} onClick={() => setShowProfiles(true)}><span className={s.avatar}>{(data?.profile.name || 'Y').slice(0, 1).toUpperCase()}</span><span>{data?.profile.name || 'You'}<small>Local learner profile</small></span><ChevronDown size={15} /></button></div>
    </aside>

    <div className={s.viewport} id="practice-scroll">
      <header className={s.topbar}><div><span className={s.headerKicker}>YOUR PERSONAL PRACTICE CLUB</span><span className={s.headerGreeting}>{data?.profile.name && data.profile.name !== 'You' ? `Hey, ${data.profile.name}.` : 'Hey, builder.'} <span aria-hidden="true">✌️</span></span></div><div className={s.topbarStats}><span className={s.streakPill}><Flame size={18} fill="currentColor" />{data?.stats.streak || 0}<span>day streak</span></span><span className={s.xpPill}><Zap size={17} fill="currentColor" />{data?.stats.xp || 0}<span>XP</span></span><button className={s.topAvatar} onClick={() => setShowProfiles(true)} aria-label="Switch learner profile">{(data?.profile.name || 'Y').slice(0, 1).toUpperCase()}</button></div></header>

      <main className={s.main}>
        {error && <div className={s.errorBanner} role="alert"><span>{error}</span><button onClick={() => void refresh()}>Try again <RefreshCw size={15} /></button></div>}
        {!data ? <div className={s.loading} role="status"><Torch /><h1>{error ? 'Your practice is waiting.' : 'Warming up your daily spark…'}</h1><p>{error ? 'Make sure TorchCode is running with ./start.sh.' : 'Putting together a little progress, just for you.'}</p></div> : <>
          {section === 'today' && <>
            <section className={s.hero}>
              <div className={s.heroContent}><div className={s.heroEyebrow}><span /> SMALL HABITS. BIG POSSIBILITIES.</div><h1>Small steps.<br /><span>Serious skills.</span></h1><p>Your next PyTorch breakthrough starts with a little<br className={s.desktopBreak} /> practice today. Let’s make it a good one.</p>{firstTask ? <PracticeLink id={firstTask.id} user={user} onSolution={setSolutionId}>Let’s practice <ArrowRight size={17} /></PracticeLink> : <button className={s.practiceButton} onClick={() => navigate('library')}>{data.plan.length ? 'Mission complete! Explore more' : 'You’re caught up! Explore more'} <ArrowRight size={17} /></button>}<span className={s.heroFootnote}><Clock3 size={14} /> About {data.plan.filter(item => !item.completed).reduce((total, item) => total + item.estimated_minutes, 0)} minutes of focus. A whole lot of growth.</span></div>
              <div className={s.heroArt}><span className={s.orbitOne} /><span className={s.orbitTwo} /><span className={s.artSparkOne}>✦</span><span className={s.artSparkTwo}>✧</span><div className={s.codeFloat}><Code2 size={21} /> one tensor at a time</div><Torch /><div className={s.youGotThis}><Sparkles size={15} /> You’ve got this!</div><div className={s.artDot} /></div>
            </section>

            <div className={s.overviewStrip}><div><span className={s.overviewIcon}><CircleCheck size={20} /></span><strong>{data.stats.solved}<small>/{data.stats.total}</small></strong><span>problems solved</span></div><div><span className={s.overviewIconAmber}><Flame size={20} /></span><strong>{data.stats.streak}<small> days</small></strong><span>showing up</span></div><div><span className={s.overviewIconPurple}><RefreshCw size={19} /></span><strong>{data.reviews.length}</strong><span>ready for review</span></div><div><span className={s.overviewIconBlue}><Target size={20} /></span><strong>{data.stats.attempts}</strong><span>steps forward</span></div></div>

            <section className={s.studyTrack} aria-label="README Suggested Study Plan"><div className={s.studyTrackHeading}><BookOpen size={15} /><span>YOUR STUDY PATH</span><small>Following the README’s Suggested Study Plan</small></div><div className={s.studyWeeks}>{data.curriculum?.map(week => <div key={week.week} className={week.solved === week.total ? s.finishedWeek : firstTask?.study_week === week.week ? s.currentWeek : ''}><span>{week.solved === week.total ? <Check size={12} /> : `0${week.week}`}</span><div><small>WEEK {week.week}</small><strong>{week.focus}</strong></div><i>{week.solved}/{week.total}</i></div>)}</div></section>
            <div className={s.dailyGrid}>
              <section className={s.missionSection}><div className={s.sectionHeading}><div><span className={s.eyebrow}>{new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' }).toUpperCase()}</span><h2>Your daily mission <span aria-hidden="true">✦</span></h2></div><div className={s.goalControl}><span>My goal</span><button disabled={goal <= 1} onClick={() => changeGoal(goal - 1)} aria-label="Fewer daily problems"><Minus size={14} /></button><output aria-label="Daily problem goal">{goal}</output><button disabled={goal >= 10} onClick={() => changeGoal(goal + 1)} aria-label="More daily problems"><Plus size={14} /></button></div></div>
                <p className={s.missionIntro}>Fresh problems follow your study path. Reviews keep it fresh. {refreshing && <span className={s.syncDot}>Updating…</span>}</p>
                <div className={s.missionCards}>{data.plan.map((item, index) => <article className={`${s.missionCard} ${item.completed ? s.completedCard : ''}`} key={item.id}><div className={s.missionNumber}>{item.completed ? <Check size={20} strokeWidth={3} /> : String(index + 1).padStart(2, '0')}</div><div className={s.missionInfo}><div className={s.problemMeta}><span className={item.kind === 'review' ? s.reviewBadge : s.newBadge}>{item.completed ? 'DONE TODAY' : item.kind === 'review' ? 'REFRESH YOUR SKILLS' : 'LEARN SOMETHING NEW'}</span><span className={`${s.difficulty} ${s[item.difficulty.toLowerCase()]}`}>{item.difficulty}</span></div><h3>{item.title}</h3><p>{item.reason}</p><span className={s.timeEstimate}><Clock3 size={13} /> ~{item.estimated_minutes} min <span>·</span> {item.study_week ? `Week ${item.study_week} · ${item.study_focus}` : item.topic}</span></div><PracticeLink id={item.id} user={user} onSolution={setSolutionId} quiet={item.completed}>{item.completed ? 'Practice again' : 'Practice It!'}</PracticeLink></article>)}</div>
                {!data.plan.length && <div className={s.reviewEmpty}><Trophy size={30} /><p>You’re caught up with your study plan and reviews. Head to the library for an optional challenge.</p></div>}
                <p className={s.planNote}><Sparkles size={14} /> A mix of fresh challenges and timely reviews. Adjust your goal from 1–10.</p>
              </section>

              <aside className={s.momentumCard}><div className={s.momentumHeader}><h3>Your momentum</h3><Leaf size={20} /></div><div className={s.progressRing} style={{ '--progress': `${percent}%` } as React.CSSProperties}><div><strong>{data.goal.completed}<span>/{data.goal.target}</span></strong><small>today’s mission</small></div></div><h4>{!data.plan.length ? 'You’re all caught up.' : percent === 100 ? 'Look at you go! 🎉' : data.goal.completed ? 'Keep that spark going.' : 'Every habit starts here.'}</h4><p>{!data.plan.length ? 'Nothing due today. Explore the library for optional practice.' : percent === 100 ? 'You showed up. You grew. See you tomorrow.' : 'One problem closer to the coder you want to be.'}</p><div className={s.weekChart}>{data.week.map(day => <div key={day.day} className={day.is_today ? s.todayColumn : ''}><div className={s.barTrack}><div style={{ height: `${Math.min(100, day.count ? Math.max(12, day.count / Math.max(goal, ...data.week.map(value => value.count)) * 100) : 0)}%` }} title={`${day.label}: ${day.count} attempts`} /></div><span>{day.label.slice(0, 1)}</span>{day.is_today && <i />}</div>)}</div><span className={s.chartLabel}>Your attempts this week</span><div className={s.momentumFooter}><Zap size={17} fill="currentColor" /><span>Showing up is already a win.</span></div></aside>
            </div>
            {reviewSection}{historySection}
          </>}
          {section === 'review' && <><div className={s.pageHeading}><div className={s.pageHeadingIcon}><RefreshCw size={28} /></div><span className={s.eyebrow}>THE COMEBACK IS PART OF THE CLIMB</span><h1>Make it stick.</h1><p>Failed tests, hint-assisted solves, and skills due for a refresh.<br />A little recall today builds confidence for tomorrow.</p></div>{reviewSection}<div className={s.reviewExplainer}><Sparkles size={21} /><div><h3>Practice, pause, revisit.</h3><p>We bring tricky problems back sooner and space out confident solves. You can also choose “Review later” in your history whenever a concept feels shaky.</p></div></div></>}
          {section === 'history' && <><JourneyMap key={user} weeks={data.curriculum} tasks={tasks} progress={data.progress} history={data.problem_history || []} user={user} onSolution={setSolutionId} onReview={(id, needsReview) => void flagReview(id, needsReview)} busyReview={busyReview} />{historySection}</>}
          {section === 'library' && <><div className={s.pageHeading}><span className={s.eyebrow}>A WORLD OF TENSORS TO EXPLORE</span><h1>Find your next challenge.</h1><p>{tasks.length} small adventures in PyTorch. Pick something that sparks your curiosity.</p></div><div className={s.filters}><div className={s.filterTabs}>{['all', 'easy', 'medium', 'hard'].map(value => <button key={value} className={filter === value ? s.selectedFilter : ''} onClick={() => setFilter(value)}>{value === 'all' ? 'All problems' : value[0].toUpperCase() + value.slice(1)}</button>)}</div><label className={s.search}><Search size={17} /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Find your next challenge…" aria-label="Search problem library" /></label></div><div className={s.libraryGrid}>{visibleTasks.map(task => <article className={s.libraryCard} key={task.id}><div className={s.libraryCardTop}><span className={`${s.difficulty} ${s[task.difficulty.toLowerCase()]}`}>{task.difficulty}</span>{progress.get(task.id)?.status === 'solved' && <span className={s.librarySolved}><CircleCheck size={14} /> Solved</span>}</div><div className={s.libraryGlyph}><Code2 size={26} /></div><h3>{task.title}</h3><p>{progress.get(task.id)?.attempts || 0} attempts{progress.get(task.id)?.needs_review ? ' · In your review list' : ''}</p><PracticeLink id={task.id} user={user} onSolution={setSolutionId}>Practice It!</PracticeLink></article>)}</div>{!visibleTasks.length && <p className={s.noResults}>No problems match your search.</p>}</>}
          <footer className={s.footer}><span><Flame size={14} /> Made for small wins and big ideas.</span><button onClick={() => void refresh()} disabled={refreshing}><RefreshCw size={13} className={refreshing ? s.spin : ''} /> {refreshing ? 'Syncing…' : 'Refresh progress'}</button><span>Keep your curiosity burning.</span></footer>
        </>}
      </main>
    </div>

    {solutionId && <SolutionDialog key={solutionId} taskId={solutionId} user={user} onClose={() => setSolutionId(null)} />}

    <dialog ref={modalRef} className={s.profileDialog} onCancel={() => setShowProfiles(false)} onClick={event => { if (event.target === event.currentTarget) setShowProfiles(false); }}><div className={s.modalHeader}><span className={s.modalIcon}><Users size={23} /></span><button onClick={() => setShowProfiles(false)} aria-label="Close profile picker"><X size={21} /></button></div><h2>Who’s practicing?</h2><p>Give everyone their own little journey.</p><div className={s.profileList}>{profiles.map(profile => <button key={profile.id} onClick={() => changeUser(profile.id)}><span className={s.avatar}>{profile.name[0].toUpperCase()}</span><strong>{profile.name}</strong>{profile.id === user ? <Check size={19} /> : <ArrowRight size={18} />}</button>)}</div><form onSubmit={createProfile}><label htmlFor="profile-name">Add a learner</label><div className={s.addProfile}><input id="profile-name" placeholder="Their name" value={profileName} maxLength={40} onChange={event => setProfileName(event.target.value)} /><button disabled={!profileName.trim() || savingProfile} type="submit" aria-label="Create learner profile"><Plus size={20} /></button></div></form><p className={s.profileNote}>Profiles live on this computer. Choose one to keep practice and progress separate.</p></dialog>
  </div>;
}
