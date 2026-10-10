"use client";

import React, { useState, useEffect, useRef } from 'react';
import Editor, { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Play, RotateCcw, ChevronDown, Check, X, AlertCircle, Sparkles, ArrowUpRight, History } from 'lucide-react';
import SubmissionHistory, { type HistoricalSubmission } from './SubmissionHistory';
import { Allotment } from "allotment";
import "allotment/dist/style.css";

const API_BASE = "http://localhost:8000/api";

type Task = {
  id: string;
  title: string;
  difficulty: string;
};

type TaskDetails = Task & {
  hint: string;
  description: string;
  initial_code: string;
};

type TestResult = {
  name: string;
  code: string;
  passed: boolean;
  time_ms: number;
  error_msg: string | null;
  error_traceback: string | null;
  stdout: string;
  stderr: string;
};

type SubmissionResponse = {
  success: boolean;
  error?: string;
  traceback?: string;
  passed: number;
  total: number;
  total_time_ms: number;
  tests: TestResult[];
  stdout: string;
  stderr: string;
};

type PracticeDashboard = {
  profile: { id: string; name: string };
  progress: { id: string; status: 'unseen' | 'attempted' | 'solved'; needs_review: boolean }[];
};

function readSavedValue(key: string) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function saveValue(key: string, value: string | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    // Practice history is stored by the backend even if browser storage is unavailable.
  }
}

function draftKey(userId: string, taskId: string) {
  return `torchcode_code_${userId}:${taskId}`;
}

function practiceQuery(userId: string) {
  const today = new Date();
  const day = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  return new URLSearchParams({
    user_id: userId,
    count: '3',
    day,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  });
}

async function getPracticeProgress(userId: string, signal: AbortSignal) {
  const response = await fetch(`${API_BASE}/practice/dashboard?${practiceQuery(userId)}`, { signal });
  if (!response.ok) throw new Error('Failed to load practice progress');
  return response.json() as Promise<PracticeDashboard>;
}

function solvedFromProgress(progress: PracticeDashboard['progress']) {
  return Object.fromEntries(progress.filter(task => task.status === 'solved').map(task => [task.id, true]));
}

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [taskDetails, setTaskDetails] = useState<TaskDetails | null>(null);
  const [code, setCode] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [results, setResults] = useState<SubmissionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'problem' | 'results' | 'history'>('problem');
  const [loadedSubmission, setLoadedSubmission] = useState<HistoricalSubmission | null>(null);
  const [previousDraft, setPreviousDraft] = useState<{ code: string; hintUsed: boolean; results: SubmissionResponse | null; loadedSubmission: HistoricalSubmission | null } | null>(null);
  const [historyRevision, setHistoryRevision] = useState(0);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [solvedTasks, setSolvedTasks] = useState<Record<string, boolean>>({});
  const [userId, setUserId] = useState<string | null>(null);
  const [profileName, setProfileName] = useState('You');
  const [hintUsed, setHintUsed] = useState(false);
  const [hintResetCounter, setHintResetCounter] = useState(0);
  const [reviewState, setReviewState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [taskError, setTaskError] = useState<string | null>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const submissionControllerRef = useRef<AbortController | null>(null);
  const reviewControllerRef = useRef<AbortController | null>(null);

  const handleEditorMount: OnMount = (editorInstance) => {
    editorRef.current = editorInstance;
  };

  // Fallback for Allotment sash drags: re-measure Monaco's layout in case the
  // container size changes in a way automaticLayout's ResizeObserver misses.
  const handlePaneSizesChange = () => {
    editorRef.current?.layout();
  };

  useEffect(() => {
    const controller = new AbortController();
    const url = new URL(window.location.href);
    const profileId = url.searchParams.get('user') || 'local';
    const requestedTaskId = url.searchParams.get('task');
    let tasksLoaded = false;

    async function initialize() {
      try {
        const response = await fetch(`${API_BASE}/tasks`, { signal: controller.signal });
        if (!response.ok) throw new Error('Failed to load problems');
        const data: Task[] = await response.json();
        if (controller.signal.aborted) return;
        tasksLoaded = true;
        setTasks(data);
        setUserId(profileId);
        const initialTask = data.find(task => task.id === requestedTaskId)
          || data.find(task => task.id === 'relu') || data[0];
        if (initialTask) {
          setSelectedTaskId(initialTask.id);
          url.searchParams.set('task', initialTask.id);
          url.searchParams.set('user', profileId);
          window.history.replaceState(null, '', url);
        }

        // Old solved markers have no attempt dates. Import them just once for the
        // original local profile; subsequent progress comes from the backend.
        if (profileId === 'local' && !readSavedValue('torchcode_solved_tasks_imported_v1')) {
          let legacySolved: string[] = [];
          try {
            const saved: unknown = JSON.parse(readSavedValue('torchcode_solved_tasks') || '{}');
            if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
              legacySolved = Object.entries(saved).filter(([, solved]) => solved === true).map(([id]) => id);
            }
          } catch {
            console.warn('Could not read old solved markers');
          }
          if (legacySolved.length > 0) {
            const imported = await fetch(`${API_BASE}/practice/import`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ user_id: profileId, solved_task_ids: legacySolved }),
              signal: controller.signal,
            });
            if (!imported.ok) throw new Error('Failed to import previous progress');
          }
          saveValue('torchcode_solved_tasks_imported_v1', 'true');
        }
        const dashboard = await getPracticeProgress(profileId, controller.signal);
        if (controller.signal.aborted) return;
        setProfileName(dashboard.profile.name);
        setSolvedTasks(solvedFromProgress(dashboard.progress));
      } catch (error) {
        if (controller.signal.aborted) return;
        console.error('Failed to initialize practice editor', error);
        if (!tasksLoaded) setTaskError('Could not load problems. Check that the backend is running, then refresh.');
      }
    }

    void initialize();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedTaskId || !userId) return;
    const controller = new AbortController();
    fetch(`${API_BASE}/tasks/${selectedTaskId}`, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error('Failed to load problem');
        return response.json() as Promise<TaskDetails>;
      })
      .then(data => {
        if (controller.signal.aborted) return;
        setTaskDetails(data);
        const scopedCode = readSavedValue(draftKey(userId, selectedTaskId));
        const legacyCode = userId === 'local' ? readSavedValue(`torchcode_code_${selectedTaskId}`) : null;
        setCode(scopedCode ?? legacyCode ?? data.initial_code);
        setResults(null);
        setLoadedSubmission(null);
        setPreviousDraft(null);
        setActiveTab('problem');
        setHintUsed(false);
        setReviewState('idle');
        setTaskError(null);
        setIsSubmitting(false);
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Failed to load task details', error);
        setTaskError('Could not load this problem. Please choose it again or refresh.');
      });
    return () => {
      controller.abort();
      submissionControllerRef.current?.abort();
      reviewControllerRef.current?.abort();
    };
  }, [selectedTaskId, userId]);

  const handleSelectTask = (taskId: string) => {
    if (taskId === selectedTaskId) {
      setIsDropdownOpen(false);
      return;
    }
    submissionControllerRef.current?.abort();
    reviewControllerRef.current?.abort();
    setSelectedTaskId(taskId);
    setTaskDetails(null);
    setTaskError(null);
    setResults(null);
    setLoadedSubmission(null);
    setPreviousDraft(null);
    setHintUsed(false);
    setIsSubmitting(false);
    setReviewState('idle');
    setActiveTab('problem');
    setIsDropdownOpen(false);
    const url = new URL(window.location.href);
    url.searchParams.set('task', taskId);
    url.searchParams.set('user', userId || 'local');
    window.history.replaceState(null, '', url);
  };

  const handleRunCode = async () => {
    if (!selectedTaskId || !userId || taskDetails?.id !== selectedTaskId || isLoadingHistory) return;
    const controller = new AbortController();
    submissionControllerRef.current?.abort();
    submissionControllerRef.current = controller;
    
    setIsSubmitting(true);
    setResults(null);
    setReviewState('idle');
    setActiveTab('results'); // Auto-switch to results tab
    
    try {
      const res = await fetch(`${API_BASE}/submit/${selectedTaskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, user_id: userId, hint_used: hintUsed }),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error('The backend could not run this submission');
      const data: SubmissionResponse = await res.json();
      if (controller.signal.aborted) return;
      setResults(data);
      setHistoryRevision(previous => previous + 1);
      setIsSubmitting(false);
      
      if (data.success) {
        setSolvedTasks(prev => {
          const updated = { ...prev, [selectedTaskId]: true };
          return updated;
        });
      }
      // A successful run can still benefit from review after hints or retries.
      // Fetch the backend's assessment without delaying the test result.
      try {
        const dashboard = await getPracticeProgress(userId, controller.signal);
        if (controller.signal.aborted) return;
        setSolvedTasks(solvedFromProgress(dashboard.progress));
        if (dashboard.progress.find(task => task.id === selectedTaskId)?.needs_review) setReviewState('saved');
      } catch (error) {
        if (!controller.signal.aborted) console.error('Failed to refresh practice history', error);
      }
    } catch (err) {
      if (controller.signal.aborted) return;
      console.error("Submission failed", err);
      setResults({
        success: false,
        error: "Failed to connect to backend engine.",
        passed: 0,
        total: 0,
        total_time_ms: 0,
        tests: [],
        stdout: "",
        stderr: ""
      });
    } finally {
      if (!controller.signal.aborted) setIsSubmitting(false);
    }
  };

  const handleRequestReview = async () => {
    if (!selectedTaskId || !userId) return;
    const controller = new AbortController();
    reviewControllerRef.current?.abort();
    reviewControllerRef.current = controller;
    setReviewState('saving');
    try {
      const response = await fetch(`${API_BASE}/practice/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, task_id: selectedTaskId, needs_review: true }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error('Could not save review request');
      if (!controller.signal.aborted) setReviewState('saved');
    } catch (error) {
      if (controller.signal.aborted) return;
      console.error('Failed to save review request', error);
      setReviewState('error');
    }
  };

  const handleReset = () => {
    if (taskDetails && selectedTaskId && userId && !isLoadingHistory) {
      saveValue(draftKey(userId, selectedTaskId), null);
      if (userId === 'local') saveValue(`torchcode_code_${selectedTaskId}`, null);
      setCode(taskDetails.initial_code);
      setResults(null);
      setLoadedSubmission(null);
      setPreviousDraft(null);
      setHintUsed(false);
      setHintResetCounter(previous => previous + 1);
      setReviewState('idle');
    }
  };

  const handleLoadSubmission = (submission: HistoricalSubmission) => {
    if (isSubmitting || submission.code === null || submission.task_id !== selectedTaskId || !userId || taskDetails?.id !== selectedTaskId) return;
    setPreviousDraft({ code, hintUsed, results, loadedSubmission });
    setCode(submission.code);
    saveValue(draftKey(userId, selectedTaskId), submission.code);
    setLoadedSubmission(submission);
    setResults(null);
    setHintUsed(hintUsed || submission.hint_used);
    setReviewState('idle');
    editorRef.current?.setPosition({ lineNumber: 1, column: 1 });
    editorRef.current?.setScrollTop(0);
    editorRef.current?.focus();
  };

  const handleUndoHistoryLoad = () => {
    if (!previousDraft || !selectedTaskId || !userId || isSubmitting) return;
    setCode(previousDraft.code);
    saveValue(draftKey(userId, selectedTaskId), previousDraft.code);
    setHintUsed(previousDraft.hintUsed);
    setResults(previousDraft.results);
    setLoadedSubmission(previousDraft.loadedSubmission);
    setPreviousDraft(null);
    editorRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-screen bg-[#1e1e1e] text-white font-sans">
      {/* Top Navbar */}
      <nav className="flex items-center justify-between px-6 py-3 bg-[#2d2d2d] border-b border-[#404040]">
        <div className="flex items-center space-x-4">
          <span className="text-xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
            TorchCode
          </span>
          <div className="h-6 w-px bg-gray-600 mx-2"></div>
          {/* Task Selector */}
          <div className="relative">
            <button 
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center justify-between w-[22rem] bg-[#3c3c3c] hover:bg-[#4c4c4c] transition-colors text-white px-4 py-2 rounded-md outline-none focus:ring-2 focus:ring-orange-500 cursor-pointer font-medium"
            >
              <span className="truncate pr-4 text-left flex-1">
                {taskDetails ? `${taskDetails.title} (${taskDetails.difficulty})` : 'Select a problem...'}
              </span>
              <ChevronDown className={`w-4 h-4 text-gray-400 flex-shrink-0 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
            </button>

            {isDropdownOpen && (
              <>
                <div 
                  className="fixed inset-0 z-40" 
                  onClick={() => setIsDropdownOpen(false)}
                ></div>
                <div className="absolute top-full left-0 mt-2 w-96 max-h-[70vh] overflow-y-auto bg-[#2d2d2d] border border-[#404040] rounded-md shadow-2xl z-50 py-2">
                  {[
                    { label: '🟢 Easy', diff: 'Easy' },
                    { label: '🟡 Medium', diff: 'Medium' },
                    { label: '🔴 Hard', diff: 'Hard' }
                  ].map((group) => {
                    const groupTasks = tasks.filter(t => t.difficulty === group.diff);
                    if (groupTasks.length === 0) return null;
                    return (
                      <div key={group.diff} className="mb-2">
                        <div className="px-4 py-1.5 text-xs font-bold text-gray-400 uppercase tracking-wider bg-[#252526]">
                          {group.label}
                        </div>
                        {groupTasks.map(t => (
                          <button
                            key={t.id}
                            onClick={() => {
                              handleSelectTask(t.id);
                            }}
                            className={`w-full text-left px-4 py-2 text-sm hover:bg-[#3c3c3c] transition-colors flex items-center
                              ${selectedTaskId === t.id ? 'bg-[#3c3c3c] text-orange-400 font-semibold' : 'text-gray-200'}
                            `}
                          >
                            <span className={`truncate ${solvedTasks[t.id] ? 'line-through text-gray-500' : ''}`}>
                              {t.title}
                            </span>
                          </button>
                        ))}
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <a
            href={`http://localhost:4000/?user=${encodeURIComponent(userId || 'local')}`}
            className="flex items-center gap-2 rounded-md border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 text-sm font-medium text-emerald-300 transition-colors hover:border-emerald-400/50 hover:bg-emerald-400/20"
            title={`Open ${profileName}'s practice dashboard`}
          >
            <Sparkles className="h-4 w-4" />
            <span>My practice</span>
            <span className="hidden xl:inline border-l border-emerald-400/25 pl-2 text-xs text-emerald-200/70">{profileName}</span>
            <ArrowUpRight className="h-3.5 w-3.5" />
          </a>
          <button 
            onClick={handleReset}
            disabled={!taskDetails || taskDetails.id !== selectedTaskId || isSubmitting || isLoadingHistory}
            className="flex items-center px-4 py-2 rounded-md bg-[#3c3c3c] hover:bg-[#4c4c4c] disabled:opacity-50 transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </button>
          <button 
            onClick={handleRunCode}
            disabled={isSubmitting || isLoadingHistory || !taskDetails || taskDetails.id !== selectedTaskId}
            className="flex items-center px-6 py-2 rounded-md bg-green-600 hover:bg-green-500 disabled:opacity-50 transition-colors font-medium"
          >
            {isSubmitting ? (
              <span className="animate-spin mr-2">⏳</span>
            ) : (
              <Play className="w-4 h-4 mr-2 fill-current" />
            )}
            Run Tests
          </button>
        </div>
      </nav>

      {/* Main Split Content */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Allotment
          defaultSizes={[100, 100]}
          minSize={300}
          onChange={handlePaneSizesChange}
        >
          {/* Left Pane: Problem Description / Results Tab */}
          <div className="h-full min-h-0 min-w-0 flex flex-col border-r border-[#404040] bg-[#1e1e1e]">
            {/* Tabs */}
          <div className="flex border-b border-[#404040] bg-[#252526]">
            <button 
              className={`px-6 py-3 font-medium transition-colors ${activeTab === 'problem' ? 'text-white border-b-2 border-orange-500 bg-[#1e1e1e]' : 'text-gray-400 hover:text-gray-200'}`}
              onClick={() => setActiveTab('problem')}
            >
              Problem
            </button>
            <button 
              className={`px-6 py-3 font-medium transition-colors ${activeTab === 'results' ? 'text-white border-b-2 border-green-500 bg-[#1e1e1e]' : 'text-gray-400 hover:text-gray-200'}`}
              onClick={() => setActiveTab('results')}
            >
              Test Results
              {results && (
                <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${results.success ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'}`}>
                  {results.passed}/{results.total}
                </span>
              )}
            </button>
            <button
              className={`flex items-center gap-2 px-5 py-3 font-medium transition-colors disabled:opacity-50 ${activeTab === 'history' ? 'text-white border-b-2 border-orange-400 bg-[#1e1e1e]' : 'text-gray-400 hover:text-gray-200'}`}
              onClick={() => setActiveTab('history')}
              disabled={!taskDetails || taskDetails.id !== selectedTaskId}
              aria-pressed={activeTab === 'history'}
              title="View your submissions for this problem"
            >
              <History className="h-4 w-4" />History
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {activeTab === 'history' ? (
              taskDetails && selectedTaskId && userId ? <SubmissionHistory
                key={`${userId}:${selectedTaskId}:${historyRevision}`}
                userId={userId}
                taskId={selectedTaskId}
                title={taskDetails.title}
                profileName={profileName}
                disabled={isSubmitting}
                loadedId={loadedSubmission?.id ?? null}
                canUndo={previousDraft !== null}
                onLoad={handleLoadSubmission}
                onUndo={handleUndoHistoryLoad}
                onLoadingChange={setIsLoadingHistory}
              /> : <p className="text-gray-400">Choose a problem to view its submissions.</p>
            ) : activeTab === 'problem' ? (
              <div className="markdown-body">
                {taskDetails ? (
                  <>
                    <h1 className="text-2xl font-semibold mb-4 text-white">{taskDetails.title}</h1>
                    <div className="flex items-center space-x-4 mb-6">
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium 
                        ${taskDetails.difficulty === 'Easy' ? 'bg-[#2c4034] text-[#2cbb5d]' : 
                          taskDetails.difficulty === 'Medium' ? 'bg-[#403622] text-[#ffc01e]' : 
                          'bg-[#402a2a] text-[#ef4743]'}`}>
                        {taskDetails.difficulty}
                      </span>
                      {selectedTaskId && solvedTasks[selectedTaskId] && (
                        <div className="flex items-center text-[#2cbb5d] text-sm font-medium">
                          <Check className="w-4 h-4 mr-1" strokeWidth={3} />
                          Solved
                        </div>
                      )}
                    </div>
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                    >
                      {taskDetails.description}
                    </ReactMarkdown>
                    {taskDetails.hint && (
                      <div className="mt-8 border-t border-[#404040] pt-6">
                        <details
                          key={`${selectedTaskId}:${hintResetCounter}`}
                          className="group"
                          onToggle={event => {
                            if (event.currentTarget.open) setHintUsed(true);
                          }}
                        >
                          <summary className="flex items-center cursor-pointer list-none font-medium text-gray-300 hover:text-white transition-colors">
                            <span className="mr-2">💡</span> Hint
                            <ChevronDown className="w-4 h-4 ml-auto group-open:rotate-180 transition-transform" />
                          </summary>
                          <div className="mt-4 text-gray-400 text-sm pl-6 border-l-2 border-[#404040] py-1 markdown-body bg-transparent">
                            <ReactMarkdown 
                              remarkPlugins={[remarkGfm, remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                            >
                              {taskDetails.hint}
                            </ReactMarkdown>
                          </div>
                        </details>
                      </div>
                    )}
                  </>
                ) : (
                  <div className={taskError ? 'text-red-300' : 'text-gray-500'} role={taskError ? 'alert' : undefined}>
                    {taskError || 'Loading problem...'}
                  </div>
                )}
              </div>
            ) : (
              <div className="results-pane">
                {isSubmitting ? (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    Executing PyTorch code...
                  </div>
                ) : results ? (
                  <div className="space-y-6">
                    <div className={`text-2xl font-bold ${results.success ? 'text-green-400' : 'text-red-400'}`}>
                      {results.success ? "Accepted" : "Wrong Answer"}
                    </div>

                    {results.success && (
                      <div className="flex flex-wrap items-center gap-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-sm">
                        <span className="mr-auto text-gray-300">Nice work. How confident do you feel?</span>
                        <button
                          onClick={handleRequestReview}
                          disabled={reviewState === 'saving' || reviewState === 'saved'}
                          className="flex items-center gap-2 rounded-md border border-[#4c4c4c] bg-[#333333] px-3 py-2 text-gray-200 transition-colors hover:bg-[#404040] disabled:cursor-default disabled:border-emerald-500/25 disabled:bg-emerald-500/10 disabled:text-emerald-300"
                        >
                          {reviewState === 'saved' ? <Check className="h-4 w-4" /> : <RotateCcw className="h-4 w-4" />}
                          {reviewState === 'saved' ? 'Added to review' : reviewState === 'saving' ? 'Saving...' : 'Need more practice'}
                        </button>
                        {reviewState === 'error' && <span className="w-full text-xs text-red-300" role="alert">Could not save. Try again.</span>}
                      </div>
                    )}

                    {results.error && (
                      <div className="p-4 bg-red-900/30 border border-red-900 rounded-lg text-red-300">
                        <div className="font-semibold mb-2 flex items-center">
                          <AlertCircle className="w-5 h-5 mr-2" /> Error
                        </div>
                        <pre className="whitespace-pre-wrap text-sm">{results.error}</pre>
                        {results.traceback && (
                          <pre className="whitespace-pre-wrap text-xs mt-2 text-red-400/80">{results.traceback}</pre>
                        )}
                      </div>
                    )}

                    {results.tests && results.tests.length > 0 && (
                      <div className="space-y-4">
                        <h3 className="text-lg font-medium text-gray-200 border-b border-gray-700 pb-2">Test Cases</h3>
                        {results.tests.map((test, idx) => (
                          <div key={idx} className="bg-[#2d2d2d] rounded-lg border border-[#404040] overflow-hidden">
                            <div className="flex items-center justify-between p-4 bg-[#333333]">
                              <div className="flex items-center space-x-3">
                                {test.passed ? (
                                  <Check className="w-5 h-5 text-green-500" />
                                ) : (
                                  <X className="w-5 h-5 text-red-500" />
                                )}
                                <span className="font-medium text-gray-200">Case {idx + 1}: {test.name}</span>
                              </div>
                              <span className="text-xs text-gray-500">{test.time_ms.toFixed(1)}ms</span>
                            </div>

                            {/* Test Code Viewer */}
                            {test.code && (
                              <div className="border-t border-[#404040]">
                                <details className="group">
                                  <summary className="flex items-center cursor-pointer list-none text-xs font-medium text-gray-400 hover:text-gray-200 px-4 py-2 bg-[#2a2a2a] transition-colors">
                                    <ChevronDown className="w-3 h-3 mr-2 group-open:rotate-180 transition-transform" />
                                    Test Source Code
                                  </summary>
                                  <div className="bg-[#1e1e1e] p-4 text-xs font-mono text-gray-300 overflow-x-auto">
                                    <pre>{test.code}</pre>
                                  </div>
                                </details>
                              </div>
                            )}
                            
                            {/* Failed Assertions */}
                            {!test.passed && test.error_msg && (
                              <div className="p-4 border-t border-[#404040] bg-red-900/10">
                                <code className="text-xs text-red-300 bg-red-950/50 px-2 py-1 rounded block whitespace-pre-wrap">
                                  {test.error_msg}
                                </code>
                                {test.error_traceback && (
                                   <pre className="text-xs text-gray-400 mt-3 overflow-x-auto whitespace-pre-wrap font-mono leading-relaxed">{test.error_traceback}</pre>
                                )}
                              </div>
                            )}

                            {/* Stdout / Stderr for specific test if any */}
                            {test.stdout && (
                              <div className="p-4 border-t border-[#404040]">
                                <div className="text-xs text-gray-400 mb-1">Standard Output</div>
                                <pre className="text-xs text-gray-300 bg-[#1e1e1e] p-2 rounded">{test.stdout}</pre>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Global Stdout */}
                    {results.stdout && (
                      <div>
                        <h3 className="text-lg font-medium text-gray-200 border-b border-gray-700 pb-2 mb-3">Global Output</h3>
                        <pre className="p-4 bg-[#2d2d2d] rounded-lg text-sm font-mono overflow-x-auto border border-[#404040]">
                          {results.stdout}
                        </pre>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-gray-500 space-y-4 mt-20">
                    <Play className="w-16 h-16 text-[#333333]" />
                    <p>Run your code to see results here.</p>
                  </div>
                )}
              </div>
            )}
          </div>
          </div>

          {/* Right Pane: Code Editor */}
          <div className="h-full min-h-0 min-w-0 flex flex-col bg-[#1e1e1e]">
            <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 bg-[#252526] border-b border-[#404040] text-sm text-gray-400">
              <span className="font-medium text-gray-300 flex items-center">
                <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span> Python (PyTorch)
              </span>
              <div className="flex items-center gap-3">
                {loadedSubmission && <span className="text-xs text-orange-300">Loaded #{loadedSubmission.id}</span>}
                <button
                  onClick={() => setActiveTab(activeTab === 'history' ? 'problem' : 'history')}
                  disabled={!taskDetails || taskDetails.id !== selectedTaskId}
                  aria-expanded={activeTab === 'history'}
                  aria-controls="submission-history-pane"
                  className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-50 ${activeTab === 'history' ? 'border-orange-400/40 bg-orange-400/10 text-orange-200' : 'border-[#454545] text-gray-300 hover:bg-[#333333]'}`}
                ><History className="h-3.5 w-3.5" />Submission history</button>
              </div>
            </div>
            <div className="flex-1 min-h-0 min-w-0">
              <Editor
                height="100%"
                language="python"
                theme="vs-dark"
                value={code}
                onMount={handleEditorMount}
                onChange={(value) => {
                  const newCode = value || "";
                  setCode(newCode);
                  if (loadedSubmission && newCode !== loadedSubmission.code) setLoadedSubmission(null);
                  if (selectedTaskId && userId && taskDetails?.id === selectedTaskId) {
                    saveValue(draftKey(userId, selectedTaskId), newCode);
                  }
                }}
                options={{
                  minimap: { enabled: false },
                  fontSize: 14,
                  fontFamily: "'JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', monospace",
                  lineHeight: 24,
                  padding: { top: 16 },
                  scrollBeyondLastLine: false,
                  smoothScrolling: true,
                  cursorBlinking: "smooth",
                  cursorSmoothCaretAnimation: "on",
                  formatOnPaste: true,
                  readOnly: !taskDetails || taskDetails.id !== selectedTaskId,
                }}
              />
            </div>
          </div>
        </Allotment>
      </div>
    </div>
  );
}
