"use client";

import React, { useState, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { Play, RotateCcw, ChevronDown, Check, X, AlertCircle } from 'lucide-react';
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

export default function Home() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [taskDetails, setTaskDetails] = useState<TaskDetails | null>(null);
  const [code, setCode] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [results, setResults] = useState<SubmissionResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'problem' | 'results'>('problem');
  const [solvedTasks, setSolvedTasks] = useState<Record<string, boolean>>({});
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);

  useEffect(() => {
    // Load solved tasks from local storage
    const savedSolved = localStorage.getItem('torchcode_solved_tasks');
    if (savedSolved) {
      try {
        setSolvedTasks(JSON.parse(savedSolved));
      } catch (e) {
        console.error("Failed to parse solved tasks", e);
      }
    }

    fetch(`${API_BASE}/tasks`)
      .then(res => res.json())
      .then((data: Task[]) => {
        setTasks(data);
        if (data.length > 0) {
          // Select 'relu' initially if available, otherwise first task
          const defaultTask = data.find(t => t.id === 'relu') || data[0];
          setSelectedTaskId(defaultTask.id);
        }
      })
      .catch(err => console.error("Failed to load tasks", err));
  }, []);

  useEffect(() => {
    if (selectedTaskId) {
      fetch(`${API_BASE}/tasks/${selectedTaskId}`)
        .then(res => res.json())
        .then((data: TaskDetails) => {
          setTaskDetails(data);
          const savedCode = localStorage.getItem(`torchcode_code_${selectedTaskId}`);
          setCode(savedCode !== null ? savedCode : data.initial_code);
          setResults(null);
          setActiveTab('problem');
        })
        .catch(err => console.error("Failed to load task details", err));
    }
  }, [selectedTaskId]);

  const handleRunCode = async () => {
    if (!selectedTaskId) return;
    
    setIsSubmitting(true);
    setResults(null);
    setActiveTab('results'); // Auto-switch to results tab
    
    try {
      const res = await fetch(`${API_BASE}/submit/${selectedTaskId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const data: SubmissionResponse = await res.json();
      setResults(data);
      
      if (data.success) {
        setSolvedTasks(prev => {
          const updated = { ...prev, [selectedTaskId]: true };
          localStorage.setItem('torchcode_solved_tasks', JSON.stringify(updated));
          return updated;
        });
      }
    } catch (err) {
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
      setIsSubmitting(false);
    }
  };

  const handleReset = () => {
    if (taskDetails && selectedTaskId) {
      localStorage.removeItem(`torchcode_code_${selectedTaskId}`);
      setCode(taskDetails.initial_code);
      setResults(null);
    }
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
                              setSelectedTaskId(t.id);
                              setIsDropdownOpen(false);
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
        
        <div className="flex space-x-3">
          <button 
            onClick={handleReset}
            className="flex items-center px-4 py-2 rounded-md bg-[#3c3c3c] hover:bg-[#4c4c4c] transition-colors"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </button>
          <button 
            onClick={handleRunCode}
            disabled={isSubmitting}
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
      <div className="flex flex-1 overflow-hidden">
        <Allotment
          defaultSizes={[100, 100]}
          minSize={300}
        >
          {/* Left Pane: Problem Description / Results Tab */}
          <div className="h-full flex flex-col border-r border-[#404040] bg-[#1e1e1e]">
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
          </div>

          <div className="flex-1 overflow-y-auto p-6">
            {activeTab === 'problem' ? (
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
                        <details className="group">
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
                  <div className="text-gray-500">Loading problem...</div>
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
          <div className="h-full flex flex-col bg-[#1e1e1e]">
            <div className="flex items-center px-4 py-2 bg-[#252526] border-b border-[#404040] text-sm text-gray-400">
              <span className="font-medium text-gray-300 flex items-center">
                <span className="w-2 h-2 rounded-full bg-green-500 mr-2"></span> Python (PyTorch)
              </span>
            </div>
            <div className="flex-1 pt-4">
              <Editor
                height="100%"
                language="python"
                theme="vs-dark"
                value={code}
                onChange={(value) => {
                  const newCode = value || "";
                  setCode(newCode);
                  if (selectedTaskId) {
                    localStorage.setItem(`torchcode_code_${selectedTaskId}`, newCode);
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
                }}
              />
            </div>
          </div>
        </Allotment>
      </div>
    </div>
  );
}
