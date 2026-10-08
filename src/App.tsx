/**
 * Fake News Detection in Social Media Using Machine Learning
 * Main Application Dashboard
 */

import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Cpu,
  BarChart3,
  Database,
  GraduationCap,
  FileCode2,
  BookOpen,
  Copy,
  Check,
  RefreshCw,
  Search,
  ExternalLink,
  ChevronDown,
  ChevronRight,
  Info,
  Terminal,
  CloudUpload,
  AlertTriangle,
  Sparkles,
  Sun,
  Moon,
  Globe,
  Send,
  Bot,
  User,
  MessageSquare,
  RotateCcw,
  Trash2,
  Layers,
  Scale,
  CheckCircle2
} from 'lucide-react';

import { predictNews, PredictionResult, EVALUATION_METRICS } from './ml/engine';
import { computeTriModuleConsensus, TriModuleResult, WebFactCheckData } from './ml/triModuleConsensus';
import { DATASET } from './ml/dataset';
import { VIVA_QUESTIONS } from './data/vivaQuestions';
import { PROJECT_DOCUMENTATION } from './data/projectDocs';
import { PYTHON_PROJECT_FILES } from './data/pythonFiles';

type TabType = 'detector' | 'chatbot' | 'evaluation' | 'dataset' | 'viva' | 'docs' | 'code';
type ThemeMode = 'dark' | 'light';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  verdict?: string;
  sources?: Array<{ title: string; uri: string }>;
  searchQueries?: string[];
  timestamp: string;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabType>('detector');

  // Theme State with LocalStorage Persistence
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (typeof window !== 'undefined') {
      const savedTheme = localStorage.getItem('fake_news_theme') as ThemeMode | null;
      if (savedTheme === 'light' || savedTheme === 'dark') {
        return savedTheme;
      }
      return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }
    return 'dark';
  });

  const isDark = theme === 'dark';

  const toggleTheme = () => {
    const nextTheme: ThemeMode = theme === 'dark' ? 'light' : 'dark';
    setTheme(nextTheme);
    localStorage.setItem('fake_news_theme', nextTheme);
  };

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
      document.documentElement.classList.remove('light');
    } else {
      document.documentElement.classList.add('light');
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  // Detector State
  const [newsInput, setNewsInput] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<'Logistic Regression' | 'Multinomial Naive Bayes'>('Logistic Regression');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [detectorError, setDetectorError] = useState<string>('');

  // Live Web Browsing Grounding State (for inline verification in Detector tab)
  const [detectorWebResult, setDetectorWebResult] = useState<{
    reply: string;
    verdict: string;
    sources: Array<{ title: string; uri: string }>;
    searchQueries: string[];
    modelUsed: string;
  } | null>(null);
  const [isDetectorWebChecking, setIsDetectorWebChecking] = useState<boolean>(false);

  // Chatbot State (for multi-turn Chat tab)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: `Hello! I am your AI News Fact-Checker powered by Gemini with live Google Search browsing grounding.

I investigate breaking headlines, viral posts, and claims by searching current web sources in real time. I cross-reference claims against reputable journalistic outlets (Reuters, AP, BBC, WHO, Snopes) to determine whether something is **REAL**, **FAKE**, or **MISLEADING** with source links.

How can I help you investigate today?`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [chatInput, setChatInput] = useState<string>('');
  const [chatModel, setChatModel] = useState<string>('gemini-3.8-flash');
  const [enableSearchGrounding, setEnableSearchGrounding] = useState<boolean>(true);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string>('');
  const [autoBrowseWithAnalysis, setAutoBrowseWithAnalysis] = useState<boolean>(true);
  const [triConsensus, setTriConsensus] = useState<TriModuleResult | null>(null);
  const [isTriAnalyzing, setIsTriAnalyzing] = useState<boolean>(false);
  const [activeConsensusTab, setActiveConsensusTab] = useState<'all' | 'm1' | 'm2' | 'm3'>('all');

  // Master Tri-Module Integrated Analysis Function
  const handleRunTriModuleAnalysis = async (textToRun?: string) => {
    const text = (textToRun || newsInput).trim();
    if (!text || text.split(/\s+/).filter(Boolean).length < 3) {
      setDetectorError('Please enter at least 3 words to perform Tri-Module analysis.');
      return;
    }

    setDetectorError('');
    setIsTriAnalyzing(true);
    setIsAnalyzing(true);
    setIsDetectorWebChecking(true);

    // 1. Module 1: Compute Live ML Detector
    const mlRes = predictNews(text, selectedModel);
    setPrediction(mlRes);

    try {
      // 2. Fetch Modules 2 & 3 (Google Search Data + Web Fact-Checking Intelligence)
      const response = await fetch('/api/fact-check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model: 'gemini-3.8-flash' })
      });

      if (!response.ok) {
        throw new Error('Failed to retrieve web fact-checking data.');
      }

      const webData: WebFactCheckData = await response.json();
      setDetectorWebResult(webData);

      // 3. Compute Tri-Module Integrated Consensus
      const consensus = computeTriModuleConsensus(mlRes, webData);
      setTriConsensus(consensus);
    } catch (err: any) {
      setDetectorError(err.message || 'Error executing web fact check verification.');
    } finally {
      setIsTriAnalyzing(false);
      setIsAnalyzing(false);
      setIsDetectorWebChecking(false);
    }
  };

  // Function to perform live web fact-checking from the Detector
  const handleVerifyWithBrowsing = async (textToVerify?: string) => {
    return handleRunTriModuleAnalysis(textToVerify);
  };

  // Function to send a message in the multi-turn Chatbot tab
  const handleSendChatMessage = async (overrideText?: string) => {
    const textToSend = (overrideText || chatInput).trim();
    if (!textToSend || isChatLoading) return;

    setChatError('');
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const updatedMessages = [...chatMessages, userMsg];
    setChatMessages(updatedMessages);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const payloadMessages = updatedMessages.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        content: m.content
      }));

      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: payloadMessages,
          model: chatModel,
          enableSearch: enableSearchGrounding
        })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to get fact check response from Gemini.');
      }

      const data = await res.json();
      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.reply || 'No response returned.',
        verdict: data.verdict,
        sources: data.sources || [],
        searchQueries: data.searchQueries || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setChatMessages(prev => [...prev, botMsg]);
    } catch (err: any) {
      setChatError(err.message || 'Chat error occurred.');
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleClearChat = () => {
    setChatMessages([
      {
        id: 'welcome-reset',
        role: 'assistant',
        content: `Chat history cleared. What news claim or headline would you like me to investigate on the live web?`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    setChatError('');
  };

  // Dataset Explorer State
  const [datasetFilter, setDatasetFilter] = useState<'ALL' | 'REAL' | 'FAKE'>('ALL');
  const [datasetSearch, setDatasetSearch] = useState<string>('');

  // Viva State
  const [vivaCategory, setVivaCategory] = useState<string>('All');
  const [vivaSearch, setVivaSearch] = useState<string>('');
  const [expandedVivaId, setExpandedVivaId] = useState<number | null>(1);

  // Code Explorer State
  const [selectedFileIndex, setSelectedFileIndex] = useState<number>(0);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  // Pre-fill with a sample on initial load so the user sees immediate results
  useEffect(() => {
    const initialSample = `The Federal Reserve announced on Wednesday that it will maintain the benchmark interest rate between 5.25% and 5.50%. Federal Reserve Chairman Jerome Powell stated that committee members are monitoring ongoing employment figures and consumer price index trends before considering future policy rate reductions. Economic analysts noted that sustained disinflation in consumer goods has lowered immediate recession risks.`;
    setNewsInput(initialSample);
    const res = predictNews(initialSample, 'Logistic Regression');
    setPrediction(res);
    handleRunTriModuleAnalysis(initialSample);
  }, []);

  const handleAnalyze = () => {
    handleRunTriModuleAnalysis();
  };

  const handleModelChange = (model: 'Logistic Regression' | 'Multinomial Naive Bayes') => {
    setSelectedModel(model);
    if (newsInput.trim().length >= 5) {
      const res = predictNews(newsInput.trim(), model);
      setPrediction(res);
      if (detectorWebResult) {
        const consensus = computeTriModuleConsensus(res, detectorWebResult);
        setTriConsensus(consensus);
      }
    }
  };

  // Sample Clickers
  const loadSample = (type: 'REAL_1' | 'REAL_2' | 'FAKE_1' | 'FAKE_2' | 'FAKE_OVERRULE') => {
    let text = '';
    if (type === 'REAL_1') {
      text = `Astronomers using the James Webb Space Telescope have identified the most distant and oldest galaxy ever observed, designated JADES-GS-z14-0. The discovery was confirmed through spectroscopic analysis and published in peer-reviewed astronomical journals by international research teams.`;
    } else if (type === 'REAL_2') {
      text = `The World Health Organization (WHO) has issued new comprehensive recommendations aimed at reducing global mortality from cardiovascular diseases. The guidance emphasizes balanced dietary patterns, reduced sodium intake, regular moderate physical activity, and early hypertension screening in primary healthcare settings.`;
    } else if (type === 'FAKE_1') {
      text = `SHOCKING: Secret government cure for all cancers discovered in lemon peels! Doctors and pharmaceutical companies are terrified! An anonymous whistleblower has revealed that boiling lemon peels with baking soda cures all stage 4 cancers in 48 hours. Big Pharma is desperately trying to delete this post from the internet! Share before it gets taken down!`;
    } else if (type === 'FAKE_2') {
      text = `ALERT: Drinking boiled garlic water cures 100% of viral infections overnight. A top military doctor has leaked the ancient natural formula that completely destroys all coronavirus and influenza viruses within six hours. Boil eight cloves of raw garlic with honey and drink immediately. Hospitals are covering this up to keep ICU beds occupied!`;
    } else if (type === 'FAKE_OVERRULE') {
      text = `VATICAN CITY — In an official joint communique released Tuesday, Pope Francis announced a formal endorsement of Donald Trump for President of the United States. The Holy See stated that the Pontiff commended the candidate's economic platform and moral leadership, urging international leaders to unite behind the administration.`;
    }
    setNewsInput(text);
    setDetectorError('');
    handleRunTriModuleAnalysis(text);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  // Filtered dataset
  const filteredDataset = DATASET.filter(item => {
    const matchesFilter = datasetFilter === 'ALL' || item.label === datasetFilter;
    const matchesSearch = datasetSearch === '' ||
      item.title.toLowerCase().includes(datasetSearch.toLowerCase()) ||
      item.text.toLowerCase().includes(datasetSearch.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  // Filtered Viva Questions
  const vivaCategories = ['All', 'Fundamentals', 'NLP & Feature Extraction', 'Machine Learning Models', 'Dataset & Validation', 'Evaluation Metrics', 'Software Architecture', 'Limitations & Ethics', 'Advanced Extensions'];
  const filteredViva = VIVA_QUESTIONS.filter(q => {
    const matchesCat = vivaCategory === 'All' || q.category === vivaCategory;
    const matchesSearch = vivaSearch === '' ||
      q.question.toLowerCase().includes(vivaSearch.toLowerCase()) ||
      q.answer.toLowerCase().includes(vivaSearch.toLowerCase());
    return matchesCat && matchesSearch;
  });

  const wordCount = newsInput.trim() ? newsInput.trim().split(/\s+/).filter(Boolean).length : 0;
  const charCount = newsInput.length;

  return (
    <div
      className={`min-h-screen flex flex-col font-sans selection:bg-blue-600 selection:text-white transition-colors duration-200 ${
        isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
      }`}
    >
      
      {/* Top Header */}
      <header
        className={`border-b backdrop-blur sticky top-0 z-40 transition-colors duration-200 ${
          isDark
            ? 'border-slate-800 bg-slate-900/90'
            : 'border-slate-200 bg-white/90 shadow-xs'
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          
          {/* Logo & Branding */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div
                className={`h-10 w-10 rounded-lg flex items-center justify-center transition-colors ${
                  isDark
                    ? 'bg-blue-600/20 border border-blue-500/30 text-blue-400'
                    : 'bg-blue-50 border border-blue-200 text-blue-600'
                }`}
              >
                <ShieldAlert className="h-5 w-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className={`text-lg font-bold tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    Fake News Detection
                  </h1>
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-medium ${
                      isDark
                        ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        : 'bg-blue-50 text-blue-700 border border-blue-200'
                    }`}
                  >
                    CSE Mini Project
                  </span>
                </div>
                <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                  Social Media NLP &amp; Supervised Machine Learning Pipeline
                </p>
              </div>
            </div>

            {/* Mobile Theme Toggle Button */}
            <div className="md:hidden">
              <button
                onClick={toggleTheme}
                aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
                className={`p-2 rounded-lg border transition-colors flex items-center justify-center ${
                  isDark
                    ? 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-amber-400'
                    : 'bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700'
                }`}
              >
                {isDark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Right Navigation & Desktop Theme Switcher */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1 md:pb-0 scrollbar-none">
            <nav className="flex items-center gap-1 text-xs font-medium">
              <button
                onClick={() => setActiveTab('detector')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'detector'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Cpu className="h-3.5 w-3.5" />
                <span>Live Detector</span>
              </button>
              <button
                onClick={() => setActiveTab('chatbot')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'chatbot'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Globe className="h-3.5 w-3.5 text-cyan-400" />
                <span>Web Fact-Checker</span>
                <span className="text-[10px] bg-cyan-500/20 text-cyan-300 px-1 rounded font-mono">Gemini AI</span>
              </button>
              <button
                onClick={() => setActiveTab('evaluation')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'evaluation'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <BarChart3 className="h-3.5 w-3.5" />
                <span>Model Evaluation</span>
              </button>
              <button
                onClick={() => setActiveTab('dataset')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'dataset'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <Database className="h-3.5 w-3.5" />
                <span>Dataset Explorer</span>
              </button>
              <button
                onClick={() => setActiveTab('viva')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'viva'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <GraduationCap className="h-3.5 w-3.5" />
                <span>Viva Prep ({VIVA_QUESTIONS.length} Q&amp;A)</span>
              </button>
              <button
                onClick={() => setActiveTab('docs')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'docs'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <BookOpen className="h-3.5 w-3.5" />
                <span>Project Report</span>
              </button>
              <button
                onClick={() => setActiveTab('code')}
                className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 cursor-pointer ${
                  activeTab === 'code'
                    ? 'bg-blue-600 text-white shadow-xs'
                    : isDark
                    ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                }`}
              >
                <FileCode2 className="h-3.5 w-3.5" />
                <span>Python &amp; Flask Files</span>
              </button>
            </nav>

            {/* Desktop Theme Toggle Button */}
            <div className="hidden md:flex items-center pl-2 border-l border-slate-700/50">
              <button
                onClick={toggleTheme}
                aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
                title={`Switch to ${isDark ? 'Light' : 'Dark'} mode`}
                className={`px-2.5 py-1.5 rounded-md text-xs font-medium border transition-colors flex items-center gap-1.5 cursor-pointer ${
                  isDark
                    ? 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-amber-300'
                    : 'bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700'
                }`}
              >
                {isDark ? (
                  <>
                    <Sun className="h-3.5 w-3.5 text-amber-400" />
                    <span>Light Mode</span>
                  </>
                ) : (
                  <>
                    <Moon className="h-3.5 w-3.5 text-slate-600" />
                    <span>Dark Mode</span>
                  </>
                )}
              </button>
            </div>

          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">

        {/* =========================================================================
            TAB 1: LIVE DETECTOR
           ========================================================================= */}
        {activeTab === 'detector' && (
          <div className="space-y-6">
            
            {/* Project Overview Banner */}
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark
                  ? 'bg-gradient-to-r from-slate-900 via-slate-900 to-slate-800 border-slate-800'
                  : 'bg-gradient-to-r from-blue-50/70 via-white to-slate-50 border-slate-200'
              }`}
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    <span>Machine Learning News Credibility Analyzer</span>
                  </h2>
                  <p className={`text-sm mt-1 max-w-3xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                    Classifies whether an article or social media post is likely <strong className="text-emerald-600 dark:text-emerald-400">REAL</strong> or <strong className="text-rose-600 dark:text-rose-400">FAKE</strong> using a supervised Machine Learning model trained with Scikit-Learn TF-IDF vectorization. Free of cost, no paid APIs.
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className={`text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Algorithm:</span>
                  <div
                    className={`inline-flex rounded-lg border p-0.5 text-xs ${
                      isDark ? 'border-slate-700 bg-slate-950' : 'border-slate-300 bg-slate-100'
                    }`}
                  >
                    <button
                      onClick={() => handleModelChange('Logistic Regression')}
                      className={`px-2.5 py-1 rounded font-medium transition cursor-pointer ${
                        selectedModel === 'Logistic Regression'
                          ? 'bg-blue-600 text-white shadow-xs'
                          : isDark
                          ? 'text-slate-400 hover:text-white'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Logistic Regression (F1: 88.9%)
                    </button>
                    <button
                      onClick={() => handleModelChange('Multinomial Naive Bayes')}
                      className={`px-2.5 py-1 rounded font-medium transition cursor-pointer ${
                        selectedModel === 'Multinomial Naive Bayes'
                          ? 'bg-blue-600 text-white shadow-xs'
                          : isDark
                          ? 'text-slate-400 hover:text-white'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Naive Bayes (F1: 84.2%)
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Split Screen Workspace */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              
              {/* Left Column: Textarea & Controls (7 Cols) */}
              <div className="lg:col-span-7 space-y-4">
                <div
                  className={`border rounded-xl p-5 flex flex-col h-full shadow-xs transition-colors ${
                    isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
                  }`}
                >
                  <div
                    className={`flex items-center justify-between pb-3 border-b ${
                      isDark ? 'border-slate-800' : 'border-slate-200'
                    }`}
                  >
                    <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                      Input News Content
                    </h3>
                    <span className={`text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                      {wordCount} words · {charCount} chars
                    </span>
                  </div>

                  {/* Sample Buttons */}
                  <div className="flex flex-wrap items-center gap-2 pt-3 pb-3">
                    <span className={`text-xs font-medium mr-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                      Quick Samples:
                    </span>
                    <button
                      onClick={() => loadSample('REAL_1')}
                      className={`text-xs px-2.5 py-1 rounded border transition flex items-center gap-1 cursor-pointer ${
                        isDark
                          ? 'bg-slate-800 hover:bg-slate-700 text-emerald-400 border-emerald-500/20'
                          : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border-emerald-300'
                      }`}
                    >
                      <span>Real: NASA Discovery</span>
                    </button>
                    <button
                      onClick={() => loadSample('REAL_2')}
                      className={`text-xs px-2.5 py-1 rounded border transition flex items-center gap-1 cursor-pointer ${
                        isDark
                          ? 'bg-slate-800 hover:bg-slate-700 text-emerald-400 border-emerald-500/20'
                          : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border-emerald-300'
                      }`}
                    >
                      <span>Real: WHO Health</span>
                    </button>
                    <button
                      onClick={() => loadSample('FAKE_1')}
                      className={`text-xs px-2.5 py-1 rounded border transition flex items-center gap-1 cursor-pointer ${
                        isDark
                          ? 'bg-slate-800 hover:bg-slate-700 text-rose-400 border-rose-500/20'
                          : 'bg-rose-50 hover:bg-rose-100 text-rose-700 border-rose-300'
                      }`}
                    >
                      <span>Fake: Lemon Cure Hoax</span>
                    </button>
                    <button
                      onClick={() => loadSample('FAKE_2')}
                      className={`text-xs px-2.5 py-1 rounded border transition flex items-center gap-1 cursor-pointer ${
                        isDark
                          ? 'bg-slate-800 hover:bg-slate-700 text-rose-400 border-rose-500/20'
                          : 'bg-rose-50 hover:bg-rose-100 text-rose-700 border-rose-300'
                      }`}
                    >
                      <span>Fake: Boiled Garlic Viral</span>
                    </button>
                    <button
                      onClick={() => loadSample('FAKE_OVERRULE')}
                      className={`text-xs px-2.5 py-1 rounded border transition flex items-center gap-1 cursor-pointer ${
                        isDark
                          ? 'bg-purple-950/60 hover:bg-purple-900/60 text-purple-300 border-purple-500/30'
                          : 'bg-purple-50 hover:bg-purple-100 text-purple-700 border-purple-300'
                      }`}
                      title="Formal language hoax where Web Fact-Checker overrules inaccurate ML Detector"
                    >
                      <span>Hoax: Pope Endorsement</span>
                    </button>
                    <button
                      onClick={() => {
                        setNewsInput('');
                        setPrediction(null);
                        setTriConsensus(null);
                        setDetectorWebResult(null);
                        setDetectorError('');
                      }}
                      className={`text-xs px-2 py-1 ml-auto cursor-pointer ${
                        isDark ? 'text-slate-400 hover:text-slate-200' : 'text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      Clear
                    </button>
                  </div>

                  {/* Textarea */}
                  <div className="flex-1 min-h-[220px]">
                    <textarea
                      value={newsInput}
                      onChange={(e) => {
                        setNewsInput(e.target.value);
                        setDetectorError('');
                      }}
                      placeholder="Paste news headline, article excerpt, or social media post here (minimum 3 words)..."
                      className={`w-full h-full min-h-[200px] border focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg p-3.5 text-sm resize-y outline-none leading-relaxed font-sans transition-colors ${
                        isDark
                          ? 'bg-slate-950 border-slate-800 text-slate-100 placeholder:text-slate-500'
                          : 'bg-slate-50 border-slate-300 text-slate-900 placeholder:text-slate-400'
                      }`}
                    />
                  </div>

                  {detectorError && (
                    <div className="mt-3 p-2.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-500 dark:text-rose-300 text-xs flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 shrink-0" />
                      <span>{detectorError}</span>
                    </div>
                  )}

                  {/* Submit Bar */}
                  <div
                    className={`mt-4 pt-3 border-t flex flex-col sm:flex-row sm:items-center justify-between gap-3 ${
                      isDark ? 'border-slate-800' : 'border-slate-200'
                    }`}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <button
                        onClick={() => handleRunTriModuleAnalysis()}
                        disabled={isTriAnalyzing || wordCount < 3}
                        className="bg-gradient-to-r from-blue-600 via-indigo-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 text-white text-sm font-semibold px-5 py-2 rounded-lg transition shadow-xs flex items-center gap-2 cursor-pointer"
                        title="Integrates ML Detector + Google Search Data + Web Fact-Checking"
                      >
                        {isTriAnalyzing ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin" />
                            <span>Running 3-Module Ensemble...</span>
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-4 w-4 text-cyan-200" />
                            <span>Analyze News (3-Module Consensus)</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={() => {
                          const res = predictNews(newsInput.trim(), selectedModel);
                          setPrediction(res);
                        }}
                        disabled={isTriAnalyzing || wordCount < 3}
                        className={`text-xs px-3 py-2 rounded-lg border font-medium transition flex items-center gap-1.5 cursor-pointer disabled:opacity-50 ${
                          isDark
                            ? 'bg-slate-900 border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800'
                            : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50'
                        }`}
                        title="Run purely the client-side Scikit-Learn TF-IDF model"
                      >
                        <Cpu className="h-3.5 w-3.5" />
                        <span>ML Only</span>
                      </button>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className={`text-[11px] font-mono ${isDark ? 'text-cyan-400' : 'text-cyan-700'}`}>
                        Tri-Module: ML (15%) + Search (35%) + Fact (50%)
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Prediction Output (5 Cols) */}
              <div className="lg:col-span-5 space-y-4">
                <div
                  className={`border rounded-xl p-5 shadow-xs h-full flex flex-col justify-between transition-colors ${
                    isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
                  }`}
                >
                  <div>
                    <div
                      className={`flex items-center justify-between pb-3 border-b mb-4 ${
                        isDark ? 'border-slate-800' : 'border-slate-200'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Layers className="h-4 w-4 text-cyan-500" />
                        <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                          Tri-Module Credibility Verdict
                        </h3>
                      </div>
                      <span className={`text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        {triConsensus ? triConsensus.timestamp : prediction ? prediction.timestamp : 'Awaiting input'}
                      </span>
                    </div>

                    {isTriAnalyzing ? (
                      <div className={`py-10 px-4 rounded-xl border space-y-4 text-center ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                        <div className="flex justify-center">
                          <RefreshCw className="h-8 w-8 text-cyan-500 animate-spin" />
                        </div>
                        <div>
                          <h4 className={`text-sm font-bold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                            Executing Tri-Module Credibility Pipeline
                          </h4>
                          <p className={`text-xs mt-1 max-w-sm mx-auto ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                            Fusing offline statistical NLP with real-time Google Search data and web fact-checking registries...
                          </p>
                        </div>
                        <div className="max-w-xs mx-auto space-y-2 text-left text-xs font-mono">
                          <div className="flex items-center gap-2 text-emerald-500">
                            <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
                            <span>Module 1: Live ML Detector (Computed)</span>
                          </div>
                          <div className="flex items-center gap-2 text-cyan-500">
                            <RefreshCw className="h-3.5 w-3.5 animate-spin shrink-0" />
                            <span>Module 2: Google Search Data Grounding</span>
                          </div>
                          <div className="flex items-center gap-2 text-indigo-400">
                            <RefreshCw className="h-3.5 w-3.5 animate-spin shrink-0" />
                            <span>Module 3: Web Fact-Checking Intelligence</span>
                          </div>
                        </div>
                      </div>
                    ) : !triConsensus && !prediction ? (
                      <div className={`py-14 text-center ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                        <Info className="h-8 w-8 mx-auto mb-2 opacity-40" />
                        <p className="text-sm font-medium">No text analyzed yet</p>
                        <p className="text-xs mt-1">Paste news content or click a sample button to run the Tri-Module analysis.</p>
                      </div>
                    ) : triConsensus ? (
                      <div className="space-y-4">
                        
                        {/* Final Integrated Output Banner */}
                        <div
                          className={`p-4 rounded-xl border flex items-center justify-between transition-all ${
                            triConsensus.finalVerdict === 'REAL NEWS'
                              ? isDark
                                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                                : 'bg-emerald-50 border-emerald-300 text-emerald-900 shadow-xs'
                              : isDark
                              ? 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                              : 'bg-rose-50 border-rose-300 text-rose-900 shadow-xs'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div
                              className={`h-12 w-12 rounded-xl flex items-center justify-center text-xl shrink-0 ${
                                triConsensus.finalVerdict === 'REAL NEWS'
                                  ? isDark
                                    ? 'bg-emerald-500/20 text-emerald-400'
                                    : 'bg-emerald-200 text-emerald-800'
                                  : isDark
                                  ? 'bg-rose-500/20 text-rose-400'
                                  : 'bg-rose-200 text-rose-800'
                              }`}
                            >
                              {triConsensus.finalVerdict === 'REAL NEWS' ? (
                                <ShieldCheck className="h-7 w-7" />
                              ) : (
                                <ShieldAlert className="h-7 w-7" />
                              )}
                            </div>
                            <div>
                              <span
                                className={`text-[10px] uppercase font-bold tracking-wider block ${
                                  isDark ? 'opacity-75' : 'text-slate-600'
                                }`}
                              >
                                Final Integrated Output
                              </span>
                              <h4 className="text-2xl font-black tracking-tight">
                                {triConsensus.finalVerdict}
                              </h4>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="text-3xl font-black font-mono block">
                              {triConsensus.finalConfidence}%
                            </span>
                            <span
                              className={`text-[10px] uppercase font-semibold ${
                                isDark ? 'text-slate-400' : 'text-slate-600'
                              }`}
                            >
                              Final Confidence
                            </span>
                          </div>
                        </div>

                        {/* Consensus Callout Alert */}
                        <div
                          className={`p-3 rounded-lg border text-xs space-y-1 ${
                            triConsensus.consensusType === 'WEB_FACT_OVERRULE'
                              ? isDark
                                ? 'bg-amber-950/30 border-amber-500/40 text-amber-300'
                                : 'bg-amber-50 border-amber-300 text-amber-900'
                              : isDark
                              ? 'bg-emerald-950/30 border-emerald-500/30 text-emerald-300'
                              : 'bg-emerald-50 border-emerald-200 text-emerald-900'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 font-bold">
                            {triConsensus.consensusType === 'WEB_FACT_OVERRULE' ? (
                              <>
                                <Scale className="h-3.5 w-3.5 text-amber-500" />
                                <span>⚡ Key Insight: Web Fact-Checker Overruled Live ML Detector</span>
                              </>
                            ) : (
                              <>
                                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                                <span>✓ Unanimous Agreement Across All 3 Modules</span>
                              </>
                            )}
                          </div>
                          <p className="leading-relaxed opacity-90 text-[11px]">
                            {triConsensus.consensusSummary}
                          </p>
                        </div>

                        {/* Master Probability Bar */}
                        <div
                          className={`space-y-1.5 p-3 rounded-lg border transition-colors ${
                            isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                          }`}
                        >
                          <div
                            className={`flex justify-between text-xs ${
                              isDark ? 'text-slate-400' : 'text-slate-600'
                            }`}
                          >
                            <span className="flex items-center gap-1 font-medium">
                              <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block"></span>
                              Final Real: {triConsensus.probabilities.real}%
                            </span>
                            <span className="flex items-center gap-1 font-medium">
                              <span className="h-2 w-2 rounded-full bg-rose-500 inline-block"></span>
                              Final Fake: {triConsensus.probabilities.fake}%
                            </span>
                          </div>
                          <div
                            className={`h-2.5 w-full rounded-full overflow-hidden flex ${
                              isDark ? 'bg-slate-800' : 'bg-slate-200'
                            }`}
                          >
                            <div
                              style={{ width: `${triConsensus.probabilities.real}%` }}
                              className="bg-emerald-500 transition-all duration-500"
                            />
                            <div
                              style={{ width: `${triConsensus.probabilities.fake}%` }}
                              className="bg-rose-500 transition-all duration-500"
                            />
                          </div>
                        </div>

                        {/* Tri-Module 3-Way Card Breakdown */}
                        <div className="space-y-2">
                          <span
                            className={`text-[11px] uppercase tracking-wider font-bold block ${
                              isDark ? 'text-slate-400' : 'text-slate-600'
                            }`}
                          >
                            Integrated 3-Module Breakdown &amp; Voting
                          </span>

                          <div className="space-y-2">
                            {/* Card 1: Module 1 Live ML Detector */}
                            <div
                              className={`p-3 rounded-lg border transition-colors text-xs space-y-1 ${
                                isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-bold flex items-center gap-1 text-blue-500 dark:text-blue-400">
                                  <Cpu className="h-3.5 w-3.5" />
                                  <span>1. Live ML Detector (15% Weight)</span>
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded font-mono font-bold ${
                                    triConsensus.module1.verdict === 'REAL NEWS'
                                      ? 'bg-emerald-500/10 text-emerald-500'
                                      : 'bg-rose-500/10 text-rose-500'
                                  }`}
                                >
                                  {triConsensus.module1.verdict} ({triConsensus.module1.confidence}%)
                                </span>
                              </div>
                              <p className={`text-[11px] leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                TF-IDF + Logistic Regression ({triConsensus.module1.modelUsed}). Analyzes lexical syntax (can be inaccurate without live web factual verification).
                              </p>
                              {triConsensus.module1.matchedTokens.length > 0 && (
                                <div className="flex flex-wrap gap-1 pt-1">
                                  {triConsensus.module1.matchedTokens.slice(0, 4).map((t, idx) => (
                                    <span
                                      key={idx}
                                      className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                                        t.leaning === 'FAKE'
                                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                      }`}
                                    >
                                      {t.word}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>

                            {/* Card 2: Module 2 Google Search Data */}
                            <div
                              className={`p-3 rounded-lg border transition-colors text-xs space-y-1 ${
                                isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-bold flex items-center gap-1 text-cyan-600 dark:text-cyan-400">
                                  <Globe className="h-3.5 w-3.5" />
                                  <span>2. Google Search Data (35% Weight)</span>
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded font-mono font-bold ${
                                    triConsensus.module2.status === 'CORROBORATED'
                                      ? 'bg-emerald-500/10 text-emerald-500'
                                      : 'bg-rose-500/10 text-rose-500'
                                  }`}
                                >
                                  {triConsensus.module2.status === 'CORROBORATED' ? 'Corroborated' : 'Zero Coverage'} ({triConsensus.module2.credibilityScore}%)
                                </span>
                              </div>
                              <p className={`text-[11px] leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                {triConsensus.module2.description}
                              </p>
                              {triConsensus.module2.queriesExecuted.length > 0 && (
                                <div className="text-[10px] font-mono text-cyan-600 dark:text-cyan-400 pt-0.5 truncate">
                                  🔍 {triConsensus.module2.queriesExecuted[0]}
                                </div>
                              )}
                            </div>

                            {/* Card 3: Module 3 Web Fact-Checking Intelligence */}
                            <div
                              className={`p-3 rounded-lg border transition-colors text-xs space-y-1 ${
                                isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-bold flex items-center gap-1 text-indigo-500 dark:text-indigo-400">
                                  <ShieldCheck className="h-3.5 w-3.5" />
                                  <span>3. Web Fact-Checker (50% Weight)</span>
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded font-mono font-bold ${
                                    triConsensus.module3.verdict === 'REAL NEWS'
                                      ? 'bg-emerald-500/10 text-emerald-500'
                                      : 'bg-rose-500/10 text-rose-500'
                                  }`}
                                >
                                  {triConsensus.module3.verdict} ({triConsensus.module3.confidenceScore}%)
                                </span>
                              </div>
                              <p className={`text-[11px] leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                Cross-referenced against Snopes, PolitiFact, and international wire bureaus.
                              </p>
                              {triConsensus.module3.evidencePoints.length > 0 && (
                                <ul className="text-[11px] list-disc list-inside space-y-0.5 pt-1 text-slate-300 dark:text-slate-300">
                                  {triConsensus.module3.evidencePoints.slice(0, 2).map((ev, idx) => (
                                    <li key={idx} className="truncate">{ev}</li>
                                  ))}
                                </ul>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Authoritative Web Sources Found */}
                        {triConsensus.module2.authoritativeSources.length > 0 && (
                          <div
                            className={`p-3 rounded-lg border transition-colors text-xs space-y-1.5 ${
                              isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                            }`}
                          >
                            <span className="text-[10px] uppercase font-bold text-slate-500 block">
                              Authoritative Sources Corroborated on Live Web:
                            </span>
                            <div className="space-y-1 max-h-24 overflow-y-auto pr-1">
                              {triConsensus.module2.authoritativeSources.map((s, idx) => (
                                <a
                                  key={idx}
                                  href={s.uri}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className={`p-1.5 rounded flex items-center justify-between border transition text-xs ${
                                    isDark
                                      ? 'bg-slate-900 border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300'
                                      : 'bg-white border-slate-200 hover:border-cyan-300 text-slate-700 hover:text-cyan-700'
                                  }`}
                                >
                                  <span className="truncate pr-2 text-[11px] font-medium">{s.title}</span>
                                  <ExternalLink className="h-3 w-3 shrink-0 text-cyan-500" />
                                </a>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Bottom Actions */}
                        <div className="pt-1 flex items-center justify-between gap-2">
                          <button
                            onClick={() => {
                              setActiveTab('chatbot');
                              handleSendChatMessage(`Investigate this news claim in depth: "${newsInput.trim().slice(0, 150)}..."`);
                            }}
                            className="text-xs text-cyan-600 dark:text-cyan-400 hover:underline flex items-center gap-1 font-medium cursor-pointer"
                          >
                            <MessageSquare className="h-3.5 w-3.5" />
                            <span>Ask follow-up questions in Fact-Check Chat &rarr;</span>
                          </button>
                          <button
                            onClick={() => {
                              setTriConsensus(null);
                              setPrediction(null);
                              setDetectorWebResult(null);
                              setNewsInput('');
                            }}
                            className={`text-[10px] px-2 py-1 rounded cursor-pointer ${
                              isDark ? 'text-slate-500 hover:text-slate-300' : 'text-slate-400 hover:text-slate-600'
                            }`}
                          >
                            Reset
                          </button>
                        </div>

                      </div>
                    ) : (
                      /* Fallback Offline ML Only View */
                      <div className="space-y-4">
                        <div
                          className={`p-4 rounded-xl border flex items-center justify-between transition-all ${
                            prediction!.prediction === 'REAL NEWS'
                              ? isDark
                                ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
                                : 'bg-emerald-50 border-emerald-300 text-emerald-900 shadow-xs'
                              : isDark
                              ? 'bg-rose-950/40 border-rose-500/40 text-rose-300'
                              : 'bg-rose-50 border-rose-300 text-rose-900 shadow-xs'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div className="h-11 w-11 rounded-lg flex items-center justify-center text-xl shrink-0">
                              {prediction!.prediction === 'REAL NEWS' ? (
                                <ShieldCheck className="h-6 w-6 text-emerald-400" />
                              ) : (
                                <ShieldAlert className="h-6 w-6 text-rose-400" />
                              )}
                            </div>
                            <div>
                              <span className="text-[10px] uppercase font-bold tracking-wider block opacity-75">
                                Stylistic ML Prediction
                              </span>
                              <h4 className="text-xl font-extrabold tracking-tight">
                                {prediction!.prediction}
                              </h4>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="text-2xl font-black font-mono block">
                              {prediction!.confidence}%
                            </span>
                            <span className="text-[10px] uppercase font-semibold text-slate-400">
                              ML Confidence
                            </span>
                          </div>
                        </div>

                        <button
                          onClick={() => handleRunTriModuleAnalysis()}
                          className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 text-white font-semibold py-2.5 rounded-lg text-xs transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
                        >
                          <Sparkles className="h-4 w-4" />
                          <span>Run Tri-Module Ensemble (Google Search + Fact-Check)</span>
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Mandatory Academic Disclaimer */}
                  <div
                    className={`mt-4 pt-3 border-t ${
                      isDark ? 'border-slate-800/80' : 'border-slate-200'
                    }`}
                  >
                    <p
                      className={`text-[11px] p-2.5 rounded-lg leading-relaxed border ${
                        isDark
                          ? 'text-amber-300/80 bg-amber-500/10 border-amber-500/20'
                          : 'text-amber-900 bg-amber-50 border-amber-200'
                      }`}
                    >
                      <strong>Important Disclaimer:</strong> Prediction is based on patterns learned from the training dataset and should not be treated as definitive proof that a news story is true or false.
                    </p>
                  </div>
                </div>
              </div>

            </div>

            {/* Quick Architecture Flow Strip */}
            <div
              className={`border rounded-xl p-4 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <span
                className={`text-xs uppercase font-bold tracking-wider block mb-2 ${
                  isDark ? 'text-slate-400' : 'text-slate-500'
                }`}
              >
                Machine Learning Execution Pipeline
              </span>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                <div
                  className={`border p-2.5 rounded-lg transition-colors ${
                    isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <span className="text-blue-500 dark:text-blue-400 font-mono text-[10px] block">Step 1</span>
                  <span className={`font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Raw Text Ingestion</span>
                  <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Headline, tweet, or article</p>
                </div>
                <div
                  className={`border p-2.5 rounded-lg transition-colors ${
                    isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <span className="text-blue-500 dark:text-blue-400 font-mono text-[10px] block">Step 2</span>
                  <span className={`font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Text Preprocessing</span>
                  <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Lower, URL &amp; symbol scrub</p>
                </div>
                <div
                  className={`border p-2.5 rounded-lg transition-colors ${
                    isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <span className="text-blue-500 dark:text-blue-400 font-mono text-[10px] block">Step 3</span>
                  <span className="font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}">TF-IDF Vectorization</span>
                  <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>(1, 2) n-gram matrices</p>
                </div>
                <div
                  className={`border p-2.5 rounded-lg transition-colors ${
                    isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <span className="text-blue-500 dark:text-blue-400 font-mono text-[10px] block">Step 4</span>
                  <span className={`font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Logistic Classifier</span>
                  <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Sigmoid decision threshold</p>
                </div>
                <div
                  className={`border p-2.5 rounded-lg transition-colors ${
                    isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                  }`}
                >
                  <span className="text-blue-500 dark:text-blue-400 font-mono text-[10px] block">Step 5</span>
                  <span className={`font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>Verdict &amp; Confidence</span>
                  <p className={`text-[11px] mt-0.5 ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>REAL / FAKE with prob split</p>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB: AI WEB FACT-CHECKER & BROWSING INVESTIGATOR
           ========================================================================= */}
        {activeTab === 'chatbot' && (
          <div className="space-y-6">
            {/* Header Banner */}
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark
                  ? 'bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 border-slate-800'
                  : 'bg-gradient-to-r from-cyan-50/70 via-white to-blue-50 border-cyan-200'
              }`}
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    <Globe className="h-5 w-5 text-cyan-500" />
                    <span>Live Web Fact-Checker &amp; Disinformation Investigator</span>
                  </h2>
                  <p className={`text-sm mt-1 max-w-3xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                    Directly browse the live internet using Google Search grounding. The model formulates search queries, cross-references breaking claims against wire services (Reuters, AP News, BBC, Snopes, PolitiFact), and provides fact-checking citations.
                  </p>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <div
                    className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono ${
                      isDark ? 'border-cyan-500/30 bg-cyan-950/30 text-cyan-300' : 'border-cyan-300 bg-cyan-50 text-cyan-800'
                    }`}
                  >
                    <span className="h-2 w-2 rounded-full bg-cyan-400 animate-pulse"></span>
                    <span>Google Search Grounding: Live</span>
                  </div>
                  <button
                    onClick={handleClearChat}
                    className={`p-2 rounded-lg border text-xs transition cursor-pointer flex items-center gap-1 ${
                      isDark
                        ? 'border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                        : 'border-slate-200 bg-white text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                    }`}
                    title="Clear Conversation History"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    <span className="hidden sm:inline">Reset</span>
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Investigation Suggestion Chips */}
            <div
              className={`border rounded-xl p-4 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex items-center gap-2 mb-2.5">
                <Sparkles className="h-3.5 w-3.5 text-cyan-500" />
                <span className={`text-xs font-semibold uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                  Click to Investigate Live Web Claims:
                </span>
              </div>
              <div className="flex flex-wrap gap-2">
                {[
                  {
                    label: '🔭 NASA Oldest Galaxy Discovery',
                    prompt: 'Fact-check whether NASA James Webb Space Telescope genuinely discovered the oldest galaxy in the universe (JADES-GS-z14-0).'
                  },
                  {
                    label: '🍋 Lemon Peels Cure Cancer Hoax',
                    prompt: 'Investigate the viral claim that boiling lemon peels with baking soda cures all stage 4 cancers in 48 hours.'
                  },
                  {
                    label: '🧄 Boiled Garlic Viral Cure Myth',
                    prompt: 'Fact check if drinking boiled garlic water cures coronavirus and all viral respiratory illnesses overnight.'
                  },
                  {
                    label: '🏛️ Pope Francis Endorsed Trump',
                    prompt: 'Did Pope Francis endorse Donald Trump for US President? What do Vatican records and Snopes say?'
                  },
                  {
                    label: '📡 5G Towers Brainwave Control',
                    prompt: 'Is there any scientific validity to claims that 5G cellular radiation controls human thoughts and brainwaves?'
                  },
                  {
                    label: '🌕 Chandrayaan-3 Lunar South Pole',
                    prompt: 'Verify the historical landing of Chandrayaan-3 near the lunar south pole by ISRO on August 23, 2023.'
                  }
                ].map((chip, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSendChatMessage(chip.prompt)}
                    disabled={isChatLoading}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition text-left flex items-center gap-1.5 cursor-pointer disabled:opacity-50 ${
                      isDark
                        ? 'bg-slate-950 border-slate-800 text-slate-300 hover:border-cyan-500/40 hover:text-cyan-300'
                        : 'bg-slate-50 border-slate-200 text-slate-700 hover:border-cyan-400 hover:text-cyan-900'
                    }`}
                  >
                    <span>{chip.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Conversation Window */}
            <div
              className={`border rounded-xl shadow-xs overflow-hidden flex flex-col min-h-[500px] transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              {/* Message Feed */}
              <div className="flex-1 p-4 sm:p-6 overflow-y-auto max-h-[620px] space-y-5">
                {chatMessages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex gap-3 sm:gap-4 ${
                      msg.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {msg.role !== 'user' && (
                      <div className="h-8 w-8 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center shrink-0 mt-0.5">
                        <Globe className="h-4 w-4" />
                      </div>
                    )}

                    <div
                      className={`max-w-3xl rounded-2xl p-4 sm:p-5 shadow-xs space-y-3 transition-colors ${
                        msg.role === 'user'
                          ? 'bg-blue-600 text-white ml-auto'
                          : isDark
                          ? 'bg-slate-950 border border-slate-800 text-slate-200'
                          : 'bg-slate-50 border border-slate-200 text-slate-900'
                      }`}
                    >
                      {/* Top Meta info */}
                      <div className="flex items-center justify-between gap-3 text-[11px] opacity-75 pb-1 border-b border-white/10 dark:border-slate-800">
                        <span className="font-semibold flex items-center gap-1">
                          {msg.role === 'user' ? (
                            <>
                              <User className="h-3 w-3" />
                              <span>You</span>
                            </>
                          ) : (
                            <>
                              <Bot className="h-3 w-3 text-cyan-400" />
                              <span>Web Fact-Checker (Gemini + Google Search)</span>
                            </>
                          )}
                        </span>
                        <span className="font-mono">{msg.timestamp}</span>
                      </div>

                      {/* Verdict Pill if present */}
                      {msg.verdict && (
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] uppercase tracking-wider font-bold opacity-75">
                            Investigative Finding:
                          </span>
                          <span
                            className={`text-xs px-2.5 py-0.5 rounded-full font-bold font-mono border ${
                              msg.verdict.includes('REAL')
                                ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
                                : msg.verdict.includes('FAKE')
                                ? 'bg-rose-500/20 text-rose-600 dark:text-rose-400 border-rose-500/30'
                                : 'bg-amber-500/20 text-amber-600 dark:text-amber-400 border-amber-500/30'
                            }`}
                          >
                            {msg.verdict}
                          </span>
                        </div>
                      )}

                      {/* Content Body */}
                      <div className="text-sm leading-relaxed whitespace-pre-wrap font-sans">
                        {msg.content}
                      </div>

                      {/* Google Search Queries Executed */}
                      {msg.searchQueries && msg.searchQueries.length > 0 && (
                        <div
                          className={`mt-3 p-3 rounded-xl border text-xs space-y-1.5 ${
                            isDark
                              ? 'bg-slate-900/90 border-slate-800 text-slate-300'
                              : 'bg-white border-slate-200 text-slate-700'
                          }`}
                        >
                          <span className="text-[10px] uppercase font-bold text-cyan-600 dark:text-cyan-400 block tracking-wider">
                            🔍 Live Search Queries Formulated &amp; Executed:
                          </span>
                          <div className="flex flex-wrap gap-1.5">
                            {msg.searchQueries.map((query, qIdx) => (
                              <span
                                key={qIdx}
                                className={`text-[11px] px-2.5 py-0.5 rounded font-mono border ${
                                  isDark
                                    ? 'bg-slate-950 border-cyan-500/20 text-cyan-300'
                                    : 'bg-cyan-50 border-cyan-200 text-cyan-900'
                                }`}
                              >
                                {query}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Web Sources & Grounding Links */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div
                          className={`mt-3 p-3 rounded-xl border text-xs space-y-2 ${
                            isDark
                              ? 'bg-slate-900/90 border-slate-800 text-slate-300'
                              : 'bg-white border-slate-200 text-slate-700'
                          }`}
                        >
                          <span className="text-[10px] uppercase font-bold text-cyan-600 dark:text-cyan-400 block tracking-wider">
                            🌐 Web Sources &amp; Fact-Checking Citations:
                          </span>
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                            {msg.sources.map((source, sIdx) => (
                              <a
                                key={sIdx}
                                href={source.uri}
                                target="_blank"
                                rel="noopener noreferrer"
                                className={`p-2 rounded-lg border transition flex items-center justify-between gap-2 group ${
                                  isDark
                                    ? 'bg-slate-950 border-slate-800 hover:border-cyan-500/40 text-slate-300 hover:text-cyan-300'
                                    : 'bg-slate-50 border-slate-200 hover:border-cyan-300 text-slate-800 hover:text-cyan-800'
                                }`}
                              >
                                <span className="text-xs truncate font-medium group-hover:underline">
                                  {source.title}
                                </span>
                                <ExternalLink className="h-3.5 w-3.5 shrink-0 text-cyan-500" />
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="h-8 w-8 rounded-lg bg-blue-600 text-white flex items-center justify-center shrink-0 mt-0.5">
                        <User className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                ))}

                {/* Loading indicator */}
                {isChatLoading && (
                  <div className="flex gap-3 sm:gap-4 justify-start">
                    <div className="h-8 w-8 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center shrink-0">
                      <Globe className="h-4 w-4 animate-spin" />
                    </div>
                    <div
                      className={`rounded-2xl p-4 border max-w-md space-y-2 ${
                        isDark ? 'bg-slate-950 border-slate-800 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-700'
                      }`}
                    >
                      <div className="flex items-center gap-2 text-xs font-semibold text-cyan-500">
                        <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                        <span>Actively searching live web &amp; cross-referencing sources...</span>
                      </div>
                      <p className="text-[11px] text-slate-500 dark:text-slate-400">
                        Querying Google Search grounding against international wire services, academic databases, and fact-check registries.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {chatError && (
                <div className="mx-4 sm:mx-6 mb-3 p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-500 text-xs flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 shrink-0" />
                  <span>{chatError}</span>
                </div>
              )}

              {/* Chat Input Bar */}
              <div
                className={`p-4 border-t transition-colors ${
                  isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                }`}
              >
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendChatMessage();
                  }}
                  className="space-y-2.5"
                >
                  <div className="flex gap-2">
                    <textarea
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleSendChatMessage();
                        }
                      }}
                      placeholder="Paste a news claim, headline, tweet, or rumor to investigate on the live web (Press Enter to send)..."
                      rows={2}
                      className={`flex-1 border rounded-xl p-3 text-sm outline-none resize-none transition-colors ${
                        isDark
                          ? 'bg-slate-900 border-slate-800 text-slate-100 placeholder:text-slate-500 focus:border-cyan-500'
                          : 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-cyan-500'
                      }`}
                    />
                    <button
                      type="submit"
                      disabled={isChatLoading || !chatInput.trim()}
                      className="bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-semibold px-5 rounded-xl transition flex flex-col items-center justify-center gap-1 cursor-pointer shrink-0 shadow-xs"
                      title="Investigate with Google Search"
                    >
                      <Send className="h-4 w-4" />
                      <span className="text-[11px] hidden sm:inline">Search Web</span>
                    </button>
                  </div>

                  <div className="flex flex-wrap items-center justify-between text-xs gap-2 pt-1">
                    <div className="flex items-center gap-3">
                      <label className="flex items-center gap-1.5 cursor-pointer select-none">
                        <input
                          type="checkbox"
                          checked={enableSearchGrounding}
                          onChange={(e) => setEnableSearchGrounding(e.target.checked)}
                          className="rounded border-slate-400 text-cyan-600 focus:ring-cyan-500 h-3.5 w-3.5 cursor-pointer"
                        />
                        <span className={isDark ? 'text-slate-300' : 'text-slate-700'}>
                          Google Search Grounding
                        </span>
                      </label>
                      <span className={`text-[11px] font-mono ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                        Engine: gemini-3.8-flash
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          if (newsInput.trim()) {
                            handleSendChatMessage(`Fact check this text from the detector: "${newsInput.trim().slice(0, 200)}..."`);
                          }
                        }}
                        disabled={!newsInput.trim()}
                        className={`text-[11px] px-2 py-0.5 rounded border transition cursor-pointer disabled:opacity-40 ${
                          isDark
                            ? 'border-slate-800 bg-slate-900 text-slate-400 hover:text-slate-200'
                            : 'border-slate-200 bg-white text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Import Detector Text
                      </button>
                    </div>
                  </div>
                </form>
              </div>
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB 3: MODEL EVALUATION & METRICS
           ========================================================================= */}
        {activeTab === 'evaluation' && (
          <div className="space-y-6">
            
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                <BarChart3 className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                <span>Empirical Model Evaluation &amp; Benchmarking</span>
              </h2>
              <p className={`text-sm mt-1 max-w-3xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                Evaluated on the held-out test split (80% Train, 20% Test) using Scikit-Learn metrics.
                The model with the highest F1-score is automatically selected for deployment.
              </p>
            </div>

            {/* Model Comparison Table */}
            <div
              className={`border rounded-xl overflow-hidden shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div
                className={`p-4 border-b flex items-center justify-between ${
                  isDark ? 'border-slate-800' : 'border-slate-200'
                }`}
              >
                <h3 className={`text-sm font-semibold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>
                  Model Performance Comparison Table
                </h3>
                <span
                  className={`text-xs px-2 py-0.5 rounded font-mono ${
                    isDark
                      ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      : 'bg-emerald-50 text-emerald-700 border border-emerald-300'
                  }`}
                >
                  Test Split: 20 Samples
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead
                    className={`text-xs uppercase font-mono border-b ${
                      isDark
                        ? 'bg-slate-950 text-slate-400 border-slate-800'
                        : 'bg-slate-50 text-slate-600 border-slate-200'
                    }`}
                  >
                    <tr>
                      <th className="py-3 px-4">Model Name</th>
                      <th className="py-3 px-4">Hyperparameters</th>
                      <th className="py-3 px-4">Accuracy</th>
                      <th className="py-3 px-4">Precision</th>
                      <th className="py-3 px-4">Recall</th>
                      <th className="py-3 px-4">F1-Score</th>
                      <th className="py-3 px-4">Selection Status</th>
                    </tr>
                  </thead>
                  <tbody className={`divide-y ${isDark ? 'divide-slate-800' : 'divide-slate-200'}`}>
                    <tr className={isDark ? 'bg-blue-950/20' : 'bg-blue-50/40'}>
                      <td className={`py-3.5 px-4 font-semibold flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                        <span>Logistic Regression</span>
                        <span className="text-[10px] bg-blue-500/20 text-blue-600 dark:text-blue-300 px-1.5 py-0.5 rounded font-medium">Winner</span>
                      </td>
                      <td className={`py-3.5 px-4 text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>C=1.0, penalty=l2</td>
                      <td className={`py-3.5 px-4 font-mono font-bold ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{EVALUATION_METRICS.logisticRegression.accuracy}%</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{EVALUATION_METRICS.logisticRegression.precision}%</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{EVALUATION_METRICS.logisticRegression.recall}%</td>
                      <td className="py-3.5 px-4 font-mono text-emerald-600 dark:text-emerald-400 font-bold">{EVALUATION_METRICS.logisticRegression.f1Score}%</td>
                      <td className="py-3.5 px-4 text-xs font-medium text-emerald-600 dark:text-emerald-400">Deployed Primary Classifier</td>
                    </tr>
                    <tr>
                      <td className={`py-3.5 px-4 font-semibold ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>Multinomial Naive Bayes</td>
                      <td className={`py-3.5 px-4 text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>&alpha;=1.0 (Laplace)</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{EVALUATION_METRICS.naiveBayes.accuracy}%</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{EVALUATION_METRICS.naiveBayes.precision}%</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{EVALUATION_METRICS.naiveBayes.recall}%</td>
                      <td className={`py-3.5 px-4 font-mono ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{EVALUATION_METRICS.naiveBayes.f1Score}%</td>
                      <td className={`py-3.5 px-4 text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Baseline Benchmark</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrices Side by Side */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Logistic Regression CM */}
              <div
                className={`border rounded-xl p-5 shadow-xs space-y-3 transition-colors ${
                  isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <h4 className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    Logistic Regression Confusion Matrix
                  </h4>
                  <span className="text-xs text-blue-500 dark:text-blue-400 font-mono">F1: 88.9%</span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400 font-mono block">
                      {EVALUATION_METRICS.logisticRegression.confusionMatrix.trueNegative}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>True Negative (TN)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Real correctly predicted Real</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-rose-600 dark:text-rose-400 font-mono block">
                      {EVALUATION_METRICS.logisticRegression.confusionMatrix.falsePositive}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>False Positive (FP)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Real incorrectly flagged Fake</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-rose-600 dark:text-rose-400 font-mono block">
                      {EVALUATION_METRICS.logisticRegression.confusionMatrix.falseNegative}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>False Negative (FN)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Fake incorrectly flagged Real</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400 font-mono block">
                      {EVALUATION_METRICS.logisticRegression.confusionMatrix.truePositive}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>True Positive (TP)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Fake correctly predicted Fake</span>
                  </div>
                </div>
              </div>

              {/* Naive Bayes CM */}
              <div
                className={`border rounded-xl p-5 shadow-xs space-y-3 transition-colors ${
                  isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <h4 className={`text-sm font-semibold ${isDark ? 'text-slate-300' : 'text-slate-800'}`}>
                    Naive Bayes Confusion Matrix
                  </h4>
                  <span className={`text-xs font-mono ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>F1: 84.2%</span>
                </div>
                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400 font-mono block">
                      {EVALUATION_METRICS.naiveBayes.confusionMatrix.trueNegative}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>True Negative (TN)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Real correctly predicted Real</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-rose-600 dark:text-rose-400 font-mono block">
                      {EVALUATION_METRICS.naiveBayes.confusionMatrix.falsePositive}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>False Positive (FP)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Real incorrectly flagged Fake</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-rose-600 dark:text-rose-400 font-mono block">
                      {EVALUATION_METRICS.naiveBayes.confusionMatrix.falseNegative}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>False Negative (FN)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Fake incorrectly flagged Real</span>
                  </div>
                  <div className={`border p-3.5 rounded-lg transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400 font-mono block">
                      {EVALUATION_METRICS.naiveBayes.confusionMatrix.truePositive}
                    </span>
                    <span className={`text-xs font-medium block ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>True Positive (TP)</span>
                    <span className={`text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Fake correctly predicted Fake</span>
                  </div>
                </div>
              </div>

            </div>

            {/* Formula Explanations for Viva Defense */}
            <div
              className={`border rounded-xl p-5 space-y-4 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <h4 className={`text-sm font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>
                Mathematical Metric Formulas (Viva Reference)
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
                <div className={`p-3.5 rounded-lg border transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block mb-1">Accuracy</span>
                  <p className={`mb-2 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>(TP + TN) / (TP + TN + FP + FN)</p>
                  <p className={`font-sans text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Overall ratio of correct classifications.</p>
                </div>
                <div className={`p-3.5 rounded-lg border transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block mb-1">Precision</span>
                  <p className={`mb-2 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>TP / (TP + FP)</p>
                  <p className={`font-sans text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Accuracy of positive claims (minimizes false alarms).</p>
                </div>
                <div className={`p-3.5 rounded-lg border transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block mb-1">Recall (Sensitivity)</span>
                  <p className={`mb-2 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>TP / (TP + FN)</p>
                  <p className={`font-sans text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Proportion of actual fake news successfully caught.</p>
                </div>
                <div className={`p-3.5 rounded-lg border transition-colors ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-emerald-600 dark:text-emerald-400 font-bold block mb-1">F1-Score</span>
                  <p className={`mb-2 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>2 × (P × R) / (P + R)</p>
                  <p className={`font-sans text-[11px] ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Harmonic mean balancing precision and recall.</p>
                </div>
              </div>
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB 3: DATASET EXPLORER
           ========================================================================= */}
        {activeTab === 'dataset' && (
          <div className="space-y-6">
            
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    <Database className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                    <span>Dataset Explorer (data/dataset.csv)</span>
                  </h2>
                  <p className={`text-sm mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                    Balanced dataset with verified real news from reputable agencies and sensational fake news articles.
                  </p>
                </div>
                <div className="flex items-center gap-2 text-xs font-mono">
                  <span
                    className={`px-2.5 py-1 rounded border ${
                      isDark
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                        : 'bg-emerald-50 text-emerald-700 border-emerald-300'
                    }`}
                  >
                    50 REAL (50%)
                  </span>
                  <span
                    className={`px-2.5 py-1 rounded border ${
                      isDark
                        ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                        : 'bg-rose-50 text-rose-700 border-rose-300'
                    }`}
                  >
                    50 FAKE (50%)
                  </span>
                </div>
              </div>
            </div>

            {/* Filter and Search Bar */}
            <div
              className={`border rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setDatasetFilter('ALL')}
                  className={`text-xs px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
                    datasetFilter === 'ALL'
                      ? 'bg-blue-600 text-white'
                      : isDark
                      ? 'bg-slate-800 text-slate-400 hover:text-white'
                      : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                  }`}
                >
                  All Articles ({DATASET.length})
                </button>
                <button
                  onClick={() => setDatasetFilter('REAL')}
                  className={`text-xs px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
                    datasetFilter === 'REAL'
                      ? 'bg-emerald-600 text-white'
                      : isDark
                      ? 'bg-slate-800 text-slate-400 hover:text-white'
                      : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Real News Only
                </button>
                <button
                  onClick={() => setDatasetFilter('FAKE')}
                  className={`text-xs px-3 py-1.5 rounded-md font-medium transition cursor-pointer ${
                    datasetFilter === 'FAKE'
                      ? 'bg-rose-600 text-white'
                      : isDark
                      ? 'bg-slate-800 text-slate-400 hover:text-white'
                      : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Fake News Only
                </button>
              </div>

              <div className="relative min-w-[260px]">
                <Search className={`h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 ${isDark ? 'text-slate-400' : 'text-slate-400'}`} />
                <input
                  type="text"
                  value={datasetSearch}
                  onChange={(e) => setDatasetSearch(e.target.value)}
                  placeholder="Search articles in dataset..."
                  className={`w-full border rounded-lg pl-8 pr-3 py-1.5 text-xs outline-none focus:border-blue-500 transition-colors ${
                    isDark
                      ? 'bg-slate-950 border-slate-800 text-slate-200'
                      : 'bg-slate-50 border-slate-300 text-slate-900'
                  }`}
                />
              </div>
            </div>

            {/* Dataset Cards List */}
            <div className="space-y-3">
              {filteredDataset.map((item) => (
                <div
                  key={item.id}
                  className={`border rounded-xl p-4 transition shadow-xs space-y-2 ${
                    isDark
                      ? 'bg-slate-900 border-slate-800 hover:border-slate-700'
                      : 'bg-white border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <span className={`text-[10px] font-mono block ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>Row #{item.id}</span>
                      <h4 className={`text-sm font-semibold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{item.title}</h4>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span
                        className={`text-xs px-2.5 py-0.5 rounded-full font-bold font-mono border ${
                          item.label === 'REAL'
                            ? isDark
                              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                              : 'bg-emerald-50 text-emerald-700 border-emerald-300'
                            : isDark
                            ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                            : 'bg-rose-50 text-rose-700 border-rose-300'
                        }`}
                      >
                        {item.label}
                      </span>
                      <button
                        onClick={() => {
                          setNewsInput(item.text);
                          setActiveTab('detector');
                          const res = predictNews(item.text, selectedModel);
                          setPrediction(res);
                        }}
                        className={`text-[11px] px-2 py-1 rounded transition cursor-pointer ${
                          isDark
                            ? 'bg-slate-800 hover:bg-slate-700 text-blue-400'
                            : 'bg-blue-50 hover:bg-blue-100 text-blue-700'
                        }`}
                      >
                        Test in Detector
                      </button>
                    </div>
                  </div>
                  <p className={`text-xs leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>{item.text}</p>
                  {item.source && (
                    <div className={`text-[11px] flex items-center gap-1 font-mono pt-1 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                      <span>Source Context:</span>
                      <span className={isDark ? 'text-slate-400' : 'text-slate-600'}>{item.source}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB 4: VIVA PREPARATION (25 QUESTIONS & DETAILED ANSWERS)
           ========================================================================= */}
        {activeTab === 'viva' && (
          <div className="space-y-6">
            
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                <GraduationCap className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                <span>College Viva Preparation Guide (25 Questions &amp; Answers)</span>
              </h2>
              <p className={`text-sm mt-1 max-w-3xl ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                Comprehensive, examiner-focused questions covering ML fundamentals, NLP feature extraction, mathematical derivations, evaluation metrics, and architectural decisions.
              </p>
            </div>

            {/* Filter and Search */}
            <div
              className={`border rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-3 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex flex-wrap gap-1.5">
                {vivaCategories.map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setVivaCategory(cat)}
                    className={`text-xs px-2.5 py-1 rounded-md font-medium transition cursor-pointer ${
                      vivaCategory === cat
                        ? 'bg-blue-600 text-white'
                        : isDark
                        ? 'bg-slate-800 text-slate-400 hover:text-white'
                        : 'bg-slate-100 text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>

              <div className="relative min-w-[240px]">
                <Search className={`h-3.5 w-3.5 absolute left-3 top-1/2 -translate-y-1/2 ${isDark ? 'text-slate-400' : 'text-slate-400'}`} />
                <input
                  type="text"
                  value={vivaSearch}
                  onChange={(e) => setVivaSearch(e.target.value)}
                  placeholder="Search questions or keywords..."
                  className={`w-full border rounded-lg pl-8 pr-3 py-1.5 text-xs outline-none focus:border-blue-500 transition-colors ${
                    isDark
                      ? 'bg-slate-950 border-slate-800 text-slate-200'
                      : 'bg-slate-50 border-slate-300 text-slate-900'
                  }`}
                />
              </div>
            </div>

            {/* Questions Accordion */}
            <div className="space-y-3">
              {filteredViva.map((q) => {
                const isOpen = expandedVivaId === q.id;
                return (
                  <div
                    key={q.id}
                    className={`border rounded-xl transition ${
                      isOpen
                        ? isDark
                          ? 'bg-slate-900 border-blue-500/40 shadow-xs'
                          : 'bg-white border-blue-400 shadow-xs'
                        : isDark
                        ? 'bg-slate-900 border-slate-800 hover:border-slate-700'
                        : 'bg-white border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <button
                      onClick={() => setExpandedVivaId(isOpen ? null : q.id)}
                      className="w-full p-4 text-left flex items-start justify-between gap-3 cursor-pointer"
                    >
                      <div className="flex items-start gap-3">
                        <span
                          className={`text-xs font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${
                            isDark
                              ? 'text-blue-400 bg-blue-500/10 border-blue-500/20'
                              : 'text-blue-700 bg-blue-50 border-blue-200'
                          }`}
                        >
                          Q{q.id}
                        </span>
                        <div>
                          <span className={`text-[11px] font-mono block mb-0.5 ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
                            {q.category}
                          </span>
                          <h4 className={`text-sm font-semibold ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{q.question}</h4>
                        </div>
                      </div>
                      <span className={`mt-1 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                      </span>
                    </button>

                    {isOpen && (
                      <div
                        className={`px-4 pb-4 pt-1 border-t space-y-3 ${
                          isDark ? 'border-slate-800/80' : 'border-slate-100'
                        }`}
                      >
                        <div
                          className={`p-3 rounded-lg border transition-colors ${
                            isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                          }`}
                        >
                          <span
                            className={`text-[11px] uppercase tracking-wider font-bold block mb-1 ${
                              isDark ? 'text-slate-400' : 'text-slate-600'
                            }`}
                          >
                            Model Answer (Explain in Viva as):
                          </span>
                          <p className={`text-sm leading-relaxed ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{q.answer}</p>
                          {q.formula && (
                            <div
                              className={`mt-2.5 p-2 rounded font-mono text-xs border ${
                                isDark
                                  ? 'bg-slate-900 border-slate-800 text-blue-300'
                                  : 'bg-blue-50/60 border-blue-200 text-blue-800'
                              }`}
                            >
                              {q.formula}
                            </div>
                          )}
                        </div>

                        <div
                          className={`flex items-center gap-2 text-xs p-2.5 rounded-lg border ${
                            isDark
                              ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                              : 'text-emerald-800 bg-emerald-50 border-emerald-200'
                          }`}
                        >
                          <Check className="h-4 w-4 shrink-0" />
                          <span><strong>Key Examiner Takeaway:</strong> {q.keyTakeaway}</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB 5: COLLEGE PROJECT REPORT & DOCUMENTATION
           ========================================================================= */}
        {activeTab === 'docs' && (
          <div className="space-y-6">
            
            <div
              className={`border rounded-xl p-5 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-4 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div>
                <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                  <BookOpen className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                  <span>College Mini Project Report &amp; Documentation</span>
                </h2>
                <p className={`text-sm mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                  Complete structured academic project documentation ready for presentation slides, viva report, and project synopsis.
                </p>
              </div>
              <button
                onClick={() => {
                  const allDocs = PROJECT_DOCUMENTATION.map(s => `${s.title}\n\n${s.content}\n\n`).join('\n---\n\n');
                  copyToClipboard(allDocs);
                }}
                className="text-xs bg-blue-600 hover:bg-blue-500 text-white font-semibold px-4 py-2 rounded-lg transition flex items-center gap-1.5 shrink-0 cursor-pointer shadow-xs"
              >
                {copiedCode ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                <span>{copiedCode ? 'Report Copied!' : 'Copy Full Academic Report'}</span>
              </button>
            </div>

            <div className="space-y-4">
              {PROJECT_DOCUMENTATION.map((sec) => (
                <div
                  key={sec.id}
                  className={`border rounded-xl p-5 space-y-2 transition-colors ${
                    isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
                  }`}
                >
                  <h3 className={`text-base font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    <span className="h-2 w-2 rounded-full bg-blue-500"></span>
                    <span>{sec.title}</span>
                  </h3>
                  <div
                    className={`text-xs leading-relaxed whitespace-pre-line pl-4 border-l-2 pt-1 ${
                      isDark ? 'text-slate-300 border-slate-800' : 'text-slate-700 border-slate-200'
                    }`}
                  >
                    {sec.content}
                  </div>
                </div>
              ))}
            </div>

          </div>
        )}

        {/* =========================================================================
            TAB 6: PYTHON & FLASK CODE EXPLORER
           ========================================================================= */}
        {activeTab === 'code' && (
          <div className="space-y-6">
            
            <div
              className={`border rounded-xl p-5 shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h2 className={`text-xl font-bold tracking-tight flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    <FileCode2 className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                    <span>Python &amp; Flask Project Codebase</span>
                  </h2>
                  <p className={`text-sm mt-1 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                    All standalone backend files created in <code className="text-blue-500 dark:text-blue-400 font-mono">fake-news-detection/</code>. Ready to run locally with VS Code or deploy to Render.
                  </p>
                </div>
              </div>
            </div>

            {/* File Selector & Code Viewer */}
            <div
              className={`border rounded-xl overflow-hidden shadow-xs transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              
              {/* File Tabs */}
              <div
                className={`border-b p-2 flex items-center justify-between gap-2 overflow-x-auto ${
                  isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
                }`}
              >
                <div className="flex items-center gap-1">
                  {PYTHON_PROJECT_FILES.map((f, idx) => (
                    <button
                      key={f.name}
                      onClick={() => setSelectedFileIndex(idx)}
                      className={`text-xs px-3 py-1.5 rounded-md font-mono transition flex items-center gap-1.5 cursor-pointer ${
                        selectedFileIndex === idx
                          ? isDark
                            ? 'bg-slate-800 text-white font-semibold border border-slate-700'
                            : 'bg-white text-slate-900 font-semibold border border-slate-300 shadow-xs'
                          : isDark
                          ? 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                          : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200'
                      }`}
                    >
                      <span>{f.name}</span>
                    </button>
                  ))}
                </div>
                <button
                  onClick={() => copyToClipboard(PYTHON_PROJECT_FILES[selectedFileIndex].content)}
                  className={`text-xs px-3 py-1 rounded transition flex items-center gap-1 font-mono shrink-0 cursor-pointer ${
                    isDark
                      ? 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                      : 'bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 shadow-xs'
                  }`}
                >
                  {copiedCode ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
                  <span>{copiedCode ? 'Copied' : 'Copy Code'}</span>
                </button>
              </div>

              {/* File Info */}
              <div
                className={`px-4 py-2.5 border-b flex items-center justify-between text-xs ${
                  isDark
                    ? 'bg-slate-900/60 border-slate-800 text-slate-400'
                    : 'bg-slate-50 border-slate-200 text-slate-600'
                }`}
              >
                <span className="font-mono text-blue-500 dark:text-blue-400">{PYTHON_PROJECT_FILES[selectedFileIndex].path}</span>
                <span>{PYTHON_PROJECT_FILES[selectedFileIndex].description}</span>
              </div>

              {/* Code Pre */}
              <div className="p-4 bg-slate-950 overflow-x-auto max-h-[500px]">
                <pre className="text-xs font-mono text-slate-200 leading-relaxed">
                  <code>{PYTHON_PROJECT_FILES[selectedFileIndex].content}</code>
                </pre>
              </div>
            </div>

            {/* Run Guide on Windows / VS Code */}
            <div
              className={`border rounded-xl p-5 space-y-4 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <h4 className={`text-sm font-semibold flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                <Terminal className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
                <span>How to Run Locally on Windows using VS Code</span>
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs font-mono">
                <div className={`p-3 rounded-lg border space-y-1 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block">1. Setup &amp; Install</span>
                  <p className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Open terminal in VS Code:</p>
                  <code className="text-emerald-600 dark:text-emerald-400 block bg-slate-900 dark:bg-slate-900 text-slate-100 p-1.5 rounded">
                    cd fake-news-detection<br/>
                    pip install -r requirements.txt
                  </code>
                </div>
                <div className={`p-3 rounded-lg border space-y-1 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block">2. Train Models</span>
                  <p className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Run training pipeline:</p>
                  <code className="text-emerald-600 dark:text-emerald-400 block bg-slate-900 dark:bg-slate-900 text-slate-100 p-1.5 rounded">
                    python train_model.py
                  </code>
                  <p className={`text-[11px] font-sans ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Generates model.pkl &amp; vectorizer.pkl</p>
                </div>
                <div className={`p-3 rounded-lg border space-y-1 ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className="text-blue-500 dark:text-blue-400 font-bold block">3. Launch Flask Server</span>
                  <p className={`font-sans ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>Start web interface:</p>
                  <code className="text-emerald-600 dark:text-emerald-400 block bg-slate-900 dark:bg-slate-900 text-slate-100 p-1.5 rounded">
                    python app.py
                  </code>
                  <p className={`text-[11px] font-sans ${isDark ? 'text-slate-500' : 'text-slate-500'}`}>Visit http://127.0.0.1:5000</p>
                </div>
              </div>
            </div>

            {/* Free Deployment Guide */}
            <div
              className={`border rounded-xl p-5 space-y-3 transition-colors ${
                isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
              }`}
            >
              <h4 className={`text-sm font-semibold flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                <CloudUpload className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                <span>Step-by-Step Free Cloud Deployment (Render.com)</span>
              </h4>
              <ol className={`list-decimal list-inside space-y-2 text-xs leading-relaxed ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                <li>Create a free account on <a href="https://render.com" target="_blank" rel="noreferrer" className="text-blue-500 dark:text-blue-400 underline">Render.com</a> and push this folder to your GitHub repo.</li>
                <li>In Render Dashboard, click <strong>New +</strong> &rarr; <strong>Web Service</strong> and select your GitHub repo.</li>
                <li>Set <strong>Environment</strong> to <code className={`font-mono px-1 py-0.5 rounded ${isDark ? 'bg-slate-950 text-blue-300' : 'bg-slate-100 text-blue-700'}`}>Python 3</code>.</li>
                <li>Set <strong>Build Command</strong> to: <code className={`font-mono px-1.5 py-0.5 rounded ${isDark ? 'bg-slate-950 text-emerald-300' : 'bg-slate-100 text-emerald-700'}`}>pip install -r requirements.txt && python train_model.py</code></li>
                <li>Set <strong>Start Command</strong> to: <code className={`font-mono px-1.5 py-0.5 rounded ${isDark ? 'bg-slate-950 text-emerald-300' : 'bg-slate-100 text-emerald-700'}`}>gunicorn app:app</code></li>
                <li>Select <strong>Free Instance Type</strong> and click <strong>Deploy</strong>. Your project is live in ~2 minutes with free HTTPS!</li>
              </ol>
            </div>

          </div>
        )}

      </main>

      {/* Footer */}
      <footer
        className={`border-t py-4 text-center text-xs transition-colors ${
          isDark
            ? 'border-slate-800 bg-slate-900/60 text-slate-500'
            : 'border-slate-200 bg-white text-slate-500'
        }`}
      >
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>Fake News Detection in Social Media Using Machine Learning &middot; CSE Mini Project</p>
          <p className={`font-mono ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
            Scikit-Learn &middot; TF-IDF &middot; Logistic Regression &middot; Naive Bayes &middot; Flask
          </p>
        </div>
      </footer>

    </div>
  );
}

