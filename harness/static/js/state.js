/**
 * State Management Module
 * Centralized state for the application
 */

import { apiRequest, addWebSocketListener, getRuns, getRun, getRunEvents, getTools, getAgents, getMetrics, getApprovals, getPolicy } from './api.js';
import { escapeHtml } from './api.js';

// ============================================================================
// Design Token Colors (CSS variables for JS access)
// ============================================================================
export const COLORS = {
  bg: 'var(--bg)',
  panel: 'var(--panel)',
  panel2: 'var(--panel2)',
  inset: 'var(--inset)',
  line: 'var(--line)',
  line2: 'var(--line2)',
  fg: 'var(--fg)',
  dim: 'var(--dim)',
  mut: 'var(--mut)',
  acc: 'var(--acc)',
  ok: 'var(--ok)',
  warn: 'var(--warn)',
  wait: 'var(--wait)',
  err: 'var(--err)',
  violet: 'var(--violet)'
};

// State colors for nodes
export const STATE_COLORS = {
  PENDING: '#8b949e',
  RUNNING: '#5a9bff',
  DONE: '#3fb950',
  WAITING: '#d29922',
  REPAIRING: '#e3b341',
  FAILED: '#f85149',
  BLOCKED: '#f85149',
  ESCALATED: '#bc4cf0'
};

// ============================================================================
// Application State
// ============================================================================

// Create a reactive state store
class ReactiveStore {
  constructor(initialState = {}) {
    this.state = { ...initialState };
    this.subscribers = new Set();
  }

  getState() {
    return this.state;
  }

  setState(newState) {
    this.state = { ...this.state, ...newState };
    this.notify();
  }

  updateState(updater) {
    this.state = { ...this.state, ...updater(this.state) };
    this.notify();
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  notify() {
    this.subscribers.forEach(callback => callback(this.state));
  }
}

// Main application state
export const appState = new ReactiveStore({
  // Connection status
  connected: false,
  ollamaOk: false,
  sandboxOk: false,
  
  // Current view
  currentRunId: null,
  currentTab: 'intent', // 'intent' or 'template'
  
  // Runs
  runs: {},
  allRuns: [],
  
  // Tools
  tools: {},
  autoApprove: [],
  
  // Agents
  agents: {},
  
  // Workspace
  workspaceFiles: [],
  currentWorkspacePath: '',
  
  // UI state
  showDevPanel: false,
  selectedNodeId: null,
  showModal: false,
  modalContent: null,
  
  // Settings (loaded from localStorage)
  settings: {
    theme: 'dark',
    fontSize: 'medium',
    animations: true,
    showAdvanced: false
  },
  
  // Metrics
  metrics: null,
  
  // Notifications
  notifications: [],
  
  // Layout
  view: { x: 40, y: 30, k: 1 },
  
  // Forest (node positions per run)
  forest: {},
  
  // Seen nodes and edges (for FLIP animations)
  seen: new Set(),
  posPrev: {},
  anim: {},
  agPos: {},
  
  // Last output
  lastOutput: '',
  
  // Plan bar
  planBarKey: ''
});

// ============================================================================
// Actions
// ============================================================================

/**
 * Initialize the application state
 */
export async function initState() {
  try {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('harness_settings');
    if (savedSettings) {
      appState.setState({ 
        settings: { ...appState.getState().settings, ...JSON.parse(savedSettings) } 
      });
    }

    // Load initial data
    await loadInitialData();
    
    // Setup WebSocket listener
    setupWebSocketListener();
    
    console.log('State initialized');
  } catch (error) {
    console.error('Failed to initialize state:', error);
    addNotification({ type: 'error', message: 'Failed to load initial data' });
  }
}

/**
 * Load initial data
 */
async function loadInitialData() {
  try {
    const [runs, tools, agents, health, metrics] = await Promise.all([
      getRuns(),
      getTools(),
      getAgents(),
      apiRequest('GET', '/health'),
      getMetrics()
    ]);

    appState.setState({
      allRuns: runs,
      tools: tools.tools || {},
      autoApprove: tools.auto_approve || [],
      agents: agents.agents || {},
      connected: true,
      ollamaOk: health.ollama || false,
      metrics: metrics
    });

    // Load first run if available
    if (runs.length > 0) {
      await selectRun(runs[0].id);
    }

    // Load workspace files
    await loadWorkspaceFiles();
  } catch (error) {
    console.error('Error loading initial data:', error);
    appState.setState({ connected: false });
  }
}

/**
 * Setup WebSocket listener
 */
function setupWebSocketListener() {
  addWebSocketListener((message) => {
    switch (message.type) {
      case 'connected':
        appState.setState({ connected: true });
        break;
        
      case 'disconnected':
        appState.setState({ connected: false });
        break;
        
      case 'state':
        handleStateUpdate(message);
        break;
        
      case 'agents':
        appState.setState({ agents: message.agents || {} });
        break;
        
      case 'run':
        handleRunUpdate(message);
        break;
        
      case 'run_deleted':
        handleRunDeleted(message.run_id);
        break;
        
      case 'approval':
        // Approvals are handled separately
        break;
        
      case 'policy':
        appState.setState({ policy: message.policy });
        break;
        
      case 'log':
        // Log messages are handled by the UI
        break;
    }
  });
}

/**
 * Handle state update from WebSocket
 */
function handleStateUpdate(message) {
  const { run_id, nodes, stats, active, phase, constraints, prompts, params, plan_ready, warnings, pending } = message;
  
  appState.updateState((state) => {
    const newRuns = { ...state.runs };
    
    if (!newRuns[run_id]) {
      newRuns[run_id] = {};
    }
    
    newRuns[run_id] = {
      ...newRuns[run_id],
      id: run_id,
      nodes: nodes || {},
      stats: stats || {},
      active: active || false,
      phase: phase || 'executing',
      constraints: constraints || [],
      prompts: prompts || {},
      params: params || {},
      plan_ready: plan_ready || false,
      warnings: warnings || [],
      pending: new Set(pending || [])
    };
    
    // If this is the current run, update forest
    if (state.currentRunId === run_id && nodes) {
      const layout = layoutRun(newRuns[run_id]);
      return { 
        ...state, 
        runs: newRuns,
        forest: { ...state.forest, [run_id]: layout }
      };
    }
    
    return { ...state, runs: newRuns };
  });
  
  // If this is the current run, reload its data
  const currentState = appState.getState();
  if (message.run_id === currentState.currentRunId) {
    // Will be handled by the UI
  }
}

/**
 * Handle run update from WebSocket
 */
function handleRunUpdate(message) {
  // Refresh runs list
  getRuns().then(runs => {
    appState.setState({ allRuns: runs });
  });
}

/**
 * Handle run deleted
 */
function handleRunDeleted(runId) {
  appState.updateState((state) => {
    const newRuns = { ...state.runs };
    delete newRuns[runId];
    
    const newForest = { ...state.forest };
    delete newForest[runId];
    
    let newCurrentRunId = state.currentRunId;
    if (newCurrentRunId === runId) {
      newCurrentRunId = state.allRuns.length > 0 ? state.allRuns[0].id : null;
    }
    
    return { 
      ...state, 
      runs: newRuns,
      forest: newForest,
      currentRunId: newCurrentRunId
    };
  });
  
  // Refresh runs list
  getRuns().then(runs => {
    appState.setState({ allRuns: runs });
  });
}

// ============================================================================
// Run Management
// ============================================================================

/**
 * Select a run
 */
export async function selectRun(runId) {
  if (!runId) return;
  
  try {
    const run = appState.getState().runs[runId] || await getRun(runId);
    const events = await getRunEvents(runId);
    
    appState.setState({
      currentRunId: runId,
      selectedNodeId: null
    });
    
    // Update runs cache
    appState.updateState((state) => {
      const newRuns = { ...state.runs };
      newRuns[runId] = run;
      return { ...state, runs: newRuns };
    });
    
    // Calculate layout
    const layout = layoutRun(run);
    appState.updateState((state) => ({
      ...state,
      forest: { ...state.forest, [runId]: layout }
    }));
    
    return run;
  } catch (error) {
    console.error('Error selecting run:', error);
    addNotification({ type: 'error', message: `Failed to load run: ${escapeHtml(error.message)}` });
  }
}

/**
 * Create a new run
 */
export async function createNewRun(data) {
  try {
    const result = await apiRequest('POST', '/runs', data);
    const newRunId = result.run_id;
    
    // Refresh runs list
    const runs = await getRuns();
    appState.setState({ allRuns: runs });
    
    // Select the new run
    await selectRun(newRunId);
    
    addNotification({ type: 'success', message: 'Run created successfully' });
    return newRunId;
  } catch (error) {
    console.error('Error creating run:', error);
    addNotification({ type: 'error', message: `Failed to create run: ${escapeHtml(error.message || error)}` });
    throw error;
  }
}

/**
 * Delete a run
 */
export async function deleteCurrentRun(runId, force = false) {
  try {
    await deleteRun(runId, force);
    
    // Remove from state
    appState.updateState((state) => {
      const newRuns = { ...state.runs };
      delete newRuns[runId];
      
      const newForest = { ...state.forest };
      delete newForest[runId];
      
      const allRuns = state.allRuns.filter(r => r.id !== runId);
      const newCurrentRunId = allRuns.length > 0 ? allRuns[0].id : null;
      
      return { 
        ...state, 
        runs: newRuns,
        forest: newForest,
        allRuns,
        currentRunId: newCurrentRunId
      };
    });
    
    addNotification({ type: 'success', message: 'Run deleted' });
  } catch (error) {
    console.error('Error deleting run:', error);
    addNotification({ type: 'error', message: `Failed to delete run: ${escapeHtml(error.message || error)}` });
    throw error;
  }
}

// ============================================================================
// Workspace Files
// ============================================================================

/**
 * Load workspace files
 */
export async function loadWorkspaceFiles(path = '') {
  try {
    const data = await apiRequest('GET', `/workspace/files?path=${encodeURIComponent(path)}`);
    appState.setState({
      workspaceFiles: data.entries || [],
      currentWorkspacePath: path
    });
    return data.entries || [];
  } catch (error) {
    console.error('Error loading workspace files:', error);
    appState.setState({ workspaceFiles: [] });
    addNotification({ type: 'error', message: 'Failed to load workspace files' });
    return [];
  }
}

/**
 * Get file content
 */
export async function getFile(path) {
  return apiRequest('GET', `/workspace/file?path=${encodeURIComponent(path)}`);
}

// ============================================================================
// Layout Algorithm
// ============================================================================

const W = 228, H = 64, CW = 320, RH = 88;

/**
 * Calculate layout for a run
 */
export function layoutRun(run) {
  const nodes = Object.values(run.nodes || {});
  if (!nodes.length) return { pos: {}, h: 0 };
  
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const primary = (n) => {
    if (n.parent && byId[n.parent]) return n.parent;
    const d = n.deps || [];
    return d.length && byId[d[0]] ? d[0] : null;
  };
  
  const kids = {};
  nodes.forEach(n => {
    const p = primary(n);
    if (p) (kids[p] = kids[p] || []).push(n.id);
  });
  
  Object.values(kids).forEach(k => k.sort());
  const roots = nodes.filter(n => !primary(n)).map(n => n.id).sort();
  
  const pos = {};
  let nextY = 0;
  
  const place = (id, d) => {
    const c = kids[id] || [];
    let y;
    if (c.length) {
      const ys = c.map(ch => place(ch, d + 1));
      y = (ys[0] + ys[ys.length - 1]) / 2;
    } else {
      y = nextY * RH;
      nextY++;
    }
    pos[id] = { x: d * CW, y };
    return y;
  };
  
  roots.forEach(rid => { place(rid, 0); nextY += 0.6; });
  nodes.forEach(n => {
    if (!pos[n.id]) {
      pos[n.id] = { x: (n.depth || 0) * CW, y: nextY * RH };
      nextY++;
    }
  });
  
  // Ensure nodes don't sit left of their inputs
  for (let guard = 0; guard < 8; guard++) {
    let moved = false;
    for (const n of nodes) {
      const p = pos[n.id];
      if (!p) continue;
      for (const d of n.deps || []) {
        const dp = pos[d];
        if (dp && dp.x + CW > p.x) {
          p.x = dp.x + CW;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
  
  return { pos, h: nextY * RH };
}

// ============================================================================
// Notifications
// ============================================================================

/**
 * Add a notification
 */
export function addNotification(notification) {
  const id = Date.now().toString();
  const newNotification = {
    id,
    type: 'info',
    message: '',
    timestamp: Date.now(),
    ...notification
  };
  
  appState.updateState((state) => ({
    ...state,
    notifications: [...state.notifications, newNotification]
  }));
  
  // Auto-remove after 5 seconds (except for errors which stay longer)
  if (newNotification.type !== 'error') {
    setTimeout(() => {
      removeNotification(id);
    }, 5000);
  }
  
  return id;
}

/**
 * Remove a notification
 */
export function removeNotification(id) {
  appState.updateState((state) => ({
    ...state,
    notifications: state.notifications.filter(n => n.id !== id)
  }));
}

// ============================================================================
// Settings
// ============================================================================

/**
 * Update settings
 */
export function updateSettings(newSettings) {
  appState.updateState((state) => {
    const updatedSettings = { ...state.settings, ...newSettings };
    
    // Persist to localStorage
    try {
      localStorage.setItem('harness_settings', JSON.stringify(updatedSettings));
    } catch {
      // Ignore storage errors
    }
    
    return { ...state, settings: updatedSettings };
  });
}

/**
 * Reset settings to defaults
 */
export function resetSettings() {
  appState.updateState((state) => {
    const defaultSettings = state.settings;
    
    try {
      localStorage.removeItem('harness_settings');
    } catch {
      // Ignore
    }
    
    return { ...state, settings: defaultSettings };
  });
}

// ============================================================================
// View Management
// ============================================================================

/**
 * Update view state
 */
export function updateView(newView) {
  appState.updateState((state) => ({
    ...state,
    view: { ...state.view, ...newView }
  }));
}

/**
 * Reset view to fit all nodes
 */
export function fitAll(runId) {
  const state = appState.getState();
  const forest = state.forest[runId];
  
  if (!forest) return;
  
  const xs = [];
  const ys = [];
  
  for (const p of Object.values(forest.pos)) {
    xs.push(p.x);
    ys.push(p.y);
  }
  
  if (!xs.length) return;
  
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs) + W;
  const y0 = Math.min(...ys) - 30;
  const y1 = Math.max(...ys) + H;
  
  const vp = document.getElementById('viewport');
  if (!vp) return;
  
  const k = Math.min(1.3, 
    vp.clientWidth / (x1 - x0 + 60),
    vp.clientHeight / (y1 - y0 + 40));
  
  const x = vp.clientWidth / 2 - (x0 + x1) / 2 * k;
  const y = vp.clientHeight / 2 - (y0 + y1) / 2 * k;
  
  updateView({ x, y, k });
}

/**
 * Select a node
 */
export function selectNode(nodeId) {
  appState.setState({
    selectedNodeId: nodeId
  });
}

/**
 * Show modal
 */
export function showModal(content, options = {}) {
  appState.setState({
    showModal: true,
    modalContent: { content, ...options }
  });
}

/**
 * Hide modal
 */
export function hideModal() {
  appState.setState({
    showModal: false,
    modalContent: null
  });
}

/**
 * Toggle dev panel
 */
export function toggleDevPanel() {
  appState.updateState((state) => ({
    ...state,
    showDevPanel: !state.showDevPanel
  }));
}

export default appState;
