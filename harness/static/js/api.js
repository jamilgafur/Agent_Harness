/**
 * API Client Module
 * Handles all HTTP and WebSocket communication with the backend
 */

// Configuration
const API_BASE = ''; // Same origin
const WS_URL = `ws://${window.location.host}/ws`;

// Retry configuration
const RETRY_CONFIG = {
  maxAttempts: 3,
  baseDelay: 1000,
  maxDelay: 10000,
  backoffMultiplier: 2
};

// Cache configuration
const CACHE_CONFIG = {
  tools: { ttl: 300000, key: 'api_tools_cache' }, // 5 minutes
  files: { ttl: 60000, key: 'api_files_cache' },  // 1 minute
  policy: { ttl: 600000, key: 'api_policy_cache' } // 10 minutes
};

// State
let wsConnection = null;
let wsListeners = new Set();
let retryTimeout = null;

/**
 * Escapes HTML special characters
 */
export function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"]/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;'
  }[c]));
}

/**
 * Simple cache manager
 */
const cache = {
  get: (key) => {
    try {
      const item = localStorage.getItem(key);
      if (!item) return null;
      const { value, expires } = JSON.parse(item);
      if (expires && expires < Date.now()) {
        localStorage.removeItem(key);
        return null;
      }
      return value;
    } catch {
      return null;
    }
  },
  set: (key, value, ttl) => {
    try {
      const expires = ttl ? Date.now() + ttl : null;
      localStorage.setItem(key, JSON.stringify({ value, expires }));
    } catch {
      // Ignore storage errors
    }
  },
  clear: (key) => {
    try {
      localStorage.removeItem(key);
    } catch {
      // Ignore
    }
  }
};

/**
 * Sleep utility
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * HTTP request with retry logic
 */
export async function apiRequest(method, path, body = null, { retry = true, cache: cacheKey = null } = {}) {
  // Check cache first for GET requests
  if (cacheKey && method === 'GET') {
    const cached = cache.get(cacheKey);
    if (cached) return cached;
  }

  let lastError;
  for (let attempt = 0; attempt < RETRY_CONFIG.maxAttempts; attempt++) {
    try {
      const options = {
        method,
        headers: { 'content-type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined
      };

      const response = await fetch(API_BASE + path, options);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const error = new Error(errorData.detail || errorData.msg || `HTTP ${response.status}`);
        error.status = response.status;
        error.data = errorData;
        throw error;
      }

      const data = await response.json().catch(() => ({}));
      
      // Cache successful GET responses
      if (cacheKey && method === 'GET') {
        cache.set(cacheKey, data, CACHE_CONFIG[cacheKey]?.ttl);
      }
      
      return data;
    } catch (error) {
      lastError = error;
      
      if (!retry || attempt >= RETRY_CONFIG.maxAttempts - 1) break;
      
      const delay = Math.min(
        RETRY_CONFIG.baseDelay * Math.pow(RETRY_CONFIG.backoffMultiplier, attempt),
        RETRY_CONFIG.maxDelay
      );
      await sleep(delay);
    }
  }
  
  throw lastError;
}

/**
 * Clear cached data
 */
export function clearApiCache() {
  Object.values(CACHE_CONFIG).forEach(config => cache.clear(config.key));
}

/**
 * WebSocket connection manager
 */
export function connectWebSocket() {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    return wsConnection;
  }
  
  // Close existing connection
  if (wsConnection) {
    wsConnection.close();
  }
  
  const ws = new WebSocket(WS_URL);
  wsConnection = ws;
  
  ws.onopen = () => {
    console.log('WebSocket connected');
    // Notify listeners
    wsListeners.forEach(fn => fn({ type: 'connected' }));
  };
  
  ws.onmessage = (event) => {
    try {
      const message = JSON.parse(event.data);
      wsListeners.forEach(fn => fn(message));
    } catch {
      // Ignore parse errors
    }
  };
  
  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting...');
    wsListeners.forEach(fn => fn({ type: 'disconnected' }));
    // Auto-reconnect
    if (retryTimeout) clearTimeout(retryTimeout);
    retryTimeout = setTimeout(() => connectWebSocket(), 3000);
  };
  
  ws.onerror = () => {
    // Handled by onclose
  };
  
  return ws;
}

/**
 * Disconnect WebSocket
 */
export function disconnectWebSocket() {
  if (retryTimeout) {
    clearTimeout(retryTimeout);
    retryTimeout = null;
  }
  if (wsConnection) {
    wsConnection.close();
    wsConnection = null;
  }
}

/**
 * Add WebSocket message listener
 */
export function addWebSocketListener(fn) {
  wsListeners.add(fn);
  return () => wsListeners.delete(fn);
}

/**
 * Send WebSocket message
 */
export function sendWebSocket(message) {
  if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
    wsConnection.send(JSON.stringify(message));
    return true;
  }
  return false;
}

/**
 * Health check
 */
export async function checkHealth() {
  return apiRequest('GET', '/health', null, { cache: 'health' });
}

/**
 * Get tools
 */
export async function getTools() {
  return apiRequest('GET', '/tools', null, { cache: 'tools' });
}

/**
 * Sync tools
 */
export async function syncTools() {
  return apiRequest('POST', '/tools/sync');
}

/**
 * Get policy
 */
export async function getPolicy() {
  return apiRequest('GET', '/policy', null, { cache: 'policy' });
}

/**
 * Update policy
 */
export async function updatePolicy(policy) {
  return apiRequest('POST', '/policy', policy);
}

/**
 * Get runs
 */
export async function getRuns() {
  return apiRequest('GET', '/runs');
}

/**
 * Get single run
 */
export async function getRun(runId) {
  return apiRequest('GET', `/runs/${runId}`);
}

/**
 * Create run
 */
export async function createRun(data) {
  return apiRequest('POST', '/runs', data);
}

/**
 * Delete run
 */
export async function deleteRun(runId, force = false) {
  return apiRequest('DELETE', `/runs/${runId}${force ? '?force=1' : ''}`);
}

/**
 * Get run events
 */
export async function getRunEvents(runId, afterId = 0) {
  return apiRequest('GET', `/runs/${runId}/events?after_id=${afterId}`);
}

/**
 * Get run knowledge
 */
export async function getRunKnowledge(runId) {
  return apiRequest('GET', `/runs/${runId}/knowledge`);
}

/**
 * Get run snapshot
 */
export async function getRunSnapshot(runId) {
  return apiRequest('GET', `/runs/${runId}/snapshot`);
}

/**
 * Start execution
 */
export async function startRun(runId) {
  return apiRequest('POST', `/runs/${runId}/go`);
}

/**
 * Update run config
 */
export async function updateRunConfig(runId, config) {
  return apiRequest('PUT', `/runs/${runId}/config`, config);
}

/**
 * Update constraints
 */
export async function updateConstraints(runId, constraints) {
  return apiRequest('PUT', `/runs/${runId}/constraints`, { constraints });
}

/**
 * Add node to run
 */
export async function addNode(runId, node) {
  return apiRequest('POST', `/runs/${runId}/nodes`, node);
}

/**
 * Update node
 */
export async function updateNode(runId, nodeId, data) {
  return apiRequest('PATCH', `/runs/${runId}/nodes/${nodeId}`, data);
}

/**
 * Replan node
 */
export async function replanNode(runId, nodeId) {
  return apiRequest('POST', `/runs/${runId}/nodes/${nodeId}/replan`);
}

/**
 * Create skill
 */
export async function createSkill(skill) {
  return apiRequest('POST', '/skills', skill);
}

/**
 * Get workspace files
 */
export async function getWorkspaceFiles(path = '') {
  return apiRequest('GET', `/workspace/files?path=${encodeURIComponent(path)}`);
}

/**
 * Get file content
 */
export async function getFileContent(path) {
  return apiRequest('GET', `/workspace/file?path=${encodeURIComponent(path)}`);
}

/**
 * Get agents
 */
export async function getAgents() {
  return apiRequest('GET', '/agents');
}

/**
 * Send heartbeat
 */
export async function sendHeartbeat(data) {
  return apiRequest('POST', '/heartbeat', data);
}

/**
 * Get approvals
 */
export async function getApprovals(status = null) {
  const path = status ? `/approvals?status=${status}` : '/approvals';
  return apiRequest('GET', path);
}

/**
 * Handle approval
 */
export async function handleApproval(approvalId, decision, args = null) {
  const body = { decision };
  if (decision === 'approve' && args) {
    body.args = args;
  }
  return apiRequest('POST', `/approvals/${approvalId}`, body);
}

/**
 * Get metrics
 */
export async function getMetrics() {
  return apiRequest('GET', '/metrics');
}

// Initialize WebSocket on module load
connectWebSocket();

export default {
  apiRequest,
  clearApiCache,
  connectWebSocket,
  disconnectWebSocket,
  addWebSocketListener,
  sendWebSocket
};
