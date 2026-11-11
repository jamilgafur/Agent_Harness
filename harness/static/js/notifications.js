/**
 * Notifications Module
 * Handles toast notifications and alerts
 */

import { appState, addNotification, removeNotification } from './state.js';

// ============================================================================
// Notification Types
// ============================================================================

export const NOTIFICATION_TYPES = {
  INFO: 'info',
  SUCCESS: 'success',
  WARNING: 'warning',
  ERROR: 'error'
};

// ============================================================================
// Notification Component
// ============================================================================

class NotificationManager {
  constructor() {
    this.container = null;
    this.init();
  }

  init() {
    // Create container
    this.container = document.createElement('div');
    this.container.id = 'notification-container';
    this.container.style.cssText = `
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 1000;
      display: flex;
      flex-direction: column;
      gap: 8px;
      pointer-events: none;
    `;
    document.body.appendChild(this.container);

    // Subscribe to state changes
    this.unsubscribe = appState.subscribe((state) => {
      this.renderNotifications(state.notifications);
    });
  }

  renderNotifications(notifications) {
    // Clear existing notifications
    while (this.container.firstChild) {
      this.container.firstChild.remove();
    }

    // Render new notifications
    notifications.forEach(notification => {
      const toast = this.createNotificationElement(notification);
      this.container.appendChild(toast);
    });
  }

  createNotificationElement(notification) {
    const toast = document.createElement('div');
    toast.className = `notification-toast notification-${notification.type}`;
    toast.dataset.id = notification.id;
    toast.style.cssText = `
      pointer-events: auto;
      padding: 12px 16px;
      border-radius: 8px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      display: flex;
      align-items: center;
      gap: 12px;
      animation: slideIn 0.3s ease-out;
      min-width: 280px;
      max-width: 400px;
    `;

    // Icon
    const icon = document.createElement('span');
    icon.style.cssText = 'font-size: 18px;';
    
    switch (notification.type) {
      case NOTIFICATION_TYPES.SUCCESS:
        icon.textContent = '✓';
        icon.style.color = 'var(--ok)';
        break;
      case NOTIFICATION_TYPES.WARNING:
        icon.textContent = '⚠';
        icon.style.color = 'var(--warn)';
        break;
      case NOTIFICATION_TYPES.ERROR:
        icon.textContent = '✕';
        icon.style.color = 'var(--err)';
        break;
      default:
        icon.textContent = 'ℹ';
        icon.style.color = 'var(--acc)';
    }
    toast.appendChild(icon);

    // Message
    const message = document.createElement('span');
    message.textContent = notification.message;
    message.style.cssText = `
      font-size: 13px;
      color: var(--fg);
      flex: 1;
    `;
    toast.appendChild(message);

    // Close button (for errors that stay longer)
    if (notification.type === NOTIFICATION_TYPES.ERROR) {
      const close = document.createElement('button');
      close.textContent = '✕';
      close.style.cssText = `
        background: none;
        border: none;
        color: var(--dim);
        font-size: 16px;
        cursor: pointer;
        padding: 0;
        line-height: 1;
      `;
      close.onclick = () => removeNotification(notification.id);
      toast.appendChild(close);
    }

    return toast;
  }

  destroy() {
    if (this.unsubscribe) {
      this.unsubscribe();
    }
    if (this.container) {
      this.container.remove();
    }
  }
}

// ============================================================================
// Alert Dialogs
// ============================================================================

class AlertManager {
  constructor() {
    this.container = null;
    this.init();
  }

  init() {
    // Container is already in the HTML
    this.container = document.getElementById('modal');
  }

  /**
   * Show a simple alert dialog
   */
  alert(message, options = {}) {
    return new Promise((resolve) => {
      const modalBody = document.getElementById('modalbody');
      if (!modalBody) return resolve();

      modalBody.innerHTML = `
        <b>${options.title || 'Alert'}</b>
        <div style="margin-top: 8px; color: var(--fg); font-size: 13px; line-height: 1.6;">
          ${message}
        </div>
        <div class="row" style="margin-top: 12px;">
          <button onclick="resolve()" style="margin-right: 8px;">OK</button>
        </div>
      `;

      this.container.style.display = 'flex';
      
      // Auto-close on OK click
      const button = modalBody.querySelector('button');
      button.onclick = () => {
        this.container.style.display = 'none';
        resolve();
      };
    });
  }

  /**
   * Show a confirmation dialog
   */
  confirm(message, options = {}) {
    return new Promise((resolve) => {
      const modalBody = document.getElementById('modalbody');
      if (!modalBody) return resolve(false);

      modalBody.innerHTML = `
        <b>${options.title || 'Confirm'}</b>
        <div style="margin-top: 8px; color: var(--fg); font-size: 13px; line-height: 1.6;">
          ${message}
        </div>
        <div class="row" style="margin-top: 12px;">
          <button onclick="resolve(true)" style="margin-right: 8px;" class="primary">${options.confirmText || 'Confirm'}</button>
          <button onclick="resolve(false)">${options.cancelText || 'Cancel'}</button>
        </div>
      `;

      this.container.style.display = 'flex';
      
      // Setup buttons
      const buttons = modalBody.querySelectorAll('button');
      buttons[0].onclick = () => {
        this.container.style.display = 'none';
        resolve(true);
      };
      buttons[1].onclick = () => {
        this.container.style.display = 'none';
        resolve(false);
      };
    });
  }

  /**
   * Show a prompt dialog
   */
  prompt(message, options = {}) {
    return new Promise((resolve) => {
      const modalBody = document.getElementById('modalbody');
      if (!modalBody) return resolve(null);

      const value = options.defaultValue || '';
      const placeholder = options.placeholder || '';

      modalBody.innerHTML = `
        <b>${options.title || 'Input'}</b>
        <div style="margin-top: 8px; color: var(--fg); font-size: 13px; line-height: 1.6;">
          ${message}
        </div>
        <input id="prompt-input" type="${options.type || 'text'}" 
               value="${value}" 
               placeholder="${placeholder}"
               style="margin-top: 8px; width: 100%; padding: 8px; border-radius: 6px; border: 1px solid var(--line2); background: var(--inset); color: var(--fg); font: 13px var(--sans);"
               ${options.type === 'number' ? `min="${options.min || ''}" max="${options.max || ''}" step="${options.step || ''}"` : ''}>
        <div class="row" style="margin-top: 12px;">
          <button onclick="resolve(document.getElementById('prompt-input').value)" style="margin-right: 8px;" class="primary">${options.confirmText || 'OK'}</button>
          <button onclick="resolve(null)">${options.cancelText || 'Cancel'}</button>
        </div>
      `;

      this.container.style.display = 'flex';
      
      // Focus input
      const input = document.getElementById('prompt-input');
      input.focus();
      if (options.selectAll) {
        input.select();
      }
      
      // Handle Enter key
      input.onkeydown = (e) => {
        if (e.key === 'Enter') {
          this.container.style.display = 'none';
          resolve(input.value);
        }
      };
      
      // Setup buttons
      const buttons = modalBody.querySelectorAll('button');
      buttons[0].onclick = () => {
        this.container.style.display = 'none';
        resolve(input.value);
      };
      buttons[1].onclick = () => {
        this.container.style.display = 'none';
        resolve(null);
      };
    });
  }
}

// ============================================================================
// Singleton Instances
// ============================================================================

let notificationManager;
let alertManager;

/**
 * Get or create notification manager
 */
export function getNotificationManager() {
  if (!notificationManager) {
    notificationManager = new NotificationManager();
  }
  return notificationManager;
}

/**
 * Get or create alert manager
 */
export function getAlertManager() {
  if (!alertManager) {
    alertManager = new AlertManager();
  }
  return alertManager;
}

/**
 * Show a toast notification
 */
export function showToast(type, message, options = {}) {
  addNotification({ type, message, ...options });
}

/**
 * Show an alert
 */
export function showAlert(message, options = {}) {
  return getAlertManager().alert(message, options);
}

/**
 * Show a confirmation dialog
 */
export function showConfirm(message, options = {}) {
  return getAlertManager().confirm(message, options);
}

/**
 * Show a prompt dialog
 */
export function showPrompt(message, options = {}) {
  return getAlertManager().prompt(message, options);
}

// ============================================================================
// CSS Styles (will be added to the page)
// ============================================================================

export function injectNotificationStyles() {
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideIn {
      from {
        transform: translateX(100%);
        opacity: 0;
      }
      to {
        transform: translateX(0);
        opacity: 1;
      }
    }

    @keyframes slideOut {
      from {
        transform: translateX(0);
        opacity: 1;
      }
      to {
        transform: translateX(100%);
        opacity: 0;
      }
    }
  `;
  document.head.appendChild(style);
}

// Initialize on module load
injectNotificationStyles();
getNotificationManager();
getAlertManager();

export default {
  showToast,
  showAlert,
  showConfirm,
  showPrompt
};
