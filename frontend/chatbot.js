// Advanced Chat Widget v2
const ChatWidget = (function () {
  let isOpen = false;
  let unread = 0;

  // Config
  const CFG = {
    src: 'https://page-insighter.vercel.app',
    accentColor: '#667eea',
    position: 'bottom-right',
    greeting: 'Hi! I am PageInsighter',
    agentName: 'Support',
    showNotif: true,
  };

  // Inject styles
  const style = document.createElement('style');
  style.textContent = `
    #cw-btn {
      position: fixed;
      bottom: 24px; right: 24px;
      width: 56px; height: 56px;
      border-radius: 50%; border: none;
      background: linear-gradient(135deg,
        ${CFG.accentColor}, #764ba2);
      cursor: pointer; z-index: 99999;
      box-shadow: 0 4px 20px
        ${CFG.accentColor}60;
      transition: transform .25s
        cubic-bezier(.34,1.56,.64,1);
    }
    #cw-btn:hover { transform:scale(1.12); }
    #cw-frame {
      position: fixed;
      bottom: 92px; right: 20px;
      width: 360px; height: 520px;
      border: none; border-radius: 18px;
      z-index: 99998;
      box-shadow: 0 12px 48px
        rgba(0,0,0,.18);
      opacity: 0;
      transform: translateY(16px)
        scale(.96);
      pointer-events: none;
      transition: opacity .3s
        cubic-bezier(.22,1,.36,1),
        transform .3s
        cubic-bezier(.22,1,.36,1);
    }
    #cw-frame.open {
      opacity:1;
      transform:translateY(0) scale(1);
      pointer-events:all;
    }
    #cw-badge {
      position:absolute;
      top:-4px; right:-4px;
      width:20px; height:20px;
      background:#f04;
      border-radius:50%;
      font:700 10px system-ui;
      color:#fff; border:2px solid #fff;
      display:flex;
      align-items:center;
      justify-content:center;
      animation: cwpulse 2s infinite;
    }
    @keyframes cwpulse {
      0%  { box-shadow:0 0 0 0
              rgba(255,0,68,.4); }
      70% { box-shadow:0 0 0 8px
              rgba(255,0,68,0); }
    }
  `;
  document.head.appendChild(style);

  // Build DOM
  const wrap =
    document.createElement('div');
  wrap.style.cssText =
    'position:fixed;z-index:99999';

  const btn =
    document.createElement('button');
  btn.id = 'cw-btn';
  btn.innerHTML = getIcon('chat');

  const badge =
    document.createElement('div');
  badge.id = 'cw-badge';
  badge.textContent = '1';
  btn.appendChild(badge);

  const frame =
    document.createElement('iframe');
  frame.id = 'cw-frame';
  frame.src = CFG.src;
  frame.setAttribute('allow',
    'microphone; camera');

  wrap.appendChild(frame);
  wrap.appendChild(btn);
  document.body.appendChild(wrap);

  // Toggle
  function toggle() {
    isOpen = !isOpen;
    frame.classList.toggle(
      'open', isOpen);
    btn.innerHTML = isOpen
      ? getIcon('close')
      : getIcon('chat');
    if (isOpen) {
      unread = 0;
      badge.style.display = 'none';
      frame.contentWindow
        ?.postMessage(
          { type: 'CW_OPEN' }, '*');
    }
  }
  btn.addEventListener(
    'click', toggle);

  // Listen for events
  window.addEventListener(
    'message', (e) => {
    if (e.data?.type ===
        'CW_CLOSE') toggle();
    if (e.data?.type ===
        'CW_UNREAD') {
      unread++;
      badge.textContent = unread;
      badge.style.display = 'flex';
    }
  });

  return { toggle, frame };
})();
