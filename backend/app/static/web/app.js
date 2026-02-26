const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const tapMarker = document.getElementById('tapMarker');
const generateButton = document.getElementById('generateButton');
const statusEl = document.getElementById('status');
const output = document.getElementById('output');
const paragraphEl = document.getElementById('paragraph');
const deepenButton = document.getElementById('deepenButton');
const surfaceButton = document.getElementById('surfaceButton');

let selectedFile = null;
let traceId = null;
let tap = null;
let isGenerating = false;

function setStatus(text) {
  statusEl.textContent = text;
}

function clearOutput() {
  output.hidden = true;
  paragraphEl.textContent = '';
  traceId = null;
}

function updateGenerateState() {
  generateButton.disabled = !(selectedFile && tap) || isGenerating;
}

async function fetchAiReply(text) {
  const response = await fetch('/api/ai/respond', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => null);
    const detail = errorPayload?.detail || 'AI response failed.';
    throw new Error(detail);
  }

  const payload = await response.json();
  return payload.reply;
}

async function submitGenerate() {
  if (!selectedFile || !tap || isGenerating) {
    return;
  }

  isGenerating = true;
  setStatus('Generating…');
  updateGenerateState();

  try {
    const form = new FormData();
    form.append('tap_x', String(tap.x));
    form.append('tap_y', String(tap.y));
    form.append('image', selectedFile, selectedFile.name || 'upload.jpg');

    const response = await fetch('/generate', { method: 'POST', body: form });
    if (!response.ok) {
      setStatus('Generation failed.');
      return;
    }

    const payload = await response.json();
    const aiReply = await fetchAiReply(payload.paragraph_text);
    paragraphEl.textContent = aiReply;
    traceId = payload.trace_id;
    output.hidden = false;
    setStatus('Generated.');
  } catch (error) {
    setStatus('Generation failed.');
  } finally {
    isGenerating = false;
    updateGenerateState();
  }
}

function placeTapMarker(clientX, clientY) {
  const rect = preview.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  tap = {
    x: Math.max(0, Math.min(1, x / rect.width)),
    y: Math.max(0, Math.min(1, y / rect.height)),
  };

  tapMarker.hidden = false;
  tapMarker.style.left = `${x}px`;
  tapMarker.style.top = `${y}px`;
  updateGenerateState();
}

imageInput.addEventListener('change', () => {
  const file = imageInput.files?.[0];
  if (!file) {
    return;
  }

  selectedFile = file;
  tap = null;
  tapMarker.hidden = true;
  updateGenerateState();
  clearOutput();

  const objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  preview.hidden = false;
  setStatus('Tap a point on the image, then generate.');
});

preview.addEventListener('click', (event) => {
  placeTapMarker(event.clientX, event.clientY);
  submitGenerate();
});

generateButton.addEventListener('click', submitGenerate);

deepenButton.addEventListener('click', async () => {
  if (!traceId) {
    return;
  }

  setStatus('Deepening…');
  deepenButton.disabled = true;

  const response = await fetch('/deepen', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trace_id: traceId }),
  });

  deepenButton.disabled = false;
  if (!response.ok) {
    setStatus('Deepen failed.');
    return;
  }

  const payload = await response.json();
  const aiReply = await fetchAiReply(payload.paragraph_text);
  paragraphEl.textContent = aiReply;
  traceId = payload.trace_id;
  setStatus('Deeper.');
});

surfaceButton.addEventListener('click', () => {
  clearOutput();
  setStatus('Returned to surface.');
});
