const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");
const sendButton = form.querySelector(
  'button[type="submit"], input[type="submit"], button:not([type])'
);
const restartButton = document.getElementById("restart-btn");

let sessionId = null;
let busy = false; 

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(value) {
  busy = value;
  if (sendButton) sendButton.disabled = value;
  restartButton.disabled = value;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;

  const text = input.value.trim();
  if (!text) return;

  setBusy(true);
  addMessage(text, "user");
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text
      })
    });

    if (!response.ok) {
      addMessage(`Chat request failed (${response.status}).`, "bot");
      return;
    }

    const data = await response.json();
    sessionId = data.session_id;
    addMessage(data.reply, "bot");
  } catch {
    addMessage("Could not complete the chat request.", "bot");
  } finally {
    setBusy(false);
    input.focus();
  }
});

restartButton.addEventListener("click", async () => {
  if (busy || !window.confirm("Clear all active chat sessions?")) return;

  setBusy(true);

  try {
    const response = await fetch("/reset-all", { method: "POST" });

    if (!response.ok) {
      addMessage(`Restart failed (${response.status}).`, "bot");
      return;
    }

    sessionId = null;
    messages.replaceChildren();
    input.value = "";
  } catch {
    addMessage("Could not confirm that the server cleared the sessions.", "bot");
  } finally {
    setBusy(false);
    input.focus();
  }
});