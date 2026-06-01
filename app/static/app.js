const form = document.getElementById("chat-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");

let sessionId = null;

function addMessage(text, role) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = input.value.trim();
  if (!text) return;

  addMessage(text, "user");
  input.value = "";

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session_id: sessionId,
        message: text
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      addMessage(`Error: ${errorText}`, "bot");
      return;
    }

    const data = await response.json();
    sessionId = data.session_id;
    addMessage(data.reply, "bot");
  } catch (error) {
    addMessage("Network error. Could not reach the server.", "bot");
  }
});