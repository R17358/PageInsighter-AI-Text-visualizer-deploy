// ============================================================================
// EMBEDDED CHATBOT WIDGET
// ============================================================================

(function () {
  const chatBtn = document.createElement("button");
  chatBtn.innerText = "💬 Chat";
  chatBtn.style.position = "fixed";
  chatBtn.style.bottom = "20px";
  chatBtn.style.right = "20px";
  chatBtn.style.zIndex = "9999";
  chatBtn.style.padding = "10px 15px";
  chatBtn.style.borderRadius = "8px";
  chatBtn.style.border = "none";
  chatBtn.style.background = "#4f9cff";
  chatBtn.style.color = "#fff";
  chatBtn.style.cursor = "pointer";

  document.body.appendChild(chatBtn);

  let iframe = null;

  chatBtn.onclick = function () {
    if (iframe) {
      iframe.remove();
      iframe = null;
      return;
    }

    iframe = document.createElement("iframe");
    iframe.src = "https://page-insighter.vercel.app"; // your deployed frontend
    iframe.style.position = "fixed";
    iframe.style.bottom = "80px";
    iframe.style.right = "20px";
    iframe.style.width = "350px";
    iframe.style.height = "500px";
    iframe.style.border = "none";
    iframe.style.borderRadius = "12px";
    iframe.style.boxShadow = "0 0 20px rgba(0,0,0,0.3)";
    iframe.style.zIndex = "9999";

    document.body.appendChild(iframe);
  };
})();
