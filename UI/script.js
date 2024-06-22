document.getElementById("openChatBtn").addEventListener("click", function() {
    document.getElementById("chatPopup").style.display = "flex";
});

document.getElementById("closeChatBtn").addEventListener("click", function() {
    document.getElementById("chatPopup").style.display = "none";
});

document.getElementById("sendBtn").addEventListener("click", function() {
    sendMessage();
});

document.getElementById("userInput").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (userInput.trim() !== "") {
        const chatBody = document.getElementById("chatBody");
        const userMessage = document.createElement("div");
        userMessage.className = "message user-message";
        userMessage.innerText = userInput;
        chatBody.appendChild(userMessage);
        chatBody.scrollTop = chatBody.scrollHeight;
        document.getElementById("userInput").value = "";
        
        // Simulate bot response
        setTimeout(function() {
            const botMessage = document.createElement("div");
            botMessage.className = "message bot-message";
            botMessage.innerText = "This is a response from the bot.";
            chatBody.appendChild(botMessage);
            chatBody.scrollTop = chatBody.scrollHeight;
        }, 1000);
    }
}
