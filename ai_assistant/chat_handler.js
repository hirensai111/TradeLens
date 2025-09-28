class TradeLensChat {
    constructor() {
        this.chatWidget = document.getElementById('chat-widget');
        this.chatToggle = document.getElementById('chat-toggle');
        this.chatInput = document.getElementById('chat-input');
        this.sendButton = document.getElementById('send-button');
        this.chatMessages = document.getElementById('chat-messages');
        this.typingIndicator = document.getElementById('typing-indicator');
        this.clearChatBtn = document.getElementById('clear-chat');
        this.chatClose = document.getElementById('chat-close');

        this.messageHistory = [];
        this.currentStockSymbol = this.getCurrentStockSymbol();
        this.maxMessages = 10;

        this.initializeEventListeners();
        this.loadChatHistory();
    }

    initializeEventListeners() {
        // Toggle chat widget
        this.chatToggle.addEventListener('click', () => this.toggleChat());
        this.chatClose.addEventListener('click', () => this.closeChat());

        // Send message events
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Quick action chips
        document.querySelectorAll('.quick-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const message = chip.getAttribute('data-message');
                this.chatInput.value = message;
                this.sendMessage();
            });
        });

        // Clear chat
        this.clearChatBtn.addEventListener('click', () => this.clearChat());

        // Auto-focus input when chat opens
        this.chatInput.addEventListener('focus', () => {
            this.updateQuickActions();
        });
    }

    toggleChat() {
        const isHidden = this.chatWidget.classList.contains('hidden');
        if (isHidden) {
            this.openChat();
        } else {
            this.closeChat();
        }
    }

    openChat() {
        this.chatWidget.classList.remove('hidden');
        this.chatInput.focus();
        this.updateQuickActions();
        this.scrollToBottom();
    }

    closeChat() {
        this.chatWidget.classList.add('hidden');
    }

    getCurrentStockSymbol() {
        // Try to get current stock symbol from various sources
        // This should be customized based on your dashboard implementation
        const urlParams = new URLSearchParams(window.location.search);
        const symbolFromUrl = urlParams.get('symbol');

        if (symbolFromUrl) return symbolFromUrl.toUpperCase();

        // Try to get from page title or other elements
        const titleMatch = document.title.match(/([A-Z]{1,5})/);
        if (titleMatch) return titleMatch[1];

        // Try to get from any element with stock symbol
        const symbolElement = document.querySelector('[data-symbol]');
        if (symbolElement) return symbolElement.getAttribute('data-symbol').toUpperCase();

        // Default fallback
        return 'AAPL';
    }

    async getCurrentStockContext() {
        try {
            // This should integrate with your existing stock data
            // For now, we'll simulate getting current context
            const response = await fetch(`/api/stock-data/${this.currentStockSymbol}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (error) {
            console.error('Error getting stock context:', error);
        }

        // Fallback context structure
        return {
            symbol: this.currentStockSymbol,
            price: 220.45,
            rsi: 62.5,
            macd: 0.75,
            dayChange: 1.2,
            volume: 52000000,
            marketCap: '3.4T',
            peRatio: 28.5
        };
    }

    updateQuickActions() {
        const quickChips = document.querySelectorAll('.quick-chip');
        const context = this.getCurrentStockContext();

        // Update quick action suggestions based on current context
        context.then(data => {
            const suggestions = this.getSmartSuggestions(data);
            quickChips.forEach((chip, index) => {
                if (suggestions[index]) {
                    chip.textContent = suggestions[index];
                    chip.setAttribute('data-message', suggestions[index]);
                }
            });
        });
    }

    getSmartSuggestions(context) {
        const suggestions = [];

        if (context.rsi > 70) {
            suggestions.push("Is it overbought?");
        } else if (context.rsi < 30) {
            suggestions.push("Is it oversold?");
        } else {
            suggestions.push("What's the trend?");
        }

        if (context.dayChange > 2) {
            suggestions.push("Why is it up today?");
        } else if (context.dayChange < -2) {
            suggestions.push("Why is it down today?");
        } else {
            suggestions.push("Should I buy?");
        }

        suggestions.push("Risk analysis");
        suggestions.push("Compare with sector");

        return suggestions;
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message) return;

        // Disable input while processing
        this.chatInput.disabled = true;
        this.sendButton.disabled = true;

        // Clear input
        this.chatInput.value = '';

        // Add user message to chat
        this.addMessage(message, 'user');

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get current stock context
            const context = await this.getCurrentStockContext();

            // Send to AI backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    context: context,
                    history: this.getRecentHistory()
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.hideTypingIndicator();
                this.addMessage(data.response, 'ai');
            } else {
                throw new Error('Failed to get AI response');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage("Sorry, I'm having trouble right now. Please try again.", 'ai');
        } finally {
            // Re-enable input
            this.chatInput.disabled = false;
            this.sendButton.disabled = false;
            this.chatInput.focus();
        }
    }

    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const timestamp = new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });

        if (type === 'ai') {
            messageDiv.innerHTML = `
                <div class="message-avatar">TL</div>
                <div class="message-content">
                    <p>${content}</p>
                    <span class="message-time">${timestamp}</span>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <p>${content}</p>
                    <span class="message-time">${timestamp}</span>
                </div>
            `;
        }

        this.chatMessages.appendChild(messageDiv);

        // Add to message history
        this.messageHistory.push({
            content: content,
            type: type,
            timestamp: Date.now()
        });

        // Limit message history
        if (this.messageHistory.length > this.maxMessages) {
            this.messageHistory.shift();
        }

        this.scrollToBottom();
        this.saveChatHistory();
    }

    showTypingIndicator() {
        this.typingIndicator.classList.remove('hidden');
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.typingIndicator.classList.add('hidden');
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    clearChat() {
        this.chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-avatar">TL</div>
                <div class="message-content">
                    <p>Hi! I'm TradeLens AI, your personal trading assistant. I can help analyze stocks, explain indicators, and provide trading insights based on real-time data.</p>
                    <span class="message-time">Just now</span>
                </div>
            </div>
        `;
        this.messageHistory = [];
        this.saveChatHistory();
    }

    getRecentHistory() {
        return this.messageHistory.slice(-6); // Last 6 messages for context
    }

    saveChatHistory() {
        try {
            localStorage.setItem('tradeLensChat', JSON.stringify(this.messageHistory));
        } catch (error) {
            console.error('Error saving chat history:', error);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('tradeLensChat');
            if (saved) {
                this.messageHistory = JSON.parse(saved);
                this.restoreChatMessages();
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }

    restoreChatMessages() {
        // Clear existing messages except welcome message
        this.chatMessages.innerHTML = `
            <div class="message ai-message">
                <div class="message-avatar">TL</div>
                <div class="message-content">
                    <p>Hi! I'm TradeLens AI, your personal trading assistant. I can help analyze stocks, explain indicators, and provide trading insights based on real-time data.</p>
                    <span class="message-time">Just now</span>
                </div>
            </div>
        `;

        // Restore messages from history
        this.messageHistory.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.type}-message`;

            const timestamp = new Date(msg.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });

            if (msg.type === 'ai') {
                messageDiv.innerHTML = `
                    <div class="message-avatar">TL</div>
                    <div class="message-content">
                        <p>${msg.content}</p>
                        <span class="message-time">${timestamp}</span>
                    </div>
                `;
            } else {
                messageDiv.innerHTML = `
                    <div class="message-content">
                        <p>${msg.content}</p>
                        <span class="message-time">${timestamp}</span>
                    </div>
                `;
            }

            this.chatMessages.appendChild(messageDiv);
        });

        this.scrollToBottom();
    }

    // Public method to update stock context when user navigates
    updateStockContext(newSymbol) {
        this.currentStockSymbol = newSymbol.toUpperCase();
        this.updateQuickActions();
    }

    // Public method to integrate with your dashboard
    static init(stockSymbol = null) {
        const chat = new TradeLensChat();
        if (stockSymbol) {
            chat.updateStockContext(stockSymbol);
        }
        return chat;
    }
}

// Auto-initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.tradeLensChat = TradeLensChat.init();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradeLensChat;
}