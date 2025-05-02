document.addEventListener('DOMContentLoaded', function() {
    const inputText = document.getElementById('input-text');
    const summarizeBtn = document.getElementById('summarize-btn');
    const clearBtn = document.getElementById('clear-btn');
    const summaryOutput = document.getElementById('summary-output');

    // Function to show loading animation
    function showLoading() {
        summaryOutput.innerHTML = `
            <div class="loading">
                <div></div>
            </div>
        `;
    }

    // Function to handle the summarization
    async function summarizeText() {
        const text = inputText.value.trim();
        
        if (!text) {
            summaryOutput.textContent = 'Please enter some text to summarize.';
            return;
        }

        try {
            showLoading();
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            if (data.summary) {
                summaryOutput.textContent = data.summary;
                summaryOutput.style.animation = 'fadeIn 0.5s ease-out';
            } else {
                summaryOutput.textContent = 'Error generating summary. Please try again.';
            }
        } catch (error) {
            console.error('Error:', error);
            summaryOutput.textContent = 'An error occurred while generating the summary.';
        }
    }

    // Event Listeners
    summarizeBtn.addEventListener('click', summarizeText);
    
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        summaryOutput.textContent = 'Your summary will appear here...';
    });

    // Add animation when typing
    let typingTimer;
    inputText.addEventListener('input', () => {
        clearTimeout(typingTimer);
        typingTimer = setTimeout(() => {
            if (inputText.value.trim() !== '') {
                summarizeBtn.classList.add('pulse');
                setTimeout(() => summarizeBtn.classList.remove('pulse'), 1000);
            }
        }, 1000);
    });
});