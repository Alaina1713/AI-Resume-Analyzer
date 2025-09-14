function analyzeResume() {
    const text = document.getElementById("resumeText").value;
    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams({'resume_text': text})
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = "<h2>Job-Fit Scores:</h2>";
        for (const [skill, score] of Object.entries(data)) {
            const div = document.createElement("div");
            div.textContent = `${skill}: ${score}%`;
            resultsDiv.appendChild(div);
        }
    });
}
