document.addEventListener('DOMContentLoaded', () => {
    const inputField = document.getElementById('indPut');
    const processButton = document.getElementById('text');
    const outputField = document.getElementById('udPut');
    const loadingSpinner = document.getElementById('loading');
    const historyButton = document.getElementById('history');
    const historyContainer = document.getElementById('historyContainer');
    const historyContent = document.getElementById('historyContent');
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('fileInput');

    fetchHistory();
    historyContainer.style.display = 'none';

    // Function to read file content
    function handleFileUpload(event) {
        loadingSpinner.style.display = 'block';
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            fetch('http://localhost:8000/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let downloadtext = '';
                if (data.length === 0) {
                    downloadtext = 'No translation history available.';
                } else {
                    downloadtext = data.output_string;
                }

                const blob = new Blob([downloadtext], { type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'output.txt';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            })
            .catch(error => {
                console.error('Error:', error);
            }).finally(() => {
                loadingSpinner.style.display = 'none';
            });
        } else {
            console.log('No file selected');
        }
    }

    // Process input and send to server
    function processInput() {
        const inputString = inputField.value;

        loadingSpinner.style.display = 'block';

        fetch(`http://localhost:8000/process_string`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_string: inputString })
        })
        .then(response => response.json())
        .then(data => {
            outputField.textContent = data.output_string;
        })
        .catch(error => {
            console.error('Error:', error);
            outputField.textContent = 'An error occurred.';
        })
        .finally(() => {
            loadingSpinner.style.display = 'none';
            fetchHistory();  // Fetch and display the history
        });
    }

    // Fetch translation history
    function fetchHistory() {
        fetch(`http://localhost:8000/get_history`)
        .then(response => response.json())
        .then(data => {
            historyContent.innerHTML = '';  // Clear previous content

            // Create a list of translations
            if (data.length === 0) {
                historyContent.innerHTML = '<p>No translation history available.</p>';
            } else {
                data.forEach(entry => {
                    const translationEntry = document.createElement('div');
                    translationEntry.style.padding = '5px 0';
                    translationEntry.innerHTML = `
                        <strong>Input:</strong> ${entry.input}<br>
                        <strong>Output:</strong> ${entry.output}<br>
                    `;
                    historyContent.appendChild(translationEntry);
                });
            }
        })
        .catch(error => {
            console.error('Error fetching history:', error);
            historyContent.innerHTML = '<p>An error occurred while fetching the history.</p>';
        });
    }

    // Toggle the history dropdown
    function toggleHistory() {
        if (historyContainer.style.display === 'none') {
            historyContainer.style.display = 'block';  // Show the dropdown
            fetchHistory();  // Fetch and display the history
        } else {
            historyContainer.style.display = 'none';  // Hide the dropdown
        }
    }

    // Add event listeners
    processButton.addEventListener('click', () => {
        processInput();
    });

    inputField.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            processInput();
        }
    });

    // Add event listener for history button to toggle the dropdown
    historyButton.addEventListener('click', () => {
        toggleHistory();
    });

    // Add event listener for file upload button
    uploadButton.addEventListener('click', () => {
        fileInput.click();  // Trigger file input click
    });

    // Handle file upload change
    fileInput.addEventListener('change', handleFileUpload);
});
